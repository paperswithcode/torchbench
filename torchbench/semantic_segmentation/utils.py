import cv2
import numpy as np
import tqdm
import time
import torch
from albumentations.core.transforms_interface import DualTransform
from PIL import Image
from sotabenchapi.check import in_check_mode
from sotabenchapi.client import Client

from torchbench.utils import calculate_run_hash, AverageMeter


def minmax_normalize(img, norm_range=(0, 1), orig_range=(0, 255)):
    # range(0, 1)
    norm_img = (img - orig_range[0]) / (orig_range[1] - orig_range[0])
    # range(min_value, max_value)
    norm_img = norm_img * (norm_range[1] - norm_range[0]) + norm_range[0]
    return norm_img


class FlipChannels(object):
    def __call__(self, img):
        img = np.array(img)[:, :, ::-1]
        return Image.fromarray(img.astype(np.uint8))


class PadIfNeededRightBottom(DualTransform):
    def __init__(
        self,
        min_height=769,
        min_width=769,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        ignore_index=255,
        always_apply=False,
        p=1.0,
    ):
        super().__init__(always_apply, p)
        self.min_height = min_height
        self.min_width = min_width
        self.border_mode = border_mode
        self.value = value
        self.ignore_index = ignore_index

    def apply(self, img, **params):
        img_height, img_width = img.shape[:2]
        pad_height = max(0, self.min_height - img_height)
        pad_width = max(0, self.min_width - img_width)
        return np.pad(
            img,
            ((0, pad_height), (0, pad_width), (0, 0)),
            "constant",
            constant_values=self.value,
        )

    def apply_to_mask(self, img, **params):
        img_height, img_width = img.shape[:2]
        pad_height = max(0, self.min_height - img_height)
        pad_width = max(0, self.min_width - img_width)
        return np.pad(
            img,
            ((0, pad_height), (0, pad_width)),
            "constant",
            constant_values=self.ignore_index,
        )


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            "global correct: {:.1f}\n"
            "average row correct: {}\n"
            "IoU: {}\n"
            "mean IoU: {:.1f}"
        ).format(
            acc_global.item() * 100,
            ["{:.1f}".format(i) for i in (acc * 100).tolist()],
            ["{:.1f}".format(i) for i in (iu * 100).tolist()],
            iu.mean().item() * 100,
        )


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs


def default_seg_collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets


def default_seg_output_transform(output, target):
    return output["out"].argmax(1).flatten(), target.flatten()


def evaluate_segmentation(
    model,
    test_loader,
    model_output_transform,
    send_data_to_device,
    device="cuda",
):
    confmat = ConfusionMatrix(test_loader.no_classes)

    inference_time = AverageMeter()

    iterator = tqdm.tqdm(test_loader, desc="Evaluation", mininterval=5)

    end = time.time()

    with torch.no_grad():
        for i, (input, target) in enumerate(iterator):
            input, target = send_data_to_device(input, target, device=device)
            output = model(input)

            inference_time.update(time.time() - end)

            output, target = model_output_transform(output, target)
            confmat.update(target, output)

            if i == 0:  # for sotabench.com caching of evaluation
                partial_memory_allocated = torch.cuda.max_memory_allocated(device=device)
                partial_tps = test_loader.batch_size / inference_time.avg
                run_hash = calculate_run_hash([], output)
                # if we are in check model we don't need to go beyond the first
                # batch
                if in_check_mode():
                    iterator.close()
                    break

                # get the cached values from sotabench.com if available
                client = Client.public()
                cached_res = client.get_results_by_run_hash(run_hash)
                if cached_res:
                    iterator.close()
                    print(
                        "No model change detected (using the first batch run "
                        "hash). Returning cached results."
                    )

                    speed_mem_metrics = {
                        'Tasks Per Second (Partial)': partial_tps,
                        'Tasks Per Second (Total)': None,
                        'Memory Allocated (Partial)': None,
                        'Memory Allocated (Partial)': partial_memory_allocated
                    }

                    return cached_res, speed_mem_metrics, run_hash

            end = time.time()

    acc_global, acc, iu = confmat.compute()

    memory_allocated = torch.cuda.max_memory_allocated(device=device)
    torch.cuda.reset_max_memory_allocated(device=device)

    speed_mem_metrics = {
        'Tasks Per Second (Total)': test_loader.batch_size / inference_time.avg,
        'Tasks Per Second (Partial)': partial_tps,
        'Max Memory Allocated (Total)': memory_allocated,
        'Max Memory Allocated (Partial)': partial_memory_allocated,
        'Evaluation Time': test_loader.batch_size / inference_time.avg,
    }

    return {
        "Accuracy": acc_global.item(),
        "Mean IOU": iu.mean().item()}, \
           speed_mem_metrics, run_hash
