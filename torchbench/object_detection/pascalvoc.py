import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from sotabenchapi.core import BenchmarkResult
from torchbench.utils import send_model_to_device

from torchbench.object_detection.transforms import (
    Normalize,
    Compose,
    ImageResize,
    VOCAnnotationTransform,
    ToTensor,
)
from torchbench.object_detection.utils import evaluate_detection_voc


def voc_data_to_device(
    input, target, device: str = "cuda", non_blocking: bool = True
):
    input = input.unsqueeze(0)
    input = input.to(device=device, non_blocking=non_blocking)
    return input, target


def voc_output_transform(output, target):
    return output.data, target


def voc_collate_fn(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


class PASCALVOC:

    dataset = datasets.VOCDetection
    normalize = Normalize(mean=[0.408, 0.459, 0.482])
    input_transform = Compose(
        [
            ImageResize((320, 320)),
            ToTensor(),
            normalize,
            VOCAnnotationTransform(),
        ]
    )
    send_data_to_device = voc_data_to_device
    collate_fn = voc_collate_fn
    model_output_transform = voc_output_transform
    task = "Object Detection"

    @classmethod
    def benchmark(
        cls,
        model,
        model_description=None,
        dataset_year="2007",
        input_transform=None,
        target_transform=None,
        transforms=None,
        model_output_transform=None,
        collate_fn=None,
        send_data_to_device=None,
        device: str = "cuda",
        data_root: str = "./.data/vision/voc",
        num_workers: int = 4,
        batch_size: int = 32,
        num_gpu: int = 1,
        paper_model_name: str = None,
        paper_arxiv_id: str = None,
        paper_pwc_id: str = None,
        paper_results: dict = None,
        pytorch_hub_url: str = None,
    ) -> BenchmarkResult:

        config = locals()
        model, device = send_model_to_device(
            model, device=device, num_gpu=num_gpu
        )
        model.eval()

        if not input_transform or target_transform or transforms:
            input_transform = cls.input_transform
            target_transform = cls.target_transform

        if not model_output_transform:
            model_output_transform = cls.model_output_transform

        if not send_data_to_device:
            send_data_to_device = cls.send_data_to_device

        if not collate_fn:
            collate_fn = cls.collate_fn

        test_dataset = cls.dataset(
            root=data_root,
            image_set="val",
            year=dataset_year,
            transform=input_transform,
            target_transform=target_transform,
            transforms=transforms,
            download=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        test_loader.no_classes = 21  # Number of classes for PASCALVoc
        test_results = evaluate_detection_voc(
            model=model,
            test_loader=test_loader,
            model_output_transform=model_output_transform,
            send_data_to_device=send_data_to_device,
            device=device,
        )
        print(test_results)

        return BenchmarkResult(
            task=cls.task,
            config=config,
            dataset=cls.dataset.__name__,
            results=test_results,
            pytorch_hub_id=pytorch_hub_url,
            model=paper_model_name,
            model_description=model_description,
            arxiv_id=paper_arxiv_id,
            pwc_id=paper_pwc_id,
            paper_results=paper_results,
        )
