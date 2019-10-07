import numpy as np
import os
from PIL import Image
import tqdm
import torch
import torchvision
from sotabenchapi.check import in_check_mode
from sotabenchapi.client import Client
import time

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from torchbench.utils import calculate_run_hash, AverageMeter
from torchbench.datasets import CocoDetection

from .coco_eval import CocoEvaluator
from .voc_eval import (
    get_voc_results_file_template,
    voc_eval,
    write_voc_results_file,
    VOC_CLASSES,
)


def convert_to_coco_api(ds):
    coco_ds = COCO()
    ann_id = 0
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"]
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(
        model_without_ddp, torchvision.models.detection.KeypointRCNN
    ):
        iou_types.append("keypoints")
    return iou_types


def get_coco_metrics(coco_evaluator):

    metrics = {
        "box AP": None,
        "AP50": None,
        "AP75": None,
        "APS": None,
        "APM": None,
        "APL": None,
    }
    iouThrs = [None, 0.5, 0.75, None, None, None]
    maxDets = [100] + [coco_evaluator.coco_eval["bbox"].params.maxDets[2]] * 5
    areaRngs = ["all", "all", "all", "small", "medium", "large"]
    bounding_box_params = coco_evaluator.coco_eval["bbox"].params

    for metric_no, metric in enumerate(metrics):
        aind = [
            i
            for i, aRng in enumerate(bounding_box_params.areaRngLbl)
            if aRng == areaRngs[metric_no]
        ]
        mind = [
            i
            for i, mDet in enumerate(bounding_box_params.maxDets)
            if mDet == maxDets[metric_no]
        ]

        s = coco_evaluator.coco_eval["bbox"].eval["precision"]

        # IoU
        if iouThrs[metric_no] is not None:
            t = np.where(iouThrs[metric_no] == bounding_box_params.iouThrs)[0]
            s = s[t]
        s = s[:, :, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        metrics[metric] = mean_s

    return metrics


def get_voc_metrics(box_list, data_loader, labelmap):

    dataset_year = data_loader.dataset.year
    data_root = data_loader.dataset.root

    write_voc_results_file(
        box_list, data_loader, data_root, dataset_year, labelmap
    )

    annopath = os.path.join(
        data_root, "VOCdevkit", "VOC" + str(dataset_year), "Annotations"
    )
    cachedir = os.path.join(
        data_root, "VOCdevkit", "VOC" + str(dataset_year), "annotations_cache"
    )
    imgsetpath = os.path.join(
        data_root,
        "VOCdevkit/",
        "VOC" + str(dataset_year),
        "ImageSets",
        "Main",
        "{:s}.txt",
    )
    aps = []

    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(
            data_root, dataset_year, "val", cls
        )
        rec, prec, ap = voc_eval(
            filename,
            annopath,
            imgsetpath.format("val"),
            cls,
            cachedir,
            ovthresh=0.5,
            use_07_metric=True,
        )
        aps += [ap]

    return {"Mean AP": np.mean(aps)}


def evaluate_detection_coco(
    model,
    test_loader,
    model_output_transform,
    send_data_to_device,
    device="cuda",
):

    coco = get_coco_api_from_dataset(test_loader.dataset)
    iou_types = ['bbox']
    coco_evaluator = CocoEvaluator(coco, iou_types)

    iterator = tqdm.tqdm(test_loader, desc="Evaluation", mininterval=5)

    init_time = time.time()

    with torch.no_grad():
        for i, (input, target) in enumerate(iterator):
            input, target = send_data_to_device(input, target, device=device)
            original_output = model(input)
            output, target = model_output_transform(original_output, target)

            result = {
                tar["image_id"].item(): out for tar, out in zip(target, output)
            }
            coco_evaluator.update(result)

            if i == 0:  # for sotabench.com caching of evaluation
                run_hash = calculate_run_hash([], original_output)
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
                        'Tasks / Evaluation Time': None,
                        'Evaluation Time': None,
                        'Tasks': None,
                        'Max Memory Allocated (Total)': None,
                    }

                    return cached_res, speed_mem_metrics, run_hash

    exec_time = (time.time() - init_time)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    memory_allocated = torch.cuda.max_memory_allocated(device=device)
    torch.cuda.reset_max_memory_allocated(device=device)

    speed_mem_metrics = {
        'Tasks / Evaluation Time': len(test_loader.dataset) / exec_time,
        'Tasks': len(test_loader.dataset),
        'Evaluation Time': (time.time() - init_time),
        'Max Memory Allocated (Total)': memory_allocated,
    }

    return (get_coco_metrics(coco_evaluator), speed_mem_metrics, run_hash)


def evaluate_detection_voc(
    model,
    test_loader,
    model_output_transform,
    send_data_to_device,
    device="cuda",
):
    """Evaluates detection ability on VOC.

    All detections are collected into N x 5 arrays of detections with
    (x1, y1, x2, y2, score).
    """

    num_images = len(test_loader.dataset)
    all_boxes = [
        [[] for _ in range(num_images)] for _ in range(len(VOC_CLASSES) + 1)
    ]

    for i in range(num_images):

        input, target = test_loader.dataset[i]
        input, target = send_data_to_device(input, target, device=device)
        height, width, channels = np.array(
            Image.open(test_loader.dataset.images[i]).convert("RGB")
        ).shape
        output = model(input)
        output = model_output_transform(output, target)

        # skip j = 0, because it's the background class
        for j in range(1, output.size(1)):
            dets = output[0, j, :]
            mask = dets[:, 0].gt(0.0).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= width
            boxes[:, 2] *= width
            boxes[:, 1] *= height
            boxes[:, 3] *= height
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack(
                (boxes.cpu().numpy(), scores[:, np.newaxis])
            ).astype(np.float32, copy=False)
            all_boxes[j][i] = cls_dets

        print("im_detect: {:d}/{:d}".format(i + 1, num_images))

    return get_voc_metrics(all_boxes, test_loader, VOC_CLASSES)
