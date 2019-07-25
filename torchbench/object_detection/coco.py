import os
from torch.utils.data import DataLoader

from sotabenchapi.core import BenchmarkResult
from torchbench.datasets import CocoDetection
from torchbench.utils import send_model_to_device

from .transforms import Compose, ConvertCocoPolysToMask, ToTensor
from .utils import evaluate_detection_coco


def coco_data_to_device(input, target, device: str = 'cuda', non_blocking: bool = True):
    input = list(inp.to(device=device, non_blocking=non_blocking) for inp in input)
    target = [{k: v.to(device=device, non_blocking=non_blocking) for k, v in t.items()} for t in target]
    return input, target


def coco_collate_fn(batch):
    return tuple(zip(*batch))


def coco_output_transform(output, target):
    output = [{k: v.to('cpu') for k, v in t.items()} for t in output]
    return output, target


class COCO:

    dataset = CocoDetection
    transforms = Compose([ConvertCocoPolysToMask(), ToTensor()])
    send_data_to_device = coco_data_to_device
    collate_fn = coco_collate_fn
    model_output_transform = coco_output_transform

    @classmethod
    def benchmark(cls, model, dataset_year='2017', input_transform=None, target_transform=None, transforms=None,
                  model_output_transform=None, collate_fn=None, send_data_to_device=None, device: str = 'cuda',
                  data_root: str = './.data/vision/coco', num_workers: int = 4, batch_size: int = 1,
                  num_gpu: int = 1, paper_model_name: str = None, paper_arxiv_id: str = None, paper_pwc_id: str = None,
                  pytorch_hub_url: str = None) -> BenchmarkResult:

        config = locals()
        model, device = send_model_to_device(model, device=device, num_gpu=num_gpu)
        model.eval()

        if not input_transform or target_transform or transforms:
            transforms = cls.transforms

        if not model_output_transform:
            model_output_transform = cls.model_output_transform

        if not send_data_to_device:
            send_data_to_device = cls.send_data_to_device

        if not collate_fn:
            collate_fn = cls.collate_fn

        test_dataset = cls.dataset(root=os.path.join(data_root, 'val%s' % dataset_year),
                                   annFile=os.path.join(data_root, 'annotations/instances_val%s.json' % dataset_year),
                                   transform=input_transform, target_transform=target_transform, transforms=transforms, download=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                                 collate_fn=collate_fn)
        test_loader.no_classes = 91  # Number of classes for COCO Detection
        test_results = evaluate_detection_coco(model=model, test_loader=test_loader, model_output_transform=model_output_transform,
                                               send_data_to_device=send_data_to_device, device=device)

        print(test_results)

        return BenchmarkResult(task="Object Detection", benchmark=cls, config=config, dataset=test_dataset,
                               results=test_results, pytorch_hub_id=pytorch_hub_url,
                               model=paper_model_name, arxiv_id=paper_arxiv_id,
                               pwc_id=paper_pwc_id)
