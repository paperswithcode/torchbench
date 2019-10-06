from torch.utils.data import DataLoader

from sotabenchapi.core import BenchmarkResult
from torchbench.datasets import CamVid
from torchbench.utils import default_data_to_device, send_model_to_device

from torchbench.semantic_segmentation.transforms import (
    Normalize,
    ToTensor,
    Compose,
)
from torchbench.semantic_segmentation.utils import (
    default_seg_collate_fn,
    default_seg_output_transform,
    evaluate_segmentation,
)


class CamVid:

    dataset = CamVid
    normalize = Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transforms = Compose([ToTensor(), normalize])
    send_data_to_device = default_data_to_device
    collate_fn = default_seg_collate_fn
    model_output_transform = default_seg_output_transform
    task = "Semantic Segmentation"

    @classmethod
    def benchmark(
        cls,
        model,
        model_description=None,
        input_transform=None,
        target_transform=None,
        transforms=None,
        model_output_transform=None,
        collate_fn=None,
        send_data_to_device=None,
        device: str = "cuda",
        data_root: str = "./.data/vision/camvid",
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
            transforms = cls.transforms

        if not model_output_transform:
            model_output_transform = cls.model_output_transform

        if not send_data_to_device:
            send_data_to_device = cls.send_data_to_device

        if not collate_fn:
            collate_fn = cls.collate_fn

        test_dataset = cls.dataset(
            root=data_root,
            split="val",
            transform=input_transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        test_loader.no_classes = 12  # Number of classes for CamVid
        test_results, speed_mem_metrics, run_hash = evaluate_segmentation(
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
            speed_mem_metrics=speed_mem_metrics,
            pytorch_hub_id=pytorch_hub_url,
            model=paper_model_name,
            model_description=model_description,
            arxiv_id=paper_arxiv_id,
            pwc_id=paper_pwc_id,
            paper_results=paper_results,
        )
