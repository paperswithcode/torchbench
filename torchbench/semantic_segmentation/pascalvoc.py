from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from sotabenchapi.core import BenchmarkResult
from torchbench.utils import default_data_to_device, send_model_to_device

from torchbench.semantic_segmentation.transforms import (
    Normalize,
    Resize,
    ToTensor,
    Compose,
)
from torchbench.semantic_segmentation.utils import (
    default_seg_collate_fn,
    default_seg_output_transform,
    evaluate_segmentation,
)


class PASCALVOC:
    """`PASCALVOC <https://www.sotabench.com/benchmark/pascalvoc2012>`_ benchmark.

    Examples:
        Evaluate a ResNeXt model from the torchvision repository:

        .. code-block:: python

            from torchbench.semantic_segmentation import PASCALVOC
            from torchbench.semantic_segmentation.transforms import (
                Normalize,
                Resize,
                ToTensor,
                Compose,
            )
            from torchvision.models.segmentation import fcn_resnet101
            import torchvision.transforms as transforms
            import PIL

            def model_output_function(output, labels):
                return output['out'].argmax(1).flatten(), target.flatten()

            def seg_collate_fn(batch):
                images, targets = list(zip(*batch))
                batched_imgs = cat_list(images, fill_value=0)
                batched_targets = cat_list(targets, fill_value=255)
                return batched_imgs, batched_targets

            model = fcn_resnet101(num_classes=21, pretrained=True)

            normalize = Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            my_transforms = Compose([Resize((520, 480)), ToTensor(), normalize])

            PASCALVOC.benchmark(batch_size=32,
                model=model,
                transforms=my_transforms,
                model_output_transform=model_output_function,
                collate_fn=seg_collate_fn,
                paper_model_name='FCN ResNet-101',
                paper_arxiv_id='1605.06211')
    """
    dataset = datasets.VOCSegmentation
    normalize = Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transforms = Compose([Resize((520, 480)), ToTensor(), normalize])
    send_data_to_device = default_data_to_device
    collate_fn = default_seg_collate_fn
    model_output_transform = default_seg_output_transform
    task = "Semantic Segmentation"

    @classmethod
    def benchmark(
        cls,
        model,
        model_description=None,
        dataset_year="2012",
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
            transforms = cls.transforms

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
        test_loader.no_classes = 21  # Number of classes for PASCALVOC
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
            dataset='PASCAL VOC %s %s' % (dataset_year, "val"),
            results=test_results,
            speed_mem_metrics=speed_mem_metrics,
            pytorch_hub_id=pytorch_hub_url,
            model=paper_model_name,
            model_description=model_description,
            arxiv_id=paper_arxiv_id,
            pwc_id=paper_pwc_id,
            paper_results=paper_results,
            run_hash=run_hash,
        )
