import os
from torch.utils.data import DataLoader

from sotabenchapi.core import BenchmarkResult
from torchbench.datasets import CocoDetection
from torchbench.utils import send_model_to_device

from torchbench.object_detection.transforms import (
    Compose,
    ConvertCocoPolysToMask,
    ToTensor,
)
from torchbench.object_detection.utils import evaluate_detection_coco


def coco_data_to_device(
    input, target, device: str = "cuda", non_blocking: bool = True
):
    input = list(
        inp.to(device=device, non_blocking=non_blocking) for inp in input
    )
    target = [
        {
            k: v.to(device=device, non_blocking=non_blocking)
            for k, v in t.items()
        }
        for t in target
    ]
    return input, target


def coco_collate_fn(batch):
    return tuple(zip(*batch))


def coco_output_transform(output, target):
    output = [{k: v.to("cpu") for k, v in t.items()} for t in output]
    return output, target


class COCO:
    """`COCO <https://www.sotabench.com/benchmark/coco-minival>`_ benchmark.

    Note that COCO 2017 validation == 'minival' == 'val2017'

    Examples:
        Evaluate a Mask R-CNN model from the torchvision repository:

        .. code-block:: python

            from torchbench.object_detection import COCO
            from torchbench.utils import send_model_to_device
            from torchbench.object_detection.transforms import Compose, ConvertCocoPolysToMask, ToTensor
            import torchvision
            import PIL

            def coco_data_to_device(input, target, device: str = "cuda", non_blocking: bool = True):
                input = list(inp.to(device=device, non_blocking=non_blocking) for inp in input)
                target = [{k: v.to(device=device, non_blocking=non_blocking) for k, v in t.items()} for t in target]
                return input, target

            def coco_collate_fn(batch):
                return tuple(zip(*batch))

            def coco_output_transform(output, target):
                output = [{k: v.to("cpu") for k, v in t.items()} for t in output]
                return output, target

            transforms = Compose([ConvertCocoPolysToMask(), ToTensor()])

            model = torchvision.models.detection.__dict__['maskrcnn_resnet50_fpn'](num_classes=91, pretrained=True)

            # Run the benchmark
            COCO.benchmark(
                model=model,
                paper_model_name='Mask R-CNN (ResNet-50-FPN)',
                paper_arxiv_id='1703.06870',
                transforms=transforms,
                model_output_transform=coco_output_transform,
                send_data_to_device=coco_data_to_device,
                collate_fn=coco_collate_fn,
                batch_size=8,
                num_gpu=1
            )
    """

    dataset = CocoDetection
    transforms = Compose([ConvertCocoPolysToMask(), ToTensor()])
    send_data_to_device = coco_data_to_device
    collate_fn = coco_collate_fn
    model_output_transform = coco_output_transform
    task = "Object Detection"

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
        dataset_year="2017",
        device: str = "cuda",
        data_root: str = "./.data/vision/coco",
        num_workers: int = 4,
        batch_size: int = 1,
        pin_memory: bool = True,
        num_gpu: int = 1,
        paper_model_name: str = None,
        paper_arxiv_id: str = None,
        paper_pwc_id: str = None,
        paper_results: dict = None,
        pytorch_hub_url: str = None,
    ) -> BenchmarkResult:
        """Benchmarking function.

        Args:
            model: a PyTorch module, (e.g. a ``nn.Module`` object), that takes
                in COCO inputs and outputs COCO predictions.
            model_description (str, optional): Optional model description.
            input_transform (transforms.Compose, optional): Composing the
                transforms used to transform the dataset, e.g. applying
                resizing (e.g ``transforms.Resize``), center cropping, to
                tensor transformations and normalization.
            target_transform (torchvision.transforms.Compose, optional):
                Composing any transforms used to transform the target.
            transforms (torchbench.object_detection.transforms.Compose, optional):
                Does a joint transform on the input and the target - please see the
                torchbench.object_detection.transforms file for more information.
            model_output_transform (callable, optional): An optional function
                that takes in model output (after being passed through your
                ``model`` forward pass) and transforms it. Afterwards, the
                output will be passed into an evaluation function.
            collate_fn (callable, optional): How the dataset is collated - an
            optional callable passed into the DataLoader
            send_data_to_device (callable, optional): An optional function
                specifying how the model is sent to a device; see
                ``torchbench.utils.send_model_to_device`` for the default
                treatment.
            dataset_year (str, optional): the dataset year for COCO to use; the
            default (2017) creates the 'minival' validation set.
            device (str): Default is 'cuda' - this is the device that the model
                is sent to in the default treatment.
            data_root (str): The location of the COCO dataset - change this
                parameter when evaluating locally if your COCO data is
                located in a different folder (or alternatively if you want to
                download to an alternative location).
            num_workers (int): The number of workers to use for the DataLoader.
            batch_size (int) : The batch_size to use for evaluation; if you get
                memory errors, then reduce this (half each time) until your
                model fits onto the GPU.
            num_gpu (int): Number of GPUs - note that sotabench.com workers
                only support 1 GPU for now.
            paper_model_name (str, optional): The name of the model from the
                paper - if you want to link your build to a machine learning
                paper. See the COCO benchmark page for model names,
                https://www.sotabench.com/benchmark/coco-minival, e.g. on the paper
                leaderboard tab.
            paper_arxiv_id (str, optional): Optional linking to ArXiv if you
                want to link to papers on the leaderboard; put in the
                corresponding paper's ArXiv ID, e.g. '1611.05431'.
            paper_pwc_id (str, optional): Optional linking to Papers With Code;
                put in the corresponding papers with code URL slug, e.g.
                'u-gat-it-unsupervised-generative-attentional'
            paper_results (dict, optional) : If the paper you are reproducing
                does not have model results on sotabench.com, you can specify
                the paper results yourself through this argument, where keys
                are metric names, values are metric values. e.g::

                    {'box AP': 0.349, 'AP50': 0.592, ...}.

                Ensure that the metric names match those on the sotabench
                leaderboard - for COCO it should be 'box AP', 'AP50',
                'AP75', 'APS', 'APM', 'APL'
            pytorch_hub_url (str, optional): Optional linking to PyTorch Hub
                url if your model is linked there; e.g:
                'nvidia_deeplearningexamples_waveglow'.
        """

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
            root=os.path.join(data_root, "val%s" % dataset_year),
            annFile=os.path.join(
                data_root, "annotations/instances_val%s.json" % dataset_year
            ),
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
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
        test_loader.no_classes = 91  # Number of classes for COCO Detection
        test_results, speed_mem_metrics, run_hash = evaluate_detection_coco(
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
            dataset='COCO minival',
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
