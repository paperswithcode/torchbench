from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sotabenchapi.core import BenchmarkResult, check_inputs

from torchbench.utils import send_model_to_device, default_data_to_device
from torchbench.image_classification.utils import evaluate_classification


class ImageNet:
    """`ImageNet <https://www.sotabench.com/benchmark/imagenet>`_ benchmark.

    Examples:
        Evaluate a ResNeXt model from the torchvision repository:

        .. code-block:: python

            from torchbench.image_classification import ImageNet
            from torchvision.models.resnet import resnext101_32x8d
            import torchvision.transforms as transforms
            import PIL

            # Define the transforms need to convert ImageNet data to expected
            # model input
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            input_transform = transforms.Compose([
                transforms.Resize(256, PIL.Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

            # Run the benchmark
            ImageNet.benchmark(
                model=resnext101_32x8d(pretrained=True),
                paper_model_name='ResNeXt-101-32x8d',
                paper_arxiv_id='1611.05431',
                input_transform=input_transform,
                batch_size=256,
                num_gpu=1
            )

        If the model you are implementing does not have *paper* results on
        sotabench, you can add them:

        .. code-block:: python

            ...

            mynet_paper_results = {
                'Top 1 Accuracy': 0.754,
                'Top 5 Accuracy': 0.8565
            }

            # Run the benchmark
            ImageNet.benchmark(
                model=mynet101(pretrained=True),
                paper_model_name='MyNet',
                paper_arxiv_id='2099.05431',
                paper_results=mynet_paper_results,
                input_transform=input_transform,
                batch_size=256,
                num_gpu=1
            )
    """

    dataset = datasets.ImageNet
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    input_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    send_data_to_device = default_data_to_device
    task = "Image Classification"

    @classmethod
    @check_inputs
    def benchmark(
        cls,
        model,
        model_description=None,
        input_transform=None,
        target_transform=None,
        model_output_transform=None,
        send_data_to_device=None,
        device: str = "cuda",
        data_root: str = "./.data/vision/imagenet",
        num_workers: int = 4,
        batch_size: int = 128,
        pin_memory: bool = False,
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
                in ImageNet inputs and outputs ImageNet predictions.
            model_description (str, optional): Optional model description.
            input_transform (transforms.Compose, optional): Composing the
                transforms used to transform the dataset, e.g. applying
                resizing (e.g ``transforms.Resize``), center cropping, to
                tensor transformations and normalization.
            target_transform (torchvision.transforms.Compose, optional):
                Composing any transforms used to transform the target. This is
                usually not used for ImageNet.
            model_output_transform (callable, optional): An optional function
                that takes in model output (after being passed through your
                ``model`` forward pass) and transforms it. Afterwards, the
                output will be passed into an evaluation function.
            send_data_to_device (callable, optional): An optional function
                specifying how the model is sent to a device; see
                ``torchbench.utils.send_model_to_device`` for the default
                treatment.
            device (str): Default is 'cuda' - this is the device that the model
                is sent to in the default treatment.
            data_root (str): The location of the ImageNet dataset - change this
                parameter when evaluating locally if your ImageNet data is
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
                paper. See the ImageNet benchmark page for model names,
                https://www.sotabench.com/benchmark/imagenet, e.g. on the paper
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

                    {'Top 1 Accuracy': 0.543, 'Top 5 Accuracy': 0.654}.

                Ensure that the metric names match those on the sotabench
                leaderboard - for ImageNet it should be 'Top 1 Accuracy' and
                'Top 5 Accuracy'.
            pytorch_hub_url (str, optional): Optional linking to PyTorch Hub
                url if your model is linked there; e.g:
                'nvidia_deeplearningexamples_waveglow'.
        """

        print("Benchmarking on ImageNet...")

        config = locals()
        model, device = send_model_to_device(
            model, device=device, num_gpu=num_gpu
        )
        model.eval()

        if not input_transform:
            input_transform = cls.input_transform

        if not send_data_to_device:
            send_data_to_device = cls.send_data_to_device

        try:
            test_dataset = cls.dataset(
                data_root,
                split="val",
                transform=input_transform,
                target_transform=target_transform,
                download=True,
            )
        except Exception:
            test_dataset = cls.dataset(
                data_root,
                split="val",
                transform=input_transform,
                target_transform=target_transform,
                download=False,
            )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_results, speed_mem_metrics, run_hash = evaluate_classification(
            model=model,
            test_loader=test_loader,
            model_output_transform=model_output_transform,
            send_data_to_device=send_data_to_device,
            device=device,
        )

        print(
            " * Acc@1 {top1:.3f} Acc@5 {top5:.3f}".format(
                top1=test_results["Top 1 Accuracy"],
                top5=test_results["Top 5 Accuracy"],
            )
        )

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
            run_hash=run_hash,
        )
