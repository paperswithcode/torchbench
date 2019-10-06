from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sotabenchapi.core import BenchmarkResult, check_inputs

from torchbench.utils import send_model_to_device, default_data_to_device
from torchbench.image_classification.utils import evaluate_classification


class CIFAR10:
    """CIFAR 10 Dataset."""

    dataset = datasets.CIFAR10
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    input_transform = transforms.Compose([transforms.ToTensor(), normalize])
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
        data_root: str = "./.data/vision/cifar10",
        num_workers: int = 4,
        batch_size: int = 128,
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

        if not input_transform:
            input_transform = cls.input_transform

        if not send_data_to_device:
            send_data_to_device = cls.send_data_to_device

        test_dataset = cls.dataset(
            data_root,
            train=False,
            transform=input_transform,
            target_transform=target_transform,
            download=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
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
