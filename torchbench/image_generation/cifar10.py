from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sotabenchapi.core import BenchmarkResult

from torchbench.utils import send_model_to_device
from torchbench.image_generation.utils import evaluate_image_generation_gan


class CIFAR10:

    dataset = datasets.CIFAR10
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    input_transform = transforms.Compose([transforms.ToTensor(), normalize])
    task = "Image Generation"

    @classmethod
    def benchmark(
        cls,
        model,
        model_description=None,
        input_transform=None,
        target_transform=None,
        model_output_transform=None,
        device: str = "cuda",
        data_root: str = "./.data/vision/cifar10",
        num_workers: int = 4,
        batch_size: int = 8,
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

        if hasattr(model, "eval"):
            model.eval()

        if not input_transform:
            input_transform = cls.input_transform

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
        test_results = evaluate_image_generation_gan(
            model=model,
            model_output_transform=model_output_transform,
            test_loader=test_loader,
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
