from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from sotabenchapi.core import BenchmarkResult
from torchbench.utils import send_model_to_device, default_data_to_device

from .utils import evaluate_classification


class CIFAR100:
    """CIFAR 100 Dataset."""
    dataset = datasets.CIFAR100
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    input_transform = transforms.Compose([transforms.ToTensor(), normalize])
    send_data_to_device = default_data_to_device

    @classmethod
    def benchmark(cls, model, input_transform=None, target_transform=None, model_output_transform=None,
                  send_data_to_device=None, device: str = 'cuda', data_root: str = './.data/vision/cifar100',
                  num_workers: int = 4, batch_size: int = 128, num_gpu: int = 1, paper_model_name: str = None,
                  paper_arxiv_id: str = None, paper_pwc_id: str = None, pytorch_hub_url: str = None) -> BenchmarkResult:

        config = locals()
        model, device = send_model_to_device(model, device=device, num_gpu=num_gpu)
        model.eval()

        if not input_transform:
            input_transform = cls.input_transform

        if not send_data_to_device:
            send_data_to_device = cls.send_data_to_device

        test_dataset = cls.dataset(data_root, train=False, transform=input_transform, target_transform=target_transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_results = evaluate_classification(model=model, test_loader=test_loader,
                                               model_output_transform=model_output_transform,
                                               send_data_to_device=send_data_to_device, device=device)

        print(' * Acc@1 {top1:.3f} Acc@5 {top5:.3f}'.format(top1=test_results['Top 1 Accuracy'],
                                                            top5=test_results['Top 5 Accuracy']))

        return BenchmarkResult(task="Image Classification", benchmark=cls, config=config, dataset=test_dataset,
                               results=test_results, pytorch_hub_id=pytorch_hub_url,
                               model=paper_model_name, arxiv_id=paper_arxiv_id,
                               pwc_id=paper_pwc_id)
