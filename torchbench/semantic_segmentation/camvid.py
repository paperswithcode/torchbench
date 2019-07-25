from torch.utils.data import DataLoader

from sotabenchapi.core import BenchmarkResult
from torchbench.datasets import CamVid
from torchbench.utils import default_data_to_device, send_model_to_device

from .transforms import Normalize, ToTensor, Compose
from .utils import default_seg_collate_fn, default_seg_output_transform, \
    evaluate_segmentation


class CamVid:

    dataset = CamVid
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transforms = Compose([ToTensor(), normalize])
    send_data_to_device = default_data_to_device
    collate_fn = default_seg_collate_fn
    model_output_transform = default_seg_output_transform

    @classmethod
    def benchmark(cls, model, input_transform=None, target_transform=None, transforms=None,
                  model_output_transform=None, collate_fn=None, send_data_to_device=None,
                  device: str = 'cuda', data_root: str = './.data/vision/camvid', num_workers: int = 4,
                  batch_size: int = 32, num_gpu: int = 1, paper_model_name: str = None,
                  paper_arxiv_id: str = None, paper_pwc_id: str = None, pytorch_hub_url: str = None) -> BenchmarkResult:

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

        test_dataset = cls.dataset(root=data_root, split='val', transform=input_transform,
                                   target_transform=target_transform, transforms=transforms)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
        test_loader.no_classes = 12  # Number of classes for CamVid
        test_results = evaluate_segmentation(model=model, test_loader=test_loader,
                                             model_output_transform=model_output_transform,
                                             send_data_to_device=send_data_to_device, device=device)

        print(test_results)

        return BenchmarkResult(task="Semantic Segmentation", benchmark=cls, config=config, dataset=test_dataset,
                               results=test_results, pytorch_hub_id=pytorch_hub_url,
                               model=paper_model_name, arxiv_id=paper_arxiv_id,
                               pwc_id=paper_pwc_id)
