from torch.utils.data import DataLoader
import torchbench.datasets as datasets

from sotabenchapi.core import BenchmarkResult
from torchbench.utils import send_model_to_device, default_data_to_device

from torchbench.language_modelling.utils import evaluate_language_model


class WikiText103:

    dataset = datasets.WikiText103
    send_data_to_device = default_data_to_device
    task = "Language Modelling"

    @classmethod
    def benchmark(
        cls,
        model,
        model_description=None,
        encoder=None,
        context_length: int = 1024,
        model_output_transform=None,
        device: str = "cuda",
        data_root: str = "./.data/nlp/wikitext-103",
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
        model.eval()

        if not encoder:
            raise ValueError(
                "Please provide an encoder to evaluate on this benchmark!"
            )

        # Test Split

        test_dataset = cls.dataset(
            data_root,
            split="test",
            context_length=context_length,
            encoder=encoder,
            download=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_results, run_hash = evaluate_language_model(
            model=model,
            model_output_transform=model_output_transform,
            send_data_to_device=cls.send_data_to_device,
            test_loader=test_loader,
            device=device,
        )

        # Valid Split

        valid_dataset = cls.dataset(
            data_root,
            split="valid",
            context_length=context_length,
            encoder=encoder,
            download=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        valid_results, valid_run_hash = evaluate_language_model(
            model=model,
            model_output_transform=model_output_transform,
            send_data_to_device=cls.send_data_to_device,
            test_loader=valid_loader,
            device=device,
        )

        # Get final results
        if "Test perplexity" in test_results:
            final_results = valid_results  # hashed
        else:
            final_results = {
                "Test perplexity": test_results["Perplexity"],
                "Validation perplexity": valid_results["Perplexity"],
            }

        print(final_results)

        return BenchmarkResult(
            task=cls.task,
            config=config,
            dataset=cls.dataset.__name__,
            results=final_results,
            pytorch_hub_id=pytorch_hub_url,
            model=paper_model_name,
            model_description=model_description,
            arxiv_id=paper_arxiv_id,
            pwc_id=paper_pwc_id,
            paper_results=paper_results,
            run_hash=run_hash,
        )
