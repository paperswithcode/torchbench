import numpy as np
import tqdm
import torch
from torch.nn import CrossEntropyLoss
from sotabenchapi.check import in_check_mode
from sotabenchapi.client import Client

from torchbench.utils import calculate_run_hash


def evaluate_language_model(
    model,
    test_loader,
    model_output_transform,
    send_data_to_device,
    device="cuda",
):
    n_steps, eval_loss = 0, 0

    iterator = tqdm.tqdm(test_loader, desc="Evaluation")

    with torch.no_grad():
        for i, labels in enumerate(iterator):

            labels, _ = send_data_to_device(labels, None, device=device)
            output = model(labels)

            if model_output_transform is not None:
                output = model_output_transform(output, None, model=model)

            shift_logits = output[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            objective = CrossEntropyLoss(ignore_index=-1)
            loss = objective(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            eval_loss += loss.item()
            n_steps += 1
            iterator.desc = (
                f"Eval loss: {eval_loss / n_steps} "
                f"ppl: {np.exp(eval_loss / n_steps)}"
            )

            if i == 0:  # for sotabench.com caching of evaluation
                run_hash = calculate_run_hash([eval_loss], output)
                # if we are in check model we don't need to go beyond the
                # first batch
                if in_check_mode():
                    iterator.close()
                    break

                # get the cached values from sotabench.com if available
                client = Client.public()
                cached_res = client.get_results_by_run_hash(run_hash)
                if cached_res:
                    iterator.close()
                    print(
                        "No model change detected (using the first batch "
                        "run_hash). Returning cached results."
                    )
                    return cached_res, run_hash

    return {"Perplexity": np.exp(eval_loss / n_steps)}, run_hash
