import os
import gzip
import hashlib
import tarfile
import zipfile
import numpy as np

import torch


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes precision@k for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def default_data_to_device(
    input, target=None, device: str = "cuda", non_blocking: bool = True
):
    """Sends data output from a PyTorch Dataloader to the device."""

    input = input.to(device=device, non_blocking=non_blocking)

    if target is not None:
        target = target.to(device=device, non_blocking=non_blocking)

    return input, target


def send_model_to_device(model, num_gpu: int = 1, device: str = "cuda"):
    """Sends PyTorch model to a device and returns the model."""

    device = torch.device(device)

    if num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu)))
    else:
        model = model

    try:
        model = model.to(device=device)
    except AttributeError:
        print("Warning: to method not found, using default object")
        return model, device

    return model, device


def _is_tar(filename):
    return filename.endswith(".tar")


def _is_targz(filename):
    return filename.endswith(".tar.gz")


def _is_gzip(filename):
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename):
    return filename.endswith(".zip")


def extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, "r") as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path):
        with tarfile.open(from_path, "r:gz") as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(
            to_path, os.path.splitext(os.path.basename(from_path))[0]
        )
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, "r") as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)


def calculate_run_hash(metrics: list, output: torch.Tensor):
    """Calculate the run hash for a given set of metrics and output.

    It calculates the hash using the metrics and the output tensor of the model
    (calculate after first batch).

    Args:
        metrics: list of metrics for the task and dataset.
        output: output from the model for a given batch of data.
    """

    if isinstance(output, dict):
        output = list(output)

    if not isinstance(output, list):
        output = np.round(output.cpu().numpy(), 3).tolist()

    hash_list = metrics + output
    m = hashlib.sha256()
    m.update(str(hash_list).encode("utf-8"))
    return m.hexdigest()
