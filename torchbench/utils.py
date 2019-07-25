import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

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
    """Computes precision@k for the specified values of k"""
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


def default_data_to_device(input, target, device: str = 'cuda', non_blocking: bool = True):
    """
    Sends data output from a PyTorch Dataloader to the device
    """

    input = input.to(device=device, non_blocking=non_blocking)
    target = target.to(device=device, non_blocking=non_blocking)

    return input, target


def send_model_to_device(model, num_gpu: int = 1, device: str = 'cuda'):
    """Sends PyTorch model to a device and returns the model"""

    device = torch.device(device)

    if num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu)))
    else:
        model = model

    try:
        model = model.to(device=device)
    except AttributeError:
        print('Warning: to method not found, using default object')
        return model, device

    return model, device
