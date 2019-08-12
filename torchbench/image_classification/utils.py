import time
import tqdm
import torch
import torchvision

from torchbench.utils import AverageMeter, accuracy


def evaluate_classification(model, test_loader, model_output_transform, send_data_to_device, device='cuda'):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    with torch.no_grad():
        for i, (input, target) in enumerate(tqdm.tqdm(test_loader)):

            input, target = send_data_to_device(input, target, device=device)
            output = model(input)

            if model_output_transform is not None:
                output = model_output_transform(output, target)

            check_metric_inputs(output, target, test_loader.dataset, i)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

    return {
        'Top 1 Accuracy': top1.avg / 100,
        'Top 5 Accuracy': top5.avg / 100
    }


def check_metric_inputs(output, target, dataset, iteration_no):

    # Only perform checks for the first batch
    if iteration_no > 0:
        return

    if not isinstance(output, torch.Tensor):
        raise ValueError(
            'The output of the model was not of type torch.Tensor. We expect the output to'
            ' be of type torch.Tensor with dimensions (batch_size, number of classes). You'
            ' gave an output of type {output_type}.'
            '\n\nTo fix this you can either change the output of your model call - i.e. '
            'output = model(input) - so it returns an output of type torch.Tensor. Alternatively you can keep'
            ' your model output the same and write or alter a model_output_transform function'
            ' that post-processes the output to return a torch.Tensor. For example:\n\n'
            'def my_model_output_transform(output, target):\n'
            '   return torch.from_numpy(output), target\n\n'
            'You can then pass this into your evaluation function, i.e.:\n\n'
            'ImageNet.benchmark(model=model, model_output_transform=my_model_output_transform, ...)'.format(
                output_type=str(type(output))))

    if not isinstance(target, torch.Tensor):
        raise ValueError(
            'The target of the data was not of type torch.Tensor. We expect the target to'
            ' be of type torch.Tensor with a single dimension representing the batch size. You'
            ' gave an output of type {target_type}.'
            '\n\nThere are two places where the target might have been changed: if you defined a custom'
            ' send_data_to_device function, or if you changed the target in a model_output_transform'
            ' function.'.format(target_type=str(type(target))))

    if len(output.shape) != 2:
        raise ValueError(
            'The output of the model was not a 2 dimensional torch.Tensor. We expect the output to'
            ' have 2 dimensions (batch_size, number of classes). The torch.Tensor you provided had'
            ' the following shape which did not fit requirements: {output_shape}.'             
            '\n\nTo fix this you can either change the output of your model call - i.e. '
            'output = model(input) - so it returns a two dimensional torch.Tensor. Alternatively you can keep'
            ' your model output the same and write or alter a model_output_transform function'
            ' that post-processes the output to return a 2 dimensional torch.Tensor. For example:\n\n'
            'def my_model_output_transform(output, target):\n'
            '   return torch.from_numpy(output), target\n\n'
            'You can then pass this into your evaluation function, i.e.:\n\n'
            'ImageNet.benchmark(model=model, model_output_transform=my_model_output_transform, ...)'.format(
                output_shape=str(output.shape)))

    if len(target.shape) != 1:
        raise ValueError(
            'The target of the model was not a 1 dimensional torch.Tensor. We expect the target to'
            ' have a single dimension representing the batch size. The torch.Tensor you provided had'
            ' the following shape which did not fit requirements: {target_shape}.'            
            '\n\nThere are two places where the target might have been changed: if you defined a custom'
            ' send_data_to_device function, or if you changed the target in a model_output_transform'
            ' function.'.format(target_shape=str(target.shape)))

    if target.shape[0] != output.shape[0]:
        raise ValueError(
            'The target first dimension was of length {target_length}, but the output first dimension was of length {output_length} - '
            ' we expect these to be the same because the first dimension is the batch size.'.format(
                target_length=str(target.shape[0]), output_length=str(output.shape[0])))

    if isinstance(dataset, torchvision.datasets.ImageNet):
        if output.shape[1] != 1000:
            raise ValueError(
                'The output second dimension was of length {output_classes}, but ImageNet has 1000 classes.'
                ' So we expect a second dimension of length 1000.'.format(output_classes=str(output.shape[1])))

