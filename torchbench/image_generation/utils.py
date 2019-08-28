import numpy as np
from scipy import linalg
from scipy.stats import entropy
import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.models.inception import inception_v3


class FIDInceptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception_model = inception_v3(pretrained=True)
        self.inception_model.Mixed_7c.register_forward_hook(self.output_hook)
        self.mixed_7c_output = None

    def output_hook(self, module, input, output):
        """Output will be of dimensions (batch_size, 2048, 8, 8)."""
        self.mixed_7c_output = output

    def forward(self, x):
        """x inputs should be (N, 3, 299, 299) in range -1 to 1.

        Returns activations in form of torch.tensor of shape (N, 2048, 1, 1)
        """

        self.inception_model(x)

        activations = self.mixed_7c_output
        activations = F.adaptive_avg_pool2d(activations, (1, 1))
        activations = activations.view(x.shape[0], 2048)
        return activations


class InceptionScore(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.inception_model = inception_v3(pretrained=True).to(device=device)
        self.up = nn.Upsample(size=(299, 299), mode="bilinear").to(
            device=device
        )

    def forward(self, x):
        """x inputs should be (N, 3, 299, 299) in range -1 to 1.

        Returns class probabilities in form of torch.tensor of shape
        (N, 1000, 1, 1).
        """
        x = self.up(x)
        x = self.inception_model(x)
        return F.softmax(x).data.cpu().numpy()


def calculate_inception_score(
    sample_dataloader,
    test_dataloader,
    device="cuda",
    num_images=50000,
    splits=10,
):
    """Calculate the inception score for a model's samples.

    Args:
        sample_dataloader: Dataloader for the generated image samples from the
            model.
        test_dataloader: Dataloader for the real images from the dataset to
            compare to.
        device: to perform the evaluation (e.g. 'cuda' for GPU).
        num_images: number of images to evaluate.
        splits: nu,ber of splits to perform for the evaluation.

    Returns:
        dict: Dictionary with key being the metric name, and values being the
            metric scores.
    """
    inception_model = InceptionScore(device=device)
    inception_model.eval()

    preds = np.zeros((num_images, 1000))

    for i, batch in enumerate(sample_dataloader, 0):
        batch = batch.to(device=device)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]
        predictions = inception_model(batchv)
        start = i * test_dataloader.batch_size
        n_predictions = len(preds[start : start + batch_size_i])
        preds[start : start + batch_size_i] = predictions[:n_predictions]

    split_scores = []

    for k in range(splits):
        part = preds[
            k * (num_images // splits) : (k + 1) * (num_images // splits), :
        ]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return {"Inception Score": np.mean(split_scores)}


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.

    This code is taken from:
    https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py

    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Args:
        mu1: Numpy array containing the activations of the pool_3 layer of the
            inception net ( like returned by the function 'get_predictions')
            for generated samples.
        mu2: The sample mean over activations of the pool_3 layer,
            precalcualted on an representive data set.
        sigma1: The covariance matrix over activations of the pool_3 layer for
            generated samples.
        sigma2: The covariance matrix over activations of the pool_3 layer,
            precalcualted on an representive data set.
        eps: Error margin.

    Returns:
        The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (
        diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    )


def get_activations(
    dataloader, device="cuda", dims=2048, num_images=500, real_dataset=False
):
    """Calculates the Inception activations for a dataset.

    Args:
        dataloader: Dataloader of data from which to obtain activations.
        device: to perform the evaluation (e.g. 'cuda' for GPU).
        dims: number of activations.
        num_images: number of images to evaluate.
        real_dataset: bool (whether the dataset is real or generated).
    """

    n_batches = len(dataloader) // dataloader.batch_size
    n_used_imgs = n_batches * dataloader.batch_size

    if real_dataset is True:
        pred_arr = np.empty((num_images, dims))
    else:
        pred_arr = np.empty((n_used_imgs, dims))

    partial_inception_model = FIDInceptionModel().to(device=device)
    partial_inception_model.to(device=device)

    for i, batch in enumerate(dataloader, 0):
        if real_dataset is True:
            batch_to_use = batch[0]
        else:
            batch_to_use = batch
        batch_to_use = batch_to_use.to(device=device)
        batch_size_i = batch_to_use.size()[0]
        start = i * dataloader.batch_size
        end = start + batch_size_i
        up = nn.Upsample(size=(299, 299), mode="bilinear")
        batch_up = up(batch_to_use)
        pred = partial_inception_model(batch_up)
        n_predictions = len(pred_arr[start:end])
        pred_arr[start:end] = (
            pred.cpu()
            .data.numpy()
            .reshape(dataloader.batch_size, -1)[:n_predictions]
        )
        if end > n_used_imgs:
            break

    return pred_arr


def calculate_activation_statistics(
    dataloader, device="cuda", num_images=500, real_dataset=False
):
    """Calculate the activation statistics for a dataset.

    Args:
        dataloader: Dataloader of data from which to obtain activations.
        device: to perform the evaluation (e.g. 'cuda' for GPU).
        num_images: number of images to evaluate.
        real_dataset: bool (whether the dataset is real or generated).

    Returns:
        Mean activations (np.array), std of activations (np.array).
    """
    act = get_activations(
        dataloader,
        device,
        dims=2048,
        num_images=num_images,
        real_dataset=real_dataset,
    )
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_fid(
    sample_dataloader, test_dataloader, device="cuda", num_images=500
):
    """Calculate the Frechet Inception Distance between two datasets.

    Args:
        sample_dataloader: Dataloader of generated image data.
        test_dataloader: Dataloader of real-life image data to test against.
        device: to perform the evaluation (e.g. 'cuda' for GPU).
        num_images: number of images to evaluate.

    Returns:
        dict: Dictionary with key as the metric name, value as the metric
            value.
    """
    m1, s1 = calculate_activation_statistics(
        dataloader=sample_dataloader, device=device, num_images=num_images
    )
    m2, s2 = calculate_activation_statistics(
        dataloader=test_dataloader,
        device=device,
        num_images=num_images,
        real_dataset=True,
    )
    return {"FID": calculate_frechet_distance(m1, s1, m2, s2)}


def evaluate_image_generation_gan(
    model, model_output_transform, test_loader, device="cuda"
):
    """Evaluate the image generation performance for a GAN.

    Evaluation will be against a base dataset (test_loader).

    Args:
        model: PyTorch model instance.
        model_output_transform: a function that transforms the model output
            (e.g. torch.Tensor output). This would be done, for instance, to
            normalize outputs to the right values (between -1 and 1 for
            inception).
        test_loader: The Dataloader for the test dataset (e.g. CIFAR-10).
        device: to perform the evaluation (e.g. 'cuda' for GPU).

    Returns:
        dict: Dictionary with keys as metric keys, and values as metric values.
    """

    num_images = 50000

    noise, _ = model.buildNoiseData(num_images)
    noise_dataloader = torch.utils.data.DataLoader(
        noise, batch_size=test_loader.batch_size
    )

    output = None
    with torch.no_grad():
        for i, noise_batch in enumerate(noise_dataloader):
            partial_output = model.test(noise_batch).to(device=device)
            if output is None:
                output = partial_output
            else:
                output = torch.cat((output, partial_output))

    if model_output_transform is not None:
        output = model_output_transform(
            output, target=None, device=device, model=model
        )

    # Set up dataloader
    sample_dataloader = torch.utils.data.DataLoader(
        output, batch_size=test_loader.batch_size
    )

    # Calculate Metrics
    inception_score = calculate_inception_score(
        test_dataloader=test_loader,
        sample_dataloader=sample_dataloader,
        device=device,
        num_images=num_images,
    )

    fid = calculate_fid(
        test_dataloader=test_loader,
        sample_dataloader=sample_dataloader,
        device=device,
        num_images=num_images,
    )

    return {**inception_score, **fid}
