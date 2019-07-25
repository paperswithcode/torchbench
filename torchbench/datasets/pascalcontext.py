import copy
import json
import os
import torch
from torchvision.datasets.vision import VisionDataset

from PIL import Image
from .utils import check_integrity, download_url, download_and_extract_archive

ARCHIVE_DICT = {
    'trainval_annot': {
        'url': 'https://codalabuser.blob.core.windows.net/public/trainval_merged.json',
        'md5': '3c2d0c0656b7be9eb61928ffe885d8ce',
    },
    'trainval': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
        'base_dir': 'VOCdevkit/VOC2010',
    },
    'mask_train': {
        'url': 'https://hangzh.s3.amazonaws.com/encoding/data/pcontext/train.pth',
        'md5': 'cf4344a5eef8139e057e3efefab18b37',
    },
    'mask_val': {
        'url': 'https://hangzh.s3.amazonaws.com/encoding/data/pcontext/val.pth',
        'md5': '92636c23dca1b3f0ef8b1fba3a9e6b9e',
    }
}


class PASCALContext(VisionDataset):
    """`Pascal Context <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self,
                 root,
                 split='train',
                 download=False,
                 transform=None,
                 target_transform=None,
                 transforms=None):
        super(PASCALContext, self).__init__(root, transforms, transform, target_transform)

        base_dir = ARCHIVE_DICT['trainval']['base_dir']
        self.voc_root = os.path.join(self.root, base_dir)
        self.image_dir = os.path.join(self.voc_root, 'JPEGImages')
        self.split = split

        if download:
            self.download()

        if not os.path.isdir(self.voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.annotations_dict = json.load(open(self.annotations_file, 'r'))

        old_imgs = self.annotations_dict['images']
        self.ids = copy.copy(old_imgs)
        for img in old_imgs:
            if img['phase'] != self.split:
                self.ids.remove(img)

        mask_file = os.path.join(self.voc_root, self.split+'.pth')
        self.masks = torch.load(mask_file)

    @property
    def annotations_file(self):
        return os.path.join(self.voc_root, 'trainval_merged.json')

    def download(self):

        if self.split == 'train':
            mask_dict = ARCHIVE_DICT['mask_train']
        elif self.split == 'val':
            mask_dict = ARCHIVE_DICT['mask_val']

        mask_file_loc = os.path.join(self.voc_root, os.path.basename(mask_dict['url']))

        if not os.path.isdir(self.voc_root):

            archive_dict = ARCHIVE_DICT['trainval']
            download_and_extract_archive(archive_dict['url'], self.root,
                                         extract_root=self.root,
                                         md5=archive_dict['md5'])

            archive_dict = ARCHIVE_DICT['trainval_annot']
            download_url(archive_dict['url'], self.voc_root,
                         filename=os.path.basename(archive_dict['url']),
                         md5=archive_dict['md5'])

        else:
            msg = ("You set download=True, but a folder VOCdevkit already exist in "
                   "the root directory. If you want to re-download or re-extract the "
                   "archive, delete the folder.")
            print(msg)

        if not os.path.isfile(mask_file_loc):

            download_url(mask_dict['url'], self.voc_root,
                         filename=os.path.basename(mask_dict['url']),
                         md5=mask_dict['md5'])

        else:
            msg = ("You set download=True, but a mask file already exists in "
                   "the root directory. If you want to re-download the "
                   "mask file, delete the file.")
            print(msg)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        img_id = self.ids[index]
        path = img_id['file_name']
        iid = img_id['image_id']
        img = Image.open(os.path.join(self.image_dir, path)).convert('RGB')
        target = self.masks[iid]

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)
