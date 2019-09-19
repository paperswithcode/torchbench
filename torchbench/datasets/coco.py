import os
from os.path import dirname as up

from PIL import Image
from torchvision.datasets.vision import VisionDataset

from torchbench.utils import extract_archive


class CocoDetection(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None, download=False, **kwargs):

        if download:
            self._download(root, annFile)

        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self._transforms = transforms
        self.annFile = annFile
        self.root = root

    def _download(self, root, annFile):
        if not os.path.isdir(root) or not os.path.isdir(annFile):
            if "2017" in root:
                image_dir_zip = os.path.join(up(root), "val2017.zip")
            elif "2014" in root:
                image_dir_zip = os.path.join(up(root), "val2014.zip")
            else:
                image_dir_zip = None

            if "2017" in annFile:
                annotations_dir_zip = os.path.join(
                    up(up(annFile)), "annotations_trainval2017.zip"
                )
            elif "2014" in annFile:
                annotations_dir_zip = os.path.join(
                    up(up(annFile)), "annotations_trainval2014.zip"
                )
            else:
                annotations_dir_zip = None

            if (
                image_dir_zip is not None
                and os.path.isfile(image_dir_zip)
                and annotations_dir_zip is not None
                and os.path.isfile(annotations_dir_zip)
            ):
                extract_archive(from_path=image_dir_zip, to_path=up(root))
                extract_archive(
                    from_path=annotations_dir_zip, to_path=up(root)
                )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        image_id = self.ids[index]
        target = dict(image_id=image_id, annotations=target, index=index)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.ids)
