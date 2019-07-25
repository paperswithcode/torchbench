import numpy as np
import unittest
import mock
import PIL
import torchvision
from torchbench.datasets import ADE20K, CamVid, Cityscapes, PASCALContext

from fakedata_generation import cityscapes_root, ade20k_root, camvid_root


class Tester(unittest.TestCase):
    def generic_segmentation_dataset_test(self, dataset, num_images=1):
        self.assertEqual(len(dataset), num_images)
        img, target = dataset[0]
        self.assertTrue(isinstance(img, PIL.Image.Image))
        self.assertTrue(isinstance(target, PIL.Image.Image))

    @mock.patch('torchvision.datasets.utils.download_url')
    def test_ade20k(self, mock_download):
        with ade20k_root() as root:
            dataset = ADE20K(root, split='train', download=True)
            self.generic_segmentation_dataset_test(dataset)
            dataset = ADE20K(root, split='val', download=True)
            self.generic_segmentation_dataset_test(dataset)

    def test_camvid(self):
        with camvid_root() as root:
            dataset = CamVid(root, split='train')
            self.generic_segmentation_dataset_test(dataset)
            dataset = CamVid(root, split='val')
            self.generic_segmentation_dataset_test(dataset)
            dataset = CamVid(root, split='test')
            self.generic_segmentation_dataset_test(dataset)

    def test_cityscapes(self):
        with cityscapes_root() as root:

            for mode in ['coarse', 'fine']:

                if mode == 'coarse':
                    splits = ['train', 'train_extra', 'val']
                else:
                    splits = ['train', 'val', 'test']

                for split in splits:
                    for target_type in ['semantic', 'instance']:
                        dataset = Cityscapes(root, split=split, target_type=target_type, mode=mode)
                        self.generic_segmentation_dataset_test(dataset, num_images=2)

                    color_dataset = Cityscapes(root, split=split, target_type='color', mode=mode)
                    color_img, color_target = color_dataset[0]
                    self.assertTrue(isinstance(color_img, PIL.Image.Image))
                    self.assertTrue(np.array(color_target).shape[2] == 4)

                    polygon_dataset = Cityscapes(root, split=split, target_type='polygon', mode=mode)
                    polygon_img, polygon_target = polygon_dataset[0]
                    self.assertTrue(isinstance(polygon_img, PIL.Image.Image))
                    self.assertTrue(isinstance(polygon_target, dict))
                    self.assertTrue(isinstance(polygon_target['imgHeight'], int))
                    self.assertTrue(isinstance(polygon_target['objects'], list))

if __name__ == '__main__':
    unittest.main()
