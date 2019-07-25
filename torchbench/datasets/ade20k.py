import os
from collections import namedtuple

from torchvision.datasets.vision import VisionDataset
from PIL import Image

from .utils import download_and_extract_archive

ARCHIVE_DICT = {
    'trainval': {
        'url': 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip',
        'md5': '7328b3957e407ddae1d3cbf487f149ef',
        'base_dir': 'ADEChallengeData2016',
    }
}


class ADE20K(VisionDataset):
    """`ADE20K <https://groups.csail.mit.edu/vision/datasets/ADE20K/>`_ Dataset

    Args:
        root (string): Root directory of the ADE20K dataset
        split (string, optional): The image split to use, ``train`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            PIL image target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Examples:

        Get dataset for training and download from internet

        .. code-block:: python

            dataset = ADE20K('./data/ade20k', split='train', download=True)

            img, target = dataset[0]

        Get dataset for validation and download from internet

        .. code-block:: python

            dataset = ADE20K('./data/ade20k', split='val', download=True)

            img, target = dataset[0]
    """

    ADE20KClass = namedtuple('ADE20KClass', ['name', 'id', 'color'])

    classes = [
        ADE20KClass('wall', 1, (120, 120, 120)),
        ADE20KClass('building;edifice', 2, (180, 120, 120)),
        ADE20KClass('sky', 3, (6, 230, 230)),
        ADE20KClass('floor;flooring', 4, (80, 50, 50)),
        ADE20KClass('tree', 5, (4, 200, 3)),
        ADE20KClass('ceiling', 6, (120, 120, 80)),
        ADE20KClass('road;route', 7, (140, 140, 140)),
        ADE20KClass('bed', 8, (204, 5, 255)),
        ADE20KClass('windowpane;window', 9, (230, 230, 230)),
        ADE20KClass('grass', 10, (4, 250, 7)),
        ADE20KClass('cabinet', 11, (224, 5, 255)),
        ADE20KClass('sidewalk;pavement', 12, (235, 255, 7)),
        ADE20KClass('person', 13, (150, 5, 61)),
        ADE20KClass('earth;ground', 14, (120, 120, 70)),
        ADE20KClass('door;double;door', 15, (8, 255, 51)),
        ADE20KClass('table', 16, (255, 6, 82)),
        ADE20KClass('mountain;mount', 17, (143, 255, 140)),
        ADE20KClass('plant;flora;plant;life', 18, (204, 255, 4)),
        ADE20KClass('curtain;drape;drapery;mantle;pall', 19, (255, 51, 7)),
        ADE20KClass('chair', 20, (204, 70, 3)),
        ADE20KClass('car;auto;automobile;machine;motorcar', 21, (0, 102, 200)),
        ADE20KClass('water', 22, (61, 230, 250)),
        ADE20KClass('painting;picture', 23, (255, 6, 51)),
        ADE20KClass('sofa;couch;lounge', 24, (11, 102, 255)),
        ADE20KClass('shelf', 25, (255, 7, 71)),
        ADE20KClass('house', 26, (255, 9, 224)),
        ADE20KClass('sea', 27, (9, 7, 230)),
        ADE20KClass('mirror', 28, (220, 220, 220)),
        ADE20KClass('rug;carpet;carpeting', 29, (255, 9, 92)),
        ADE20KClass('field', 30, (112, 9, 255)),
        ADE20KClass('armchair', 31, (8, 255, 214)),
        ADE20KClass('seat', 32, (7, 255, 224)),
        ADE20KClass('fence;fencing', 33, (255, 184, 6)),
        ADE20KClass('desk', 34, (10, 255, 71)),
        ADE20KClass('rock;stone', 35, (255, 41, 10)),
        ADE20KClass('wardrobe;closet;press', 36, (7, 255, 255)),
        ADE20KClass('lamp', 37, (224, 255, 8)),
        ADE20KClass('bathtub;bathing;tub;bath;tub', 38, (102, 8, 255)),
        ADE20KClass('railing;rail', 39, (255, 61, 6)),
        ADE20KClass('cushion', 40, (255, 194, 7)),
        ADE20KClass('base;pedestal;stand', 41, (255, 122, 8)),
        ADE20KClass('box', 42, (0, 255, 20)),
        ADE20KClass('column;pillar', 43, (255, 8, 41)),
        ADE20KClass('signboard;sign', 44, (255, 5, 153)),
        ADE20KClass('chest;of;drawers;chest;bureau;dresser', 45, (6, 51, 255)),
        ADE20KClass('counter', 46, (235, 12, 255)),
        ADE20KClass('sand', 47, (160, 150, 20)),
        ADE20KClass('sink', 48, (0, 163, 255)),
        ADE20KClass('skyscraper', 49, (140, 140, 140)),
        ADE20KClass('fireplace;hearth;open;fireplace', 50, (250, 10, 15)),
        ADE20KClass('refrigerator;icebox', 51, (20, 255, 0)),
        ADE20KClass('grandstand;covered;stand', 52, (31, 255, 0)),
        ADE20KClass('path', 53, (255, 31, 0)),
        ADE20KClass('stairs;steps', 54, (255, 224, 0)),
        ADE20KClass('runway', 55, (153, 255, 0)),
        ADE20KClass('case;display;case;showcase;vitrine', 56, (0, 0, 255)),
        ADE20KClass('pool;table;billiard;table;snooker;table', 57, (255, 71, 0)),
        ADE20KClass('pillow', 58, (0, 235, 255)),
        ADE20KClass('screen;door;screen', 59, (0, 173, 255)),
        ADE20KClass('stairway;staircase', 60, (31, 0, 255)),
        ADE20KClass('river', 61, (11, 200, 200)),
        ADE20KClass('bridge;span', 62, (255, 82, 0)),
        ADE20KClass('bookcase', 63, (0, 255, 245)),
        ADE20KClass('blind;screen', 64, (0, 61, 255)),
        ADE20KClass('coffee;table;cocktail;table', 65, (0, 255, 112)),
        ADE20KClass('toilet;can;commode;crapper;pot;potty;stool', 66, (0, 255, 133)),
        ADE20KClass('flower', 67, (255, 0, 0)),
        ADE20KClass('book', 68, (255, 163, 0)),
        ADE20KClass('hill', 69, (255, 102, 0)),
        ADE20KClass('bench', 70, (194, 255, 0)),
        ADE20KClass('countertop', 71, (0, 143, 255)),
        ADE20KClass('stove;kitchen;stove;range;kitchen;cooking;stove', 72, (51, 255, 0)),
        ADE20KClass('palm;palm;tree', 73, (0, 82, 255)),
        ADE20KClass('kitchen;island', 74, (0, 255, 41)),
        ADE20KClass('computer', 75, (0, 255, 173)),
        ADE20KClass('swivel;chair', 76, (10, 0, 255)),
        ADE20KClass('boat', 77, (173, 255, 0)),
        ADE20KClass('bar', 78, (0, 255, 153)),
        ADE20KClass('arcade;machine', 79, (255, 92, 0)),
        ADE20KClass('hovel;hut;hutch;shack;shanty', 80, (255, 0, 255)),
        ADE20KClass('bus;coach;double-decker;passenger;vehicle', 81, (255, 0, 245)),
        ADE20KClass('towel', 82, (255, 0, 102)),
        ADE20KClass('light;light;source', 83, (255, 173, 0)),
        ADE20KClass('truck;motortruck', 84, (255, 0, 20)),
        ADE20KClass('tower', 85, (255, 184, 184)),
        ADE20KClass('chandelier;pendant;pendent', 86, (0, 31, 255)),
        ADE20KClass('awning;sunshade;sunblind', 87, (0, 255, 61)),
        ADE20KClass('streetlight;street;lamp', 88, (0, 71, 255)),
        ADE20KClass('booth;cubicle;stall;kiosk', 89, (255, 0, 204)),
        ADE20KClass('television', 90, (0, 255, 194)),
        ADE20KClass('airplane;aeroplane;plane', 91, (0, 255, 82)),
        ADE20KClass('dirt;track', 92, (0, 10, 255)),
        ADE20KClass('apparel;wearing;apparel;dress;clothes', 93, (0, 112, 255)),
        ADE20KClass('pole', 94, (51, 0, 255)),
        ADE20KClass('land;ground;soil', 95, (0, 194, 255)),
        ADE20KClass('bannister;banister;balustrade;balusters;handrail', 96, (0, 122, 255)),
        ADE20KClass('escalator;moving;staircase;moving;stairway', 97, (0, 255, 163)),
        ADE20KClass('ottoman;pouf;pouffe;puff;hassock', 98, (255, 153, 0)),
        ADE20KClass('bottle', 99, (0, 255, 10)),
        ADE20KClass('buffet;counter;sideboard', 100, (255, 112, 0)),
        ADE20KClass('poster;posting;placard;notice;bill;card', 101, (143, 255, 0)),
        ADE20KClass('stage', 102, (82, 0, 255)),
        ADE20KClass('van', 103, (163, 255, 0)),
        ADE20KClass('ship', 104, (255, 235, 0)),
        ADE20KClass('fountain', 105, (8, 184, 170)),
        ADE20KClass('conveyer;belt;conveyor;belt;conveyor;transporter', 106, (133, 0, 255)),
        ADE20KClass('canopy', 107, (0, 255, 92)),
        ADE20KClass('washer;automatic;washer;washing;machine', 108, (184, 0, 255)),
        ADE20KClass('plaything;toy', 109, (255, 0, 31)),
        ADE20KClass('swimming;pool;swimming;bath;natatorium', 110, (0, 184, 255)),
        ADE20KClass('stool', 111, (0, 214, 255)),
        ADE20KClass('barrel;cask', 112, (255, 0, 112)),
        ADE20KClass('basket;handbasket', 113, (92, 255, 0)),
        ADE20KClass('waterfall;falls', 114, (0, 224, 255)),
        ADE20KClass('tent;collapsible;shelter', 115, (112, 224, 255)),
        ADE20KClass('bag', 116, (70, 184, 160)),
        ADE20KClass('minibike;motorbike', 117, (163, 0, 255)),
        ADE20KClass('cradle', 118, (153, 0, 255)),
        ADE20KClass('oven', 119, (71, 255, 0)),
        ADE20KClass('ball', 120, (255, 0, 163)),
        ADE20KClass('food;solid;food', 121, (255, 204, 0)),
        ADE20KClass('step;stair', 122, (255, 0, 143)),
        ADE20KClass('tank;storage;tank', 123, (0, 255, 235)),
        ADE20KClass('trade;name;brand;name;brand;marque', 124, (133, 255, 0)),
        ADE20KClass('microwave;microwave;oven', 125, (255, 0, 235)),
        ADE20KClass('pot;flowerpot', 126, (245, 0, 255)),
        ADE20KClass('animal;animate;being;beast;brute;creature;fauna', 127, (255, 0, 122)),
        ADE20KClass('bicycle;bike;wheel;cycle', 128, (255, 245, 0)),
        ADE20KClass('lake', 129, (10, 190, 212)),
        ADE20KClass('dishwasher;dish;washer;dishwashing;machine', 130, (214, 255, 0)),
        ADE20KClass('screen;silver;screen;projection;screen', 131, (0, 204, 255)),
        ADE20KClass('blanket;cover', 132, (20, 0, 255)),
        ADE20KClass('sculpture', 133, (255, 255, 0)),
        ADE20KClass('hood;exhaust;hood', 134, (0, 153, 255)),
        ADE20KClass('sconce', 135, (0, 41, 255)),
        ADE20KClass('vase', 136, (0, 255, 204)),
        ADE20KClass('traffic;light;traffic;signal;stoplight', 137, (41, 0, 255)),
        ADE20KClass('tray', 138, (41, 255, 0)),
        ADE20KClass('trash;can;garbage;wastebin;bin;ashbin;dustbin;barrel;bin', 139, (173, 0, 255)),
        ADE20KClass('fan', 140, (0, 245, 255)),
        ADE20KClass('pier;wharf;wharfage;dock', 141, (71, 0, 255)),
        ADE20KClass('crt;screen', 142, (122, 0, 255)),
        ADE20KClass('plate', 143, (0, 255, 184)),
        ADE20KClass('monitor;monitoring;device', 144, (0, 92, 255)),
        ADE20KClass('bulletin;board;notice;board', 145, (184, 255, 0)),
        ADE20KClass('shower', 146, (0, 133, 255)),
        ADE20KClass('radiator', 147, (255, 214, 0)),
        ADE20KClass('glass;drinking;glass', 148, (25, 194, 194)),
        ADE20KClass('clock', 149, (102, 255, 0)),
        ADE20KClass('flag', 150, (92, 0, 255)),
    ]

    def __init__(self, root, split='train', download=False, transform=None, target_transform=None, transforms=None):
        super(ADE20K, self).__init__(root, transforms, transform, target_transform)

        base_dir = ARCHIVE_DICT['trainval']['base_dir']

        if split not in ['train', 'val']:
            raise ValueError('Invalid split! Please use split="train" or split="val"')

        if split == 'train':
            self.images_dir = os.path.join(self.root, base_dir, 'images', 'training')
            self.targets_dir = os.path.join(self.root, base_dir, 'annotations', 'training')
        elif split == 'val':
            self.images_dir = os.path.join(self.root, base_dir, 'images', 'validation')
            self.targets_dir = os.path.join(self.root, base_dir, 'annotations', 'validation')

        self.split = split

        if download:
            self.download()

        self.images = []
        self.targets = []

        for file_name in os.listdir(self.images_dir):
            self.images.append(os.path.join(self.images_dir, file_name))
            self.targets.append(os.path.join(self.targets_dir, file_name.replace('jpg', 'png')))

    def download(self):
        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):

            archive_dict = ARCHIVE_DICT['trainval']
            download_and_extract_archive(archive_dict['url'], self.root,
                                        extract_root=self.root,
                                        md5=archive_dict['md5'])

        else:
            msg = ("You set download=True, but a folder VOCdevkit already exist in "
                    "the root directory. If you want to re-download or re-extract the "
                    "archive, delete the folder.")
            print(msg)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target)
        """

        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.images)

    def extra_repr(self):
        lines = ["Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)