import os
import json
import zipfile
import contextlib

import numpy as np
from PIL import Image

from torchbench.tests.common_utils import get_tmp_dir


@contextlib.contextmanager
def ade20k_root():
    def _make_image(file):
        Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)).save(file)

    def _make_zip(archive, tmp_dir, content):
        zipf = zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED)
        for root, dirs, files in os.walk(content):
            for file in files:
                file_name = os.path.join(root, file)
                zipf.write(file_name, arcname=file_name.split(tmp_dir)[-1])
        zipf.close()

    def _make_data_archive(root):
        with get_tmp_dir() as tmp_dir:
            base_dir = "ADEChallengeData2016"

            for folder_name in ["images", "annotations"]:
                folder_dir = os.path.join(tmp_dir, base_dir, folder_name)
                os.makedirs(folder_dir)
                for split_name in ["training", "validation"]:
                    split_dir = os.path.join(folder_dir, split_name)
                    os.makedirs(os.path.join(split_dir))
                    _make_image(
                        os.path.join(split_dir, "ADE_train_00000000.png")
                    )

            archive = os.path.join(root, "ADEChallengeData2016.zip")

            _make_zip(
                archive=archive,
                tmp_dir=tmp_dir,
                content=os.path.join(tmp_dir, base_dir),
            )

    with get_tmp_dir() as root:
        _make_data_archive(root)

        yield root


@contextlib.contextmanager
def cityscapes_root():
    def _make_image(file):
        Image.fromarray(np.zeros((1024, 2048, 3), dtype=np.uint8)).save(file)

    def _make_regular_target(file):
        Image.fromarray(np.zeros((1024, 2048), dtype=np.uint8)).save(file)

    def _make_color_target(file):
        Image.fromarray(np.zeros((1024, 2048, 4), dtype=np.uint8)).save(file)

    def _make_polygon_target(file):
        polygon_example = {
            "imgHeight": 1024,
            "imgWidth": 2048,
            "objects": [
                {
                    "label": "sky",
                    "polygon": [
                        [1241, 0],
                        [1234, 156],
                        [1478, 197],
                        [1611, 172],
                        [1606, 0],
                    ],
                },
                {
                    "label": "road",
                    "polygon": [
                        [0, 448],
                        [1331, 274],
                        [1473, 265],
                        [2047, 605],
                        [2047, 1023],
                        [0, 1023],
                    ],
                },
            ],
        }
        with open(file, "w") as outfile:
            json.dump(polygon_example, outfile)

    with get_tmp_dir() as tmp_dir:

        for mode in ["Coarse", "Fine"]:
            gt_dir = os.path.join(tmp_dir, "gt%s" % mode)
            os.makedirs(gt_dir)

            if mode == "Coarse":
                splits = ["train", "train_extra", "val"]
            else:
                splits = ["train", "test", "val"]

            for split in splits:
                split_dir = os.path.join(gt_dir, split)
                os.makedirs(split_dir)
                for city in ["bochum", "bremen"]:
                    city_dir = os.path.join(split_dir, city)
                    os.makedirs(city_dir)
                    _make_color_target(
                        os.path.join(
                            city_dir,
                            "{city}_000000_000000_gt{mode}_color.png".format(
                                city=city, mode=mode
                            ),
                        )
                    )
                    _make_regular_target(
                        os.path.join(
                            city_dir,
                            "{city}_000000_000000_gt"
                            "{mode}_instanceIds.png".format(
                                city=city, mode=mode
                            ),
                        )
                    )
                    _make_regular_target(
                        os.path.join(
                            city_dir,
                            "{city}_000000_000000_gt"
                            "{mode}_labelIds.png".format(city=city, mode=mode),
                        )
                    )
                    _make_polygon_target(
                        os.path.join(
                            city_dir,
                            "{city}_000000_000000_gt"
                            "{mode}_polygons.json".format(
                                city=city, mode=mode
                            ),
                        )
                    )

        # leftImg8bit dataset
        leftimg_dir = os.path.join(tmp_dir, "leftImg8bit")
        os.makedirs(leftimg_dir)
        for split in ["test", "train_extra", "train", "val"]:
            split_dir = os.path.join(leftimg_dir, split)
            os.makedirs(split_dir)
            for city in ["bochum", "bremen"]:
                city_dir = os.path.join(split_dir, city)
                os.makedirs(city_dir)
                _make_image(
                    os.path.join(
                        city_dir,
                        "{city}_000000_000000_leftImg8bit.png".format(
                            city=city
                        ),
                    )
                )

        yield tmp_dir


@contextlib.contextmanager
def camvid_root():
    def _make_image(file):
        Image.fromarray(np.zeros((480, 360, 3), dtype=np.uint8)).save(file)

    with get_tmp_dir() as tmp_dir:
        for folder_name in [
            "val",
            "valannot",
            "train",
            "trainannot",
            "test",
            "testannot",
        ]:
            split_dir = os.path.join(tmp_dir, folder_name)
            os.makedirs(split_dir)
            _make_image(os.path.join(split_dir, "0016E5_00000.png"))

        yield tmp_dir
