# This code was based on the evaluation procedure at
# https://github.com/cybertronai/bflm/

import os
import numpy as np

from torch.utils.data import Dataset, Subset
from torchbench.utils import extract_archive
from torchbench.datasets.utils import download_and_extract_archive

ARCHIVE_DICT = {
    "full": {
        "url": (
            "https://s3.amazonaws.com/research.metamind.io/wikitext/"
            "wikitext-103-v1.zip"
        ),
        "md5": "9ddaacaf6af0710eda8c456decff7832",
        "base_dir": "wikitext-103",
    }
}


class WikiText103(Dataset):
    def __init__(
        self,
        root,
        encoder=None,
        split="test",
        context_length=1024,
        download=False,
    ):

        if not encoder:
            raise ValueError(
                "Encoder not specified : please specify an encoder"
            )

        if split not in ["train", "valid", "test"]:
            raise ValueError(
                'Invalid split, please use split="train", split="valid", '
                'split="test"'
            )

        self.root = root
        self.encoder = encoder
        self.split = split

        self.file = os.path.join(
            root, "wikitext-103", "wiki.{split}.tokens".format(split=split)
        )

        if download:
            self._download(root)

        self.data = self.load_dataset(combine=1e99)[0]
        self.subset = Subset(
            self.data,
            [
                slice(i, i + context_length)
                for i in range(
                    0,
                    len(self.data) - (len(self.data) % context_length),
                    context_length,
                )
            ],
        )

    def load_dataset(self, combine=50000):
        token_chunks = []
        raw_text = ""

        with open(self.file, "r") as fp:
            raw_text += fp.read()
        if len(raw_text) >= combine:
            tokens = np.stack(self.encoder.encode(raw_text))
            token_chunks.append(tokens)
            raw_text = ""
        else:
            raw_text += "<|endoftext|>"

        if raw_text:
            tokens = np.stack(self.encoder.encode(raw_text))
            token_chunks.append(tokens)
        return token_chunks

    def _download(self, root):

        if not os.path.isdir("{root}/wikitext-103".format(root=self.root)):
            archive_dict = ARCHIVE_DICT["full"]
            download_and_extract_archive(
                archive_dict["url"],
                self.root,
                extract_root=self.root,
                md5=archive_dict["md5"],
            )

        if not os.path.isfile(self.file):
            file_zip = os.path.join(root, "wikitext-103-v1.zip")

            if os.path.isfile(file_zip):
                extract_archive(from_path=file_zip, to_path=root)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        return self.subset[index]
