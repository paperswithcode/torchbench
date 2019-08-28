# This code is based on and heavily modified from
# https://github.com/zalandoresearch/flair/
# We have removed some of the abstractions used in the library, but much of the
# logic is directly taken from the flair implementations.

import os
import re

from torch.utils.data import Dataset

from torchbench.utils import extract_archive


class CoNLL2003(Dataset):
    def __init__(
        self,
        root,
        language="eng",
        tagging_scheme="ner",
        split="train",
        download=False,
    ):

        if language not in ["eng"]:
            raise ValueError('Invalid language, please use "eng"')

        if split not in ["train", "dev", "test"]:
            raise ValueError(
                'Invalid split, please use split="train", split="dev", '
                'split="test"'
            )

        self.columns = {0: "text", 1: "pos", 2: "np", 3: "ner"}
        self.language = language
        self.tagging_scheme = tagging_scheme
        self.split = split

        if split == "train":
            self.file = os.path.join(root, "{lan}.train".format(lan=language))
        elif split == "dev":
            self.file = os.path.join(root, "{lan}.testa".format(lan=language))
        elif split == "test":
            self.file = os.path.join(root, "{lan}.testb".format(lan=language))

        if download:
            self._download(root)

        self.encoding = "utf-8"
        self.sentences = self.process_corpus()

    def _download(self, root):
        if not os.path.isfile(self.file):
            file_zip = os.path.join(root, "CoNLL2003.zip")

            if os.path.isfile(file_zip):
                extract_archive(from_path=file_zip, to_path=root)

    def convert_tag_scheme(self, sentence, target_scheme="iobes"):
        tags = []

        for token_dict in sentence:
            tags.append(token_dict["tags"][self.tagging_scheme])

        if target_scheme == "iob":
            iob2(tags)

        if target_scheme == "iobes":
            iob2(tags)
            tags = iob_iobes(tags)

        for index, tag in enumerate(tags):
            sentence[index]["tags"][self.tagging_scheme] = tag

    @staticmethod
    def infer_space_after(sentence):
        """From https://github.com/zalandoresearch/flair/data.py.

        Heuristics in case you wish to infer whitespace_after values for
        tokenized text. This is useful for some old NLP tasks (such as CoNLL-03
        and CoNLL-2000) that provide only tokenized data with no info of
        original whitespacing.
        """
        last_token_dict = None
        quote_count = 0

        for token_dict in sentence:
            token_dict["whitespace_after"] = None
            if token_dict["name"] == '"':
                quote_count += 1
                if quote_count % 2 != 0:
                    token_dict["whitespace_after"] = False
                elif last_token_dict is not None:
                    last_token_dict["whitespace_after"] = False

            if last_token_dict is not None:

                if token_dict["name"] in [
                    ".",
                    ":",
                    ",",
                    ";",
                    ")",
                    "n't",
                    "!",
                    "?",
                ]:
                    last_token_dict["whitespace_after"] = False

                if token_dict["name"].startswith("'"):
                    last_token_dict["whitespace_after"] = False

            if token_dict["name"] in ["("]:
                token_dict["whitespace_after"] = False

            last_token_dict = token_dict

        return sentence

    def process_corpus(self):
        """Read the corpus file and turns it into sentences of tagged tokens.

        The output is a list of lists, with each list being a sentence, and
        each item in the sentence being a token dictionary of the form
        ``{'name': ..., 'tags': {}}``.
        """
        sentences = []
        sentence = []
        with open(str(self.file), encoding=self.encoding) as f:

            line = f.readline()

            while line:

                if line.startswith("#"):
                    line = f.readline()
                    continue

                if line.strip().replace("﻿", "") == "":
                    if len(sentence) > 0:
                        self.infer_space_after(sentence)
                    if self.tagging_scheme is not None:
                        self.convert_tag_scheme(
                            sentence, target_scheme="iobes"
                        )

                    sentences.append(sentence)
                    sentence = []

                else:
                    fields = re.split(r"\s+", line)
                    token = fields[0]  # text column
                    token_tags = {
                        v: fields[k]
                        for k, v in self.columns.items()
                        if v != "text"
                    }
                    sentence.append({"name": token, "tags": token_tags})

                line = f.readline()

        return sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index]


def iob2(tags):
    """Check that tags have a valid IOB format.

    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == "O":
            continue
        split = tag.split("-")
        if len(split) != 2 or split[0] not in ["I", "B"]:
            return False
        if split[0] == "B":
            continue
        elif i == 0 or tags[i - 1] == "O":  # conversion IOB1 to IOB2
            tags[i] = "B" + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = "B" + tag[1:]
    return True


def iob_iobes(tags):
    """IOB -> IOBES."""
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == "O":
            new_tags.append(tag)
        elif tag.split("-")[0] == "B":
            if i + 1 != len(tags) and tags[i + 1].split("-")[0] == "I":
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace("B-", "S-"))
        elif tag.split("-")[0] == "I":
            if i + 1 < len(tags) and tags[i + 1].split("-")[0] == "I":
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace("I-", "E-"))
        else:
            raise Exception("Invalid IOB format!")
    return new_tags
