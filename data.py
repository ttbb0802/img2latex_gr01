from os.path import join

from torch.utils.data import Dataset
from utils import load_and_transform_image

import torch


class Im2LatexDataset(Dataset):
    def __init__(self, data_dir, image_dir, split, max_len):
        """args:
        data_dir: root dir storing the prepoccessed data
        split: train, validate or test
        """
        assert split in ["train", "validate", "test"]
        self.data_dir = data_dir
        self.image_dir = image_dir
        self.split = split
        self.max_len = max_len
        self.pairs = self._load_pairs()

    def _load_pairs(self):
        pairs = torch.load(join(self.data_dir, "{}.pkl".format(self.split)))
        return pairs

    def __getitem__(self, index):
        img_name = self.pairs[index][0]
        formula  = self.pairs[index][1]
        img_tensor = load_and_transform_image(join(self.image_dir, img_name))
        formula = " ".join(formula.split()[:self.max_len])
        pair = (img_tensor, formula)

        return pair

    def __len__(self):
        return len(self.pairs)