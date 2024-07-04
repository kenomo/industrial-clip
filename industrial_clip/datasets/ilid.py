import os
import pickle
import orjson
import numpy as np
from tqdm import tqdm

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


@DATASET_REGISTRY.register()
class ILID(DatasetBase):
    """Industrial Language-Image Dataset (ILID).
    """

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        if not cfg.DATASET.FOLDER:
            self.dataset_dir = os.path.join(root, "ilid")
        else:
            self.dataset_dir = os.path.join(root, cfg.DATASET.FOLDER)
        
        # dataset image_dir
        self.dataset_image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        
        # load preprocessed data if available
        if os.path.exists(self.preprocessed) and not cfg.DATASET.FORCE_PREPROCESS:
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                dataset = preprocessed["dataset"]
                label_dict = preprocessed["label_dict"]
        else:
            # dataset
            dataset_json = os.path.join(self.dataset_dir, "ilid.json")
            dataset = self.read_from_json(dataset_json)
            dataset = self.make_split(dataset, num_splits=cfg.DATASET.SPLIT.NUM_SPLITS, seed=cfg.SEED)
            
            label_dict = self.generate_unique_label_dict(
                dataset, 
                label_tag=cfg.DATASET.LABEL_TAG
            )

            preprocessed = { 
                "dataset": dataset, 
                "label_dict": label_dict 
            }
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("Using label tags for train: ", cfg.DATASET.LABEL_TAG)
        print("Using label tags for test: ", cfg.DATASET.TEST_LABEL_TAG)

        if cfg.DATASET.SPLIT.SPLIT < 0:
            no_split = True
        else:
            no_split = False

        train = self.read_data(
            self.dataset_image_dir,
            dataset, 
            label_dict, 
            train=True, 
            label_tag=cfg.DATASET.LABEL_TAG, 
            split=cfg.DATASET.SPLIT.SPLIT, 
            max_num_words=cfg.DATASET.MAX_NUM_WORDS,
            no_split=no_split
        )

        validation = self.read_data(
            self.dataset_image_dir,
            dataset, 
            label_dict, 
            train=False, 
            label_tag=cfg.DATASET.LABEL_TAG, 
            split=cfg.DATASET.SPLIT.SPLIT, 
            max_num_words=cfg.DATASET.MAX_NUM_WORDS,
            no_split=no_split
        )

        ########
        # for dassl (basically copied from super().__init__ prevents calls of static methods get_lab2cname and get_num_classes)
        self._train_x = train       # labeled training data
        self._train_u = None        # unlabeled training data (optional)
        self._val = None            # validation data (optional)
        self._test = validation     # test data

        self.set_peripherical(label_dict)

    def set_peripherical(self, label_dict):
        self._num_classes = len(label_dict)
        self._lab2cname = { a: b for b, a in label_dict.items() } # dict {label: classname}
        labels = list(self._lab2cname.keys())
        labels.sort()
        self._classnames = [self._lab2cname[l] for l in labels]

    def read_from_json(self, dataset_json):
        with open(dataset_json, "rb") as f:
            dataset = orjson.loads(f.read())
            
            num_items = len(dataset)
            print("Dataset size:", num_items)

        return dataset

    def make_split(self, dataset, num_splits=6, seed=0):
        # labels each item from dataset with a split index

        num_items = len(dataset)

        rng = np.random.default_rng(seed)
        indices = rng.random(size=num_items)
        # split index for each item
        split_indices = np.floor(indices * num_splits).astype(int)
        
        print("Splitting dataset...")
        for i, item in enumerate(dataset):
            item["split"] = split_indices[i]
        
        return dataset

    def shorten_to_x_words(self, text, x):
        words = text.split()
        shortened_words = words[:x]
        return " ".join(shortened_words)

    def generate_unique_label_dict(self, dataset, label_dict = {}, label_tag="label_short"):
        labels = [ b for a, b in label_dict.items() ]
        label_counter = 0 if len(labels) <= 0 else max(labels) + 1
        for item in dataset:
            # continue if label is empty
            if not item[label_tag]:
                continue
            # add to label_dict if not already present
            if item[label_tag] not in label_dict:
                label_dict[item[label_tag]] = label_counter
                label_counter += 1
        return label_dict

    def read_data(self, image_dir, dataset, label_dict, no_split=False, train=True, split=0, max_num_words=10, label_tag="label_short"):
        
        samples = []

        for item in tqdm(dataset):
            
            impath = os.path.join(image_dir, item["image"])
            label_text = item[label_tag]
            # continue if label is empty
            if not label_text:
                continue
            label_text = self.shorten_to_x_words(label_text, max_num_words)

            # continue if label is not in label_dict
            if item[label_tag] not in label_dict:
                continue
            label_int = label_dict[item[label_tag]]
            
            # respect split
            if not no_split:
                if train:
                    if item["split"] == split:
                        continue
                else:
                    if item["split"] != split:
                        continue
            
            samples.append(Datum(impath=impath, label=label_int, classname=label_text))
            
        return samples