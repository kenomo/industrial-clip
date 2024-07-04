import os
import pickle
import orjson
import numpy as np
from tqdm import tqdm

from dassl.data.datasets import DATASET_REGISTRY, DatasetBase
from dassl.data.datasets import Datum as _Datum


# just a clone of the original Datum class to prevent already checking for files during initialization
class Datum(_Datum):
    def __init__(self, impath="", label=0, domain=0, classname=""):
        assert isinstance(impath, str)
        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname


@DATASET_REGISTRY.register()
class Generic(DatasetBase):
    """Generic dataset for testing purposes.
    The dataset can be used as an image opaque dataset:
    If no images are provided, the dataset will be empty. Labels can be used to specify the classnames. Batch must then be assembled customly.
    """

    dataset_dir = "generic"

    def __init__(self, cfg):
        
        # list of images
        images = cfg.DATASET.GENERIC.IMAGES
        prompts = cfg.DATASET.GENERIC.PROMPTS

        prompt_dict = {}
        for p in prompts:
            if p not in prompt_dict:
                prompt_dict[p] = len(prompt_dict)

        if len(images) > 0:
            # create dataset with images
            dataset = [Datum(impath=image, label=prompt_dict[prompts[i]], classname=prompts[i]) for i, image in enumerate(images)]
        else:
            # add opaque images to prevent errors
            dataset = [Datum(impath="", label=prompt_dict[prompts[i]], classname=prompts[i]) for i, _ in enumerate(prompts)]

        ########
        # for dassl (basically copied from super().__init__ prevents calls of static methods get_lab2cname and get_num_classes)
        self._train_x = dataset
        self._train_u = None
        self._val = None
        self._test = dataset

        self._num_classes = len(prompt_dict)
        self._lab2cname = { a: b for b, a in prompt_dict.items() } # dict {label: classname}
        lab2cname_list = list(self._lab2cname.keys())
        lab2cname_list.sort()
        self._classnames = [self._lab2cname[label] for label in lab2cname_list]