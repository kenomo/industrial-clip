import os, random
import numpy as np
import torch

from yacs.config import CfgNode as CN
from dassl.config import get_cfg_default


# from https://github.com/YangYongJin/APEX/blob/master/utils.py
def set_random_seed(seed) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def reset_cfg(cfg, args):
    """
    Reset config values from input arguments.
    """
    if hasattr(args, 'root'):
        if args.root:
            cfg.DATASET.ROOT = args.root

    if hasattr(args, 'output_dir'):
        if args.output_dir:
            cfg.OUTPUT_DIR = args.output_dir

    if hasattr(args, 'do_not_resume'):
        if args.do_not_resume:
            cfg.DO_NOT_RESUME = args.do_not_resume

    if hasattr(args, 'resume'):
        if args.resume:
            cfg.RESUME = args.resume

    if hasattr(args, 'seed'):
        if args.seed:
            cfg.SEED = args.seed

    if hasattr(args, 'trainer'):
        if args.trainer:
            cfg.TRAINER.NAME = args.trainer


def extend_cfg(cfg):
    """
    Add new config variables.
    """

    cfg.DO_NOT_RESUME = False
    # override the evaluator to use the ExtendedClassification
    cfg.TEST.EVALUATOR = "ExtendedClassification"
    cfg.TEST.REPORT_LOGITS = False

    # only for the ExtendedClassification evaluator - for cfg.TEST.* already exists
    cfg.VALIDATION = CN()
    cfg.VALIDATION.PER_CLASS_RESULT = False
    cfg.VALIDATION.COMPUTE_CMAT = False

    # only for eval
    cfg.EVAL = CN()
    cfg.EVAL.SPLIT = "test"
    cfg.EVAL.SAVE_EMBEDDINGS = False

    # CoOp
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    # CoOpIA / CoOpIATA
    cfg.TRAINER.IMAGE_ADAPTER = CN()
    cfg.TRAINER.IMAGE_ADAPTER.REDUCTION = 4
    cfg.TRAINER.IMAGE_ADAPTER.RATIO = 0.5
    cfg.TRAINER.TEXT_ADAPTER = CN()
    cfg.TRAINER.TEXT_ADAPTER.REDUCTION = 4
    cfg.TRAINER.TEXT_ADAPTER.RATIO = 0.5

    # ZsCLIP
    cfg.TRAINER.ZSCLIP = CN()
    cfg.TRAINER.ZSCLIP.PROMPT_TEMPLATE = "a photo of a {}."


    # WANDB
    cfg.WANDB = CN()
    cfg.WANDB.LOG = False
    cfg.WANDB.PROJECT = "test"
    cfg.WANDB.RUN_NAME = None


    # DATASET
    cfg.DATASET.FOLDER = None
    cfg.DATASET.FORCE_PREPROCESS = False
    cfg.DATASET.MAX_NUM_WORDS = 10
    cfg.DATASET.LABEL_TAG = "label_short"
    cfg.DATASET.TEST_LABEL_TAG = "label_short"
    cfg.DATASET.SPLIT = CN()
    cfg.DATASET.SPLIT.NUM_SPLITS = 6
    cfg.DATASET.SPLIT.SPLIT = 0

    # GENERIC DATASET
    cfg.DATASET.GENERIC = CN()
    cfg.DATASET.GENERIC.IMAGES = []
    cfg.DATASET.GENERIC.PROMPTS = []


def setup_cfg(args):
    """
    Setup config.
    """

    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg