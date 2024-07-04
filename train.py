import argparse
import sys
import os

from dassl.utils import setup_logger
from dassl.engine import build_trainer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "CLIP"))

from industrial_clip.utils import set_random_seed, setup_cfg
import industrial_clip.datasets.ilid
import industrial_clip.datasets.generic
import industrial_clip.trainers.zsclip
import industrial_clip.trainers.coop
import industrial_clip.trainers.coop_ia
import industrial_clip.trainers.coop_ia_ta
import industrial_clip.trainers.clip_adapter


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED > 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    else:
        raise NotImplementedError
    
    setup_logger(cfg.OUTPUT_DIR)

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("--seed", type=int, default=-1, help="only positive value enables a fixed seed")
    parser.add_argument("--do-not-resume", action="store_true", help="do not resume training")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument("--config-file", type=str, default="", help="path to config file")
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")

    # remainder overrides
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )

    args = parser.parse_args()
    main(args)
