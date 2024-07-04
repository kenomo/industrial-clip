import argparse
import sys
import os
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

from yacs.config import CfgNode as CN
from dassl.utils import setup_logger
from dassl.engine import build_trainer
from dassl.config import get_cfg_default

from .utils import extend_cfg
import industrial_clip.datasets.ilid
import industrial_clip.datasets.generic
import industrial_clip.trainers.zsclip
import industrial_clip.trainers.coop
import industrial_clip.trainers.coop_ia
import industrial_clip.trainers.coop_ia_ta
import industrial_clip.trainers.clip_adapter


class Evaluation:

    def __init__(self, trainer, cfg_list, prompts, images = [], model_dir="", epoch=0):
        
        self.cfg = get_cfg_default()
        extend_cfg(self.cfg)
        self.cfg.merge_from_list(cfg_list)

        self.cfg.DO_NOT_RESUME = True
        self.cfg.TRAINER.NAME = trainer
        self.cfg.DATASET.NAME = "Generic"
        self.cfg.MODEL.BACKBONE.NAME = "ViT-B/16"
        self.cfg.DATASET.GENERIC.IMAGES = images
        self.cfg.DATASET.GENERIC.PROMPTS = prompts
    
        self.cfg.freeze()

        self.model_dir = model_dir
        self.epoch = epoch        

    def build(self):
        self.trainer = build_trainer(self.cfg)
        self.trainer.load_model(self.model_dir, epoch=self.epoch)

    def forward(self, images, prompt_idx):
        images = images.to(self.trainer.model.dtype).to(self.trainer.device)
        prompt_idx = torch.tensor(prompt_idx)
        prompt_idx = prompt_idx.to(self.trainer.device)
        output = self.trainer.forward(images, prompt_idx)
        
        logits = F.softmax(output, dim=1).cpu()
        prompts = np.asarray(self.cfg.DATASET.GENERIC.PROMPTS)[prompt_idx.cpu()]
        pred = torch.argmax(logits, dim=1).numpy()
        predicted_prompts = prompts[pred]
        return logits, predicted_prompts
    
    @property
    def visual_resolution(self):
        d = self.trainer.model.clip_model.visual.input_resolution
        return (d, d)

    def load_images(self, images, normalize=False):
        transforms_list = []

        transforms_list.append(transforms.Resize(self.visual_resolution))
        transforms_list.append(lambda x: x.convert("RGB"))
        transforms_list.append(transforms.ToTensor())
        if normalize:
            transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)))

        transform = transforms.Compose(transforms_list)
        image_tensors = [transform(Image.open(image_path))[0:3, :, :] for image_path in images]
        return torch.stack(image_tensors)
        