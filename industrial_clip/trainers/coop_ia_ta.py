import os
import torch
import torch.nn as nn

from dassl.engine import TRAINER_REGISTRY
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from industrial_clip.engine.trainer import TrainerPP
from industrial_clip.trainers.coop import CoOp
from industrial_clip.trainers.coop import CustomCLIP as _CustomCLIP
from industrial_clip.trainers.coop import load_clip_to_cpu

from .clip_adapter import Adapter


class CustomCLIP(_CustomCLIP):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)

        self.image_adapter = Adapter(clip_model.ln_final.weight.shape[0], cfg.TRAINER.IMAGE_ADAPTER.REDUCTION).to(clip_model.dtype)
        self.image_adapter_ratio = cfg.TRAINER.IMAGE_ADAPTER.RATIO

        self.text_adapter = Adapter(clip_model.ln_final.weight.shape[0], cfg.TRAINER.IMAGE_ADAPTER.REDUCTION).to(clip_model.dtype)
        self.text_adapter_ratio = cfg.TRAINER.IMAGE_ADAPTER.RATIO

    def forward(self, image, label):
        #
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        x = self.image_adapter(image_features)
        image_features = self.image_adapter_ratio * x + (1 - self.image_adapter_ratio) * image_features

        #
        token_embeddings_prefix, token_embeddings_suffix = self.prompt_learner.generate_embeddings(label)
        prompts = self.prompt_learner(label, token_embeddings_prefix, token_embeddings_suffix)
        text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts[label])
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        x = self.text_adapter(text_features)
        text_features = self.text_adapter_ratio * x + (1 - self.text_adapter_ratio) * text_features

        #
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits, image_features, text_features


@TRAINER_REGISTRY.register()
class CoOpIATA(CoOp):

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if name == "prompt_learner.ctx":
                param.requires_grad_(True)
            elif "image_adapter" in name:
                param.requires_grad_(True)
            elif "text_adapter" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        print("Trainable parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # give model to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("coop_ia_ta", self.model, self.optim, self.sched)
