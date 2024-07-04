import torch
import torch.nn as nn

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import convert_weights
from industrial_clip.engine.trainer import TrainerPP

from .coop import load_clip_to_cpu


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        # build prompts
        prompt = cfg.TRAINER.ZSCLIP.PROMPT_TEMPLATE
        prompts = [prompt.format(c.replace("_", " ")) for c in classnames]

        print(f"Prompts: {prompts}")
        self.prompts = torch.cat([clip.tokenize(p) for p in prompts])
        self.prompts = self.prompts.to(device)

    @torch.no_grad()
    def forward(self, image, label):
        image_features = self.clip_model.encode_image(image.type(self.dtype))
        text_features = self.clip_model.encode_text(self.prompts[label])

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits, image_features, text_features


@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerPP):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, self.device)

        print("Turning off gradients for CLIP model")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)

        self.model.to(self.device)