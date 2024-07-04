import torch
import torch.nn as nn

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import convert_weights
from industrial_clip.engine.trainer import TrainerPP

from .coop import load_clip_to_cpu


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # image adapter
        self.image_adapter = Adapter(clip_model.ln_final.weight.shape[0], cfg.TRAINER.IMAGE_ADAPTER.REDUCTION).to(self.dtype)
        self.image_adapter_ratio = cfg.TRAINER.IMAGE_ADAPTER.RATIO

        # text adapter
        self.text_adapter = Adapter(clip_model.ln_final.weight.shape[0], cfg.TRAINER.IMAGE_ADAPTER.REDUCTION).to(self.dtype)
        self.text_adapter_ratio = cfg.TRAINER.IMAGE_ADAPTER.RATIO
        
        # build prompts
        prompt = cfg.TRAINER.ZSCLIP.PROMPT_TEMPLATE
        prompts = [prompt.format(c.replace("_", " ")) for c in classnames]

        print(f"Prompts: {prompts}")
        self.prompts = torch.cat([clip.tokenize(p) for p in prompts])
        self.prompts = self.prompts.to(device)

    def forward(self, image, label):
        #
        image_features = self.clip_model.encode_image(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        x = self.image_adapter(image_features)
        image_features = self.image_adapter_ratio * x + (1 - self.image_adapter_ratio) * image_features

        #
        text_features = self.clip_model.encode_text(self.prompts[label])
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
class CLIPAdapter(TrainerPP):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, self.device)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "image_adapter" in name:
                param.requires_grad_(True)
            elif "text_adapter" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        print("Trainable parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)

        self.model.to(self.device)
        
        # give model to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("clip_adapter", self.model, self.optim, self.sched)