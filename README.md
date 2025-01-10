# Industrial Language-Image Dataset (ILID): Adapting Vision Foundation Models for Industrial Settings

This repository provides training- and evaluation-related code to [Industrial Language-Image Dataset (ILID): Adapting Vision Foundation Models for Industrial Settings](https://doi.org/10.48550/arXiv.2406.09637). We provide the dataset at [github.com/kenomo/ilid](https://github.com/kenomo/ilid). 
This repository builds upon the code base of [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch), uses the original [CLIP](https://github.com/openai/CLIP) implementation, and contains a `devcontainer.json` and `Dockerfile`, which setups all the necessary dependencies. The training and code was tested on a 4090.

## 🏋️ Training
We provide trainers for [CLIP-Adapter](https://arxiv.org/abs/2110.04544), [CoOp](https://arxiv.org/abs/2109.01134), zero-shot CLIP and combination of CoOp and adapters in [industrial_clip/trainers](industrial_clip/trainers).
For training use [train.py](./train.py), for evaluation only use [eval.py](./eval.py). You will find example configuration files under [configs/](configs/), datasets must be placed under [data/](data/). For `ILID` create a folder structure as:
```
│
├── data/
│   └── ilid/
│       ├── images/
│       │   └── ...     # downloaded images
│       └── ilid.json   # dataset json file
│
├── ...
```
You can find example run configurations in [train.sh](train.sh) and [eval.sh](eval.sh). If you want to use [W&B](https://wandb.ai/site) to track your runs, you will need an API key and export it as `WANDB_API_KEY`.
We extended Dassl's configuration parameters (s. `/root/dassl/configs` and `/root/dassl/dassl/config/defaults.py`), which you will find in [utils.py](industrial_clip/utils.py).

## 📈 Evaluation
We provide a set of notebooks for evaluation:

1. [cross_validation.ipynb](evaluation/cross_validation.ipynb): Get cross-validation results of multiple runs from [W&B](https://wandb.ai/site).
2. [embeddings.ipynb](evaluation/embeddings.ipynb): Generate TSNE diagrams. Before running the notebook, you have to set the configuration flag `EVAL.SAVE_EMBEDDINGS` to `True`to save image and text encoder embeddings.
3. [prompting.ipynb](evaluation/prompting.ipynb): Example for prompting.
4. [material_prompting.ipynb](evaluation/material_prompting.ipynb): Example for prompting for materials.
5. [samclip.ipynb](evaluation/samclip.ipynb): Example for language-guided segmentation with [SAM](https://github.com/facebookresearch/segment-anything). You have to download a chackpoint before:
    ```
    curl -L -o /root/industrial-clip/evaluation/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    ```

## ☎ Contact
You are welcome to submit issues, send pull requests, or share some ideas with us. If you have any other questions, please contact 📧: [Keno Moenck](mailto:keno.moenck@tuhh.de).

## ✍ Citation
If you find __ILID__ or the provided code useful to your work/research, please cite:
```bibtex
@article{Moenck.2024,
  title = {Industrial {{Language-Image Dataset}} ({{ILID}}): {{Adapting Vision Foundation Models}} for {{Industrial Settings}}},
  author = {Moenck, Keno and Thieu, Duc Trung and Koch, Julian and Sch{\"u}ppstuhl, Thorsten},
  year = {2024},
  journal = {Procedia CIRP},
  series = {57th {{CIRP Conference}} on {{Manufacturing Systems}} 2024 ({{CMS}} 2024)},
  volume = {130},
  pages = {250--263},
  issn = {2212-8271},
  doi = {10.1016/j.procir.2024.10.084}
}

@misc{Moenck.14.06.2024,
  author = {Moenck, Keno and Thieu, Duc Trung and Koch, Julian and Sch{\"u}ppstuhl, Thorsten},
  title = {Industrial Language-Image Dataset (ILID): Adapting Vision Foundation Models for Industrial Settings},
  date = {14.06.2024},
  year = {2024},
  url = {http://arxiv.org/pdf/2406.09637},
  doi = {https://doi.org/10.48550/arXiv.2406.09637}
}
```

## 🙏 Acknowledgment
[Dassl](https://github.com/KaiyangZhou/Dassl.pytorch), [CoOp](https://github.com/KaiyangZhou/CoOp), and [APEX](https://github.com/YangYongJin/APEX) that helped during the course of this work.
