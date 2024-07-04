#!/bin/bash

COMMON_ARGUMENTS_CONFIG=(
    --root "/root/industrial-clip/data"
    --seed "1"
    --do-not-resume
    --dataset-config-file "configs/datasets/ilid.yaml"
)
COMMON_PARAMETERS_CONFIG=(
    WANDB.LOG "True"
    WANDB.PROJECT "industrial-clip"
    OPTIM.LR "0.15"
    OPTIM.MAX_EPOCH "10"
    DATALOADER.TRAIN_X.BATCH_SIZE "64"
    DATALOADER.TEST.BATCH_SIZE "32"
    DATASET.FORCE_PREPROCESS "True"
    TEST.PER_CLASS_RESULT "False"
    TRAINER.COOP.CLASS_TOKEN_POSITION "end"
    TRAINER.COOP.CTX_INIT "X X X X a photo of an industrial product"
    TRAINER.ZSCLIP.PROMPT_TEMPLATE "a photo of an industrial product {}"
    DATASET.MAX_NUM_WORDS "40"
)


###################################################################################################
# CoOp

python train.py \
    "${COMMON_ARGUMENTS_CONFIG[@]}" \
    --trainer "CoOp" \
    --config-file "configs/trainers/CoOp/vit_b16_ilid.yaml" \
    --output-dir "/root/industrial-clip/output/coop" \
    WANDB.RUN_NAME "coop" \
    DATASET.SPLIT.SPLIT "0" \
    DATASET.LABEL_TAG "label_short" \
    DATASET.TEST_LABEL_TAG "label_short" \
    "${COMMON_PARAMETERS_CONFIG[@]}"


###################################################################################################
# CoOpIA

python train.py \
    "${COMMON_ARGUMENTS_CONFIG[@]}" \
    --trainer "CoOpIA" \
    --config-file "configs/trainers/CoOpIA/vit_b16_ilid.yaml" \
    --output-dir "/root/industrial-clip/output/coop_ia" \
    WANDB.RUN_NAME "coop_ia" \
    DATASET.SPLIT.SPLIT "0" \
    DATASET.LABEL_TAG "label_short" \
    DATASET.TEST_LABEL_TAG "label_short" \
    "${COMMON_PARAMETERS_CONFIG[@]}"


###################################################################################################
# CoOpIATA

python train.py \
    "${COMMON_ARGUMENTS_CONFIG[@]}" \
    --trainer "CoOpIATA" \
    --config-file "configs/trainers/CoOpIATA/vit_b16_ilid.yaml" \
    --output-dir "/root/industrial-clip/output/coop_ia_ta" \
    WANDB.RUN_NAME "coop_ia_ta" \
    DATASET.SPLIT.SPLIT "0" \
    DATASET.LABEL_TAG "label_short" \
    DATASET.TEST_LABEL_TAG "label_short" \
    "${COMMON_PARAMETERS_CONFIG[@]}"

###################################################################################################
# CLIPAdapter

python train.py \
    "${COMMON_ARGUMENTS_CONFIG[@]}" \
    --trainer "CLIPAdapter" \
    --config-file "configs/trainers/CLIPAdapter/vit_b16_ilid.yaml" \
    --output-dir "/root/industrial-clip/output/clip_adapter" \
    WANDB.RUN_NAME "clip_adapter" \
    DATASET.SPLIT.SPLIT "0" \
    DATASET.LABEL_TAG "label_short" \
    DATASET.TEST_LABEL_TAG "label_short" \
    "${COMMON_PARAMETERS_CONFIG[@]}"

###################################################################################################
# ZeroshotCLIP

python eval.py \
    --root "/root/industrial-clip/data" \
    --seed "1" \
    --dataset-config-file "configs/datasets/ilid.yaml" \
    --trainer "ZeroshotCLIP" \
    --config-file "configs/trainers/ZsCLIP/vit_b16.yaml" \
    --output-dir "/root/industrial-clip/output/zeroshot_clip" \
    WANDB.RUN_NAME "zeroshot_clip" \
    DATASET.SPLIT.SPLIT "0" \
    DATASET.LABEL_TAG "label_short" \
    DATASET.TEST_LABEL_TAG "label_short" \
    "${COMMON_PARAMETERS_CONFIG[@]}"

###################################################################################################