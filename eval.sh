#!/bin/bash

COMMON_ARGUMENTS_CONFIG=(
    --root "/root/industrial-clip/data"
    --seed "1"
    --dataset-config-file "configs/datasets/ilid.yaml"
)

# use DATASET.SPLIT.SPLIT "-1" for testing on the whole dataset

COMMON_PARAMETERS_CONFIG=(
    WANDB.LOG "False"
    TRAINER.COOP.CLASS_TOKEN_POSITION "end"
    TRAINER.COOP.CTX_INIT "X X X X a photo of an industrial product"
    TRAINER.ZSCLIP.PROMPT_TEMPLATE "a photo of an industrial product {}"
    DATASET.MAX_NUM_WORDS "40"
    DATASET.FORCE_PREPROCESS "True"
    TEST.PER_CLASS_RESULT "False"
    DATALOADER.TEST.BATCH_SIZE "32"
    DATASET.SPLIT.SPLIT "0"
    EVAL.SAVE_EMBEDDINGS "False"
    DATASET.TEST_LABEL_TAG "label_short"
    DATASET.LABEL_TAG "label_short"
    DATALOADER.TEST.SAMPLER "RandomSampler"
)


###################################################################################################
# CoOp

python eval.py \
    "${COMMON_ARGUMENTS_CONFIG[@]}" \
    --output-dir "/root/industrial-clip/output/coop" \
    --trainer "CoOp" \
    --config-file "configs/trainers/CoOp/vit_b16_ilid.yaml" \
    --model-dir "/root/industrial-clip/output/coop" \
    --load-epoch "10" \
    "${COMMON_PARAMETERS_CONFIG[@]}"

###################################################################################################
# CoOpIA

python eval.py \
    "${COMMON_ARGUMENTS_CONFIG[@]}" \
    --output-dir "/root/industrial-clip/output/coop_ia" \
    --trainer "CoOpIA" \
    --config-file "configs/trainers/CoOpIA/vit_b16_ilid.yaml" \
    --model-dir "/root/industrial-clip/output/coop_ia" \
    --load-epoch "10" \
    "${COMMON_PARAMETERS_CONFIG[@]}"

###################################################################################################
# CoOpIATA

python eval.py \
    "${COMMON_ARGUMENTS_CONFIG[@]}" \
    --output-dir "/root/industrial-clip/output/coop_ia_ta" \
    --trainer "CoOpIATA" \
    --config-file "configs/trainers/CoOpIATA/vit_b16_ilid.yaml" \
    --model-dir "/root/industrial-clip/output/coop_ia_ta" \
    --load-epoch "10" \
    "${COMMON_PARAMETERS_CONFIG[@]}"

###################################################################################################
# CLIPAdapter

python eval.py \
    "${COMMON_ARGUMENTS_CONFIG[@]}" \
    --output-dir "/root/industrial-clip/output/clip_adapter" \
    --trainer "CLIPAdapter" \
    --config-file "configs/trainers/CLIPAdapter/vit_b16_ilid.yaml" \
    --model-dir "/root/industrial-clip/output/clip_adapter" \
    --load-epoch "10" \
    "${COMMON_PARAMETERS_CONFIG[@]}"

###################################################################################################
# ZeroshotCLIP

python eval.py \
    "${COMMON_ARGUMENTS_CONFIG[@]}" \
    --output-dir "/root/industrial-clip/output/zsclip" \
    --trainer "ZeroshotCLIP" \
    --config-file "configs/trainers/ZsCLIP/vit_b16.yaml" \
    "${COMMON_PARAMETERS_CONFIG[@]}"
    
###################################################################################################