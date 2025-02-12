#!/usr/bin/env bash
set -x

export MASTER_PORT=${MASTER_PORT:-12322}

OUTPUT_DIR='/home/dani/data/results/vit_b_pt_600e_part2'
DATA_PATH='/mnt/hdd1/data/Meta/Pretraining/video_files_list_part2.csv'
DATA_ROOT='/mnt/hdd1/data'
#MODEL_PATH='/home/dani/data/params/vit_b_k710_dl_from_giant.pth'
MODEL_PATH='/home/dani/data/results/vit_b_pt_300e_scratch_10000/checkpoint-299.pth'

N_NODES=${N_NODES:-1}  # Number of nodes
GPUS_PER_NODE=${GPUS_PER_NODE:-4}  # Number of GPUs in each node
SRUN_ARGS=${SRUN_ARGS:-""}  # Other slurm task args
PY_ARGS=${@:3}  # Other training args

# sampling_rate = downsample factor for videos
# num_sample = number of differently-augmented views of the same instance
# effective batch size = batch_size * num_sample

# batch_size can be adjusted according to the graphics card
# batch_size = 3 (two A4000 GPUs)
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} \
        --master_port ${MASTER_PORT} --nnodes=${N_NODES} --node_rank=$1 --master_addr=$2 \
        run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --data_root ${DATA_ROOT} \
        --finetune ${MODEL_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --decoder_mask_type run_cell \
        --decoder_mask_ratio 0.5 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 6 \
        --num_sample 4 \
        --num_frames 16 \
        --sampling_rate 10 \
        --num_workers 10 \
        --lr 0.5e-3 \
	--clip_grad 0.1 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 10 \
        --save_ckpt_freq 5 \
	--start_epoch 300 \
        --epochs 600 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        ${PY_ARGS}
