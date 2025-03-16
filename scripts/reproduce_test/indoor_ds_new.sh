#!/bin/bash -l
# a indoor_ds model with the pos_enc impl bug fixed.

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

data_cfg_path="configs/data/scannet_test_1500.py"
main_cfg_path="configs/loftr/indoor/scannet/loftr_ds_eval_new.py"
# ckpt_path="logs/tb_logs/indoor-ds-bs=20/version_14/checkpoints/epoch=18-auc@5=0.657-auc@10=0.793-auc@20=0.876.ckpt"
ckpt_path="logs/tb_logs/indoor-ds-bs=20/version_15/checkpoints/epoch=19-auc@5=0.675-auc@10=0.803-auc@20=0.880.ckpt"
dump_dir="dump/loftr_ds_indoor_new"
profiler_name="inference"
n_nodes=1  # mannually keep this the same with --nodes
n_gpus_per_node=1
torch_num_workers=4
batch_size=1  # per gpu

python -u ./test.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --ckpt_path=${ckpt_path} \
    --dump_dir=${dump_dir} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers}\
    --profiler_name=${profiler_name} \
    --benchmark 
    