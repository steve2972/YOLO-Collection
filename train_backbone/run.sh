source /home/${USER}/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh

conda activate torch

python train.py \
    --model 2 \
    --batch-size 256 \
    --epochs 160 \
    --gpus 4 \
    --opt sgd \
    --weight-decay 0.0005 \
    --momentum 0.9 \
    --lr 0.256 \
    --lr-scheduler steplr \
    --lr-step-size 2 \
    --lr-gamma 0.97 \
    --workers 128 \
    --output-dir ./Logs \
    --wandb \
    --wandb-id Darknet19 \
