source /home/${USER}/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh

conda activate torch

python main.py \
    --model 3 \
    --batch-size 256 \
    --gpus 1 \
    --workers 128 \
    --output-dir ./Logs \
    --resume Logs/Darknet\ Backbone/5uakhel1/checkpoints/epoch=159-step=800800.ckpt \
    --test-only 
