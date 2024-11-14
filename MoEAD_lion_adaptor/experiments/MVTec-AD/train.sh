PYTHONPATH=$PYTHONPATH:../../ \
CUDA_VISIBLE_DEVICES=0 \
torchrun --nproc_per_node=1 ../../tools/train_val.py