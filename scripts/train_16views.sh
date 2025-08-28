ulimit -n 4096

SAVE_DIR="checkpoints/output"

torchrun --nproc_per_node=8 train.py \
    --train_dataset "5_000 @ Scannet(split='train', ROOT='data/scannet_processed', resolution=(256, 256), num_views=16) + 5_000 @ Scannetpp(split='train', ROOT='data/scannetpp_processed', resolution=(256, 256), num_views=16)" \
    --test_dataset "100 @ Scannet(split='val', ROOT='data/scannet_processed', resolution=(256, 256), seed=777, num_views=16)" \
    --train_criterion "GaussianLoss()" \
    --test_criterion "GaussianLoss()" \
    --lr 0.00005 \
    --min_lr 1e-06 \
    --warmup_epochs 4 \
    --epochs 40 \
    --batch_size 1 \
    --accum_iter 2 \
    --save_freq 1 \
    --keep_freq 5 \
    --eval_freq 1 \
    --pretrained "./pretrained_weights/checkpoint-last.pth" \
    --output_dir $SAVE_DIR
