SAVE_DIR="checkpoints/output"

CUDA_VISIBLE_DEVICES=1 python test.py \
    --test_dataset "TestDataset(split='test', ROOT='data/scannet_test', resolution=(256, 256), seed=777, num_views=16)" \
    --test_criterion "TestLoss(pose_align_steps=100, num_views=16)" \
    --pretrained ./ckpt/checkpoint-last.pth
    # --test_criterion "TestLoss(pose_align_steps=100, num_views=16)" \
    # --pretrained ./checkpoints/checkpoint-last.pth