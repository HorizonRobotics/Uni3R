SAVE_DIR="checkpoints/output"

python test.py \
    --test_dataset "TestDataset(split='test', ROOT='/home/data/scannet_test', resolution=(256, 256), seed=777, num_views=16)" \
    --test_criterion "TestLoss(pose_align_steps=100, num_views=16)" \
    --pretrained checkpoints/output/16views.pth

