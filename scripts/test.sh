SAVE_DIR="checkpoints/output"

python test.py \
    --test_dataset "TestDataset(split='test', ROOT='/home/data/scannet_test', resolution=(256, 256), seed=777, num_views=2)" \
    --test_criterion "TestLoss(pose_align_steps=100, num_views=2)" \
    --pretrained checkpoints/output/2views.pth
