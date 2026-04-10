## Arbitrary-view model

### Training
After preparing the datasets, you can train the model using the following command:
```bash
bash scripts/train_arbitrary.sh
```

The training results will be saved to `SAVE_DIR`. By default, it is set to `checkpoints/output`.

### Evaluation
First, please download arbitrary model checkpoints from [here](https://huggingface.co/HorizonRobotics/Uni3R/tree/main).

Then run these scripts to do evaluation on ScanNet Dataset, including 4, 8 and 16 views.

```bash
bash scripts/test_4views.sh
bash scripts/test_8views.sh
bash scripts/test_16views.sh
```
