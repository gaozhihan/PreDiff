# Train knowledge alignment network on SEVIR-LR dataset

## Commands for training
Run the following command to train knowledge alignment network on SEVIR-LR dataset.
Change the configurations in [corresponding cfg.yaml](cfg.yaml)
```bash
cd ROOT_DIR/PreDiff
MASTER_ADDR=localhost MASTER_PORT=10001 python ./scripts/alignment/sevirlr/train_sevirlr_avg_x.py --gpus 2 --cfg ./scripts/alignment/sevirlr/cfg.yaml --save tmp_sevirlr_avg_x
```
Run the tensorboard command to visualize the experiment records
```bash
cd ROOT_DIR/PreDiff
tensorboard --logdir ./experiments/tmp_sevirlr_avg_x/lightning_logs
```
