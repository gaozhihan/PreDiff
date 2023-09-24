# Train PreDiff on SEVIR-LR dataset

## Configurations for training and inference
Change the configurations in [corresponding cfg.yaml](cfg.yaml)
You might consider modifying the following configurations according to your specific requirements:
- `trainer.check_val_every_n_epoch`: Run validation every `n` training epochs. Set a larger value for it if you want to alleviate the time costs in validation.
- `vis.eval_example_only`: If `true`, only data with indices in the `example_data_idx_list` will be evaluated. Set it to `false` if you want to evaluate the whole val/test set.
- `vis.eval_aligned`: If `true`, PreDiff-KA will be evaluated.
- `vis.eval_unaligned`: If `true`, PreDiff without knowledge alignment will be evaluated.
- `vis.num_samples_per_context`: Generate `n` samples for each context sequence.
- `model.align.alignment_type`: `null` by default means not to load the knowledge alignment module. Setting it to `avg_x` for knowledge alignment with anticipated future average intensity.
- `model.align.model_ckpt_path`: Point it to your own pretrained checkpoint if you want a custom knowledge alignment network.
- `model.vae.pretrained_ckpt_path`: Point it to your own pretrained checkpoint if you want a custom vae.

## Commands for training and inference
Run the following command to train PreDiff on SEVIR-LR dataset.
```bash
cd ROOT_DIR/PreDiff
MASTER_ADDR=localhost MASTER_PORT=10001 python ./scripts/prediff/sevirlr/train_sevirlr_prediff.py --gpus 2 --cfg ./scripts/prediff/sevirlr/cfg.yaml --save tmp_sevirlr_prediff
```
Or run the following command to directly load pretrained checkpoint for test.
```bash
cd ROOT_DIR/PreDiff
MASTER_ADDR=localhost MASTER_PORT=10001 python ./scripts/prediff/sevirlr/train_sevirlr_prediff.py --gpus 2 --pretrained --save tmp_sevirlr_prediff
```
Run the following command to train PreDiff using multi-node DDP.
```bash
# On the master node
MASTER_ADDR=localhost MASTER_PORT=10001 WORLD_SIZE=16 NODE_RANK=0 python ./scripts/prediff/sevirlr/train_sevirlr_prediff.py --nodes 2 --gpus 2 --cfg ./scripts/prediff/sevirlr/train_sevirlr_prediff/cfg.yaml --save tmp_sevirlr_prediff
# On the 1st node
MASTER_ADDR=$master_ip MASTER_PORT=10001 WORLD_SIZE=16 NODE_RANK=1 python ./scripts/prediff/sevirlr/train_sevirlr_prediff.py --nodes 2 --gpus 2 --cfg ./scripts/prediff/sevirlr/train_sevirlr_prediff/cfg.yaml --save tmp_sevirlr_prediff
```
Run the following command to upload and share experiment records via [tensorboard dev](https://tensorboard.dev/)
```bash
cd ROOT_DIR/PreDiff
tensorboard dev upload --logdir ./experiments/tmp_sevirlr_prediff/lightning_logs --name 'tmp_sevirlr_prediff'
```
