# Train VAE from scratch on SEVIR-LR (downsampled low-resolution version) frames

## Commands for training
Run the following command to train VAE on SEVIR-LR frames. 
Change the configurations in [corresponding cfg.yaml](cfg.yaml)
```bash
cd ROOT_DIR/PreDiff
MASTER_ADDR=localhost MASTER_PORT=10001 python ./scripts/vae/sevirlr/train_vae_sevirlr.py --gpus 2 --cfg ./scripts/vae/sevirlr/cfg.yaml --save tmp_vae_sevirlr
```
Or run the following command to directly load pretrained checkpoint for test.
```bash
MASTER_ADDR=localhost MASTER_PORT=10001 python ./scripts/vae/sevirlr/train_vae_sevirlr.py --gpus 2 --pretrained --save tmp_vae_sevirlr
```
Run the tensorboard command to upload experiment records
```bash
cd ROOT_DIR/PreDiff
tensorboard dev upload --logdir ./experiments/tmp_vae_sevirlr/lightning_logs --name 'tmp_vae_sevirlr'
```
