# PreDiff

By [Zhihan Gao](https://scholar.google.com/citations?user=P6ACUAUAAAAJ), 
[Xingjian Shi](https://github.com/sxjscience), 
[Boran Han](https://scholar.google.com/citations?user=Prwxh24AAAAJ), 
[Hao Wang](http://www.wanghao.in/), 
[Xiaoyong Jin](https://scholar.google.com/citations?user=EWiYf7YAAAAJ), 
[Danielle Maddix Robinson](https://dcmaddix.github.io/),
[Yi Zhu](https://bryanyzhu.github.io/), 
[Mu Li](https://github.com/mli), 
[Yuyang Bernie Wang](http://web.mit.edu/~ywang02/www/).

This repo contains the official implementation of the ["PreDiff: Precipitation Nowcasting with Latent Diffusion Models"](https://openreview.net/pdf?id=Gh67ZZ6zkS) paper accepted to NeurIPS 2023.

## Introduction
Earth system forecasting has traditionally relied on complex physical models that are computationally expensive and require significant domain expertise. 
In the past decade, the unprecedented increase in spatiotemporal Earth observation data has enabled data-driven forecasting models using deep learning techniques. 
These models have shown promise for diverse Earth system forecasting tasks but either struggle with handling uncertainty or neglect domain-specific prior knowledge, resulting in averaging possible futures to blurred forecasts or generating physically implausible predictions. 
To address these limitations, we propose a **two-stage pipeline for probabilistic spatiotemporal forecasting**: 
1) We develop PreDiff, a conditional latent diffusion model capable of probabilistic forecasts. 
2) We incorporate an explicit **knowledge alignment** mechanism to align forecasts with domain-specific physical constraints. This is achieved by estimating the deviation from imposed constraints at each denoising step and adjusting the transition distribution accordingly. 

We conduct empirical studies on two datasets: N-body MNIST, a synthetic dataset with chaotic behavior, and SEVIR, a real-world precipitation nowcasting dataset. 
Specifically, we impose the law of conservation of energy in N-body MNIST and anticipated precipitation intensity in SEVIR. 
Experiments demonstrate the effectiveness of PreDiff in handling uncertainty, incorporating domain-specific prior knowledge, and generating forecasts that exhibit high operational utility.  

![teaser](figures/method/teaser_v1.png)

**Overview of PreDiff inference with knowledge alignment.**
An observation sequence $y$ is encoded into a latent context $z_{\text{cond}}$ by the frame-wise encoder $\mathcal{E}$. 
The latent diffusion model $p_\theta(z_t|z_{t+1}, z_{\text{cond}})$, which is parameterized by an Earthformer-UNet, then generates the latent future $z_0$ by autoregressively denoising Gaussian noise $z_T$ conditioned on $z_{\text{cond}}$.
It takes the concatenation of the latent context $z_{\text{cond}}$ and the previous-step noisy latent future $z_{t+1}$ as input, and outputs $z_t$.
The transition distribution of each step from $z_{t+1}$ to $z_t$ can be further refined via knowledge alignment, according to auxiliary prior knowledge.
$z_0$ is decoded back to pixel space by the frame-wise decoder $\mathcal{D}$ to produce the final prediction $\hat{x}$.

**Qualitative Analysis on SEVIR**
![exp_vis](figures/exp/sevir_vis_both_v2.png)
(a) PreDiff succeeds in keeping the correct patterns, which can be crucial for recognizing weather events.
(b) PreDiff-KA (PreDiff with knowledge alignment) is flexible at highlighting possible extreme cases like rainstorms and droughts.

## Installation
Create Conda environment
```bash
conda create --name prediff python=3.10.12
conda activate prediff
```
Install PyTorch and PyTorch-Lightning with correct CUDA support
```bash
python -m pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install lightning==2.0.9
```
Install PreDiff in dev mode
```bash
cd ROOT_DIR/PreDiff
python -m pip install -e . --no-build-isolation
```

## Datasets
We adopted a downsampled version of [Storm EVent ImageRy (SEVIR) dataset](https://sevir.mit.edu/), denoted as SEVIR-LR, where the temporal downscaling factor is 2, and the spatial downscaling factor is 3 for each dimension. 
On SEVIR-LR dataset, PreDiff generates $6\times 128\times 128$ forecasts for a given $7\times 128\times 128$ context sequence.

To download SEVIR-LR dataset directly from AWS S3, run:
```bash
cd ROOT_DIR/PreDiff
python ./scripts/datasets/sevir/download_sevirlr.py
```
We can also let the [`SEVIRLightningDataModule`](./src/prediff/datasets/sevir/sevir_torch_wrap.py) do it for you automatically the first time you call `prepare_data()`.

Alternatively, if you already have the original SEVIR dataset, you may want to get SEVIR-LR by downsampling the original SEVIR. In this case run:
```bash
cd ROOT_DIR/PreDiff
ln -s path_to_SEVIR ./datasets/sevir  # link to your SEVIR dataset if it is not in `ROOT_DIR/PreDiff/datasets`
python ./scripts/datasets/sevir/downsample_sevir.py
```

## Training Script and Pretrained Models
### Test pretrained PreDiff
Run the following command to download all pretrained weights in advance. 
Use `--model` flag to download a specific pretrained model component. 
The available candidates are `vae`, `earthformerunet`, `alignment`, and `all`.
```bash
cd ROOT_DIR/PreDiff
python ./scripts/download_pretrained.py --model all
``` 

Run the following commands to load pretrained models for inference on SEVIR-LR dataset, following the [instruction](./scripts/prediff/sevirlr/README.md).
```bash
cd ROOT_DIR/PreDiff
MASTER_ADDR=localhost MASTER_PORT=10001 python ./scripts/prediff/sevirlr/train_sevirlr_prediff.py --gpus 2 --pretrained --save tmp_sevirlr_prediff
```
The results will be saved to directory `ROOT_DIR/PreDiff/experiments/tmp_sevirlr_prediff`.

Notice that since the inference is extremely time-consuming, the inference is only done for those example sequences for visualization. 
To evaluate the whole val/test sets, please set `vis.eval_example_only: false` in the [config](./scripts/prediff/sevirlr/cfg.yaml).

### Train from scratch
Our **two-stage pipeline** sequentially trains PreDiff and the knowledge alignment network. 
The training of PreDiff is further decomposed into two sequential phases: training the VAE and the latent Earthformer-UNet. 
To train all components from scratch, follow these sequential steps: 
1. Train the VAE.
2. Train the latent Earthformer-UNet with the VAE trained in step 1.
3. Train the knowledge alignment network with the VAE trained in step 1.

In practice, the training of the knowledge alignment network is independent of the training of the latent Earthformer-UNet. 
Therefore, steps 2 and 3 can be performed in parallel. 
To achieve this, specify the path to the PyTorch state_dict of the VAE trained in step 1 by setting `vae.pretrained_ckpt_path` in the corresponding config files.  

Find detailed instructions in how to train the models or running inference with our pretrained models in the corresponding script folder.

| Model Component             | Script Folder                          | Config                                                          |
|-----------------------------|----------------------------------------|-----------------------------------------------------------------|
| VAE                         | [scripts](./scripts/vae/sevirlr)       | [config](./scripts/vae/sevirlr/vae_sevirlr_v1.yaml)             |
| Latent Earthformer-UNet     | [scripts](./scripts/prediff/sevirlr)   | [config](./scripts/prediff/sevirlr/prediff_sevirlr_v1.yaml)     |
| Knowledge Alignment Network | [scripts](./scripts/alignment/sevirlr) | [config](./scripts/alignment/sevirlr/alignment_sevirlr_v1.yaml) |

## Citing PreDiff

```
@inproceedings{gao2023prediff,
  title={PreDiff: Precipitation Nowcasting with Latent Diffusion Models},
  author={Gao, Zhihan and Shi, Xingjian and Han, Boran and Wang, Hao and Jin, Xiaoyong and Maddix, Danielle C and Zhu, Yi and Li, Mu and Wang, Bernie},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```

## Credits
Third-party libraries:
- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://lightning.ai/)
- [OpenCV](https://opencv.org/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)
- [OmegaConf](https://github.com/omry/omegaconf)
- [YACS](https://github.com/rbgirshick/yacs)
- [Pillow](https://python-pillow.org/)
- [scikit-learn](https://scikit-learn.org/stable/)

## License

This project is licensed under the Apache-2.0 License.
