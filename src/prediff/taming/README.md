# Taming Transformers for High-Resolution Image Synthesis
This subdirectory contains the implementations of `AutoencoderKL` and `LPIPSWithDiscriminator`.
All codes are adopted from [taming-transformers](https://github.com/CompVis/taming-transformers), [stable-diffusion](https://github.com/CompVis/stable-diffusion) and [Diffusers](https://huggingface.co/docs/diffusers) 

Alternatively, you can use the implementation of `AutoencoderKL` from [diffusers==0.13.0](https://github.com/huggingface/diffusers/blob/v0.13.0/src/diffusers/models/autoencoder_kl.py) by
```python
from diffusers.models import AutoencoderKL
```
and set `return_dict=False` in methods `forward`, `encode`, `_decode` and `decode`.
