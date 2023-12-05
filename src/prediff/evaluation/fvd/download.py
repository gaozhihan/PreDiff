import requests
from tqdm import tqdm
import os
import torch

from ...utils.path import default_pretrained_metrics_dir
from ...utils.download import (
    download_pretrained_weights,
    pretrained_i3d_400_name,
    pretrained_i3d_600_name,
)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination, chunk_size=8192):
    pbar = tqdm(total=0, unit='iB', unit_scale=True)
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    pbar.close()


def download(id='1mQK8KD8G6UWRa5t87SRMm5PVXtlpneJT',
             fname="i3d_pretrained_400.pt",
             root=None):
    # deprecated: google drive fails
    if root is None:
        root = default_pretrained_metrics_dir
    os.makedirs(root, exist_ok=True)
    destination = os.path.join(root, fname)

    if os.path.exists(destination):
        return destination

    google_drive_prefix = 'https://drive.google.com/uc?export=download'
    session = requests.Session()

    response = session.get(google_drive_prefix, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(google_drive_prefix, params=params, stream=True)
    save_response_content(response, destination)
    return destination


def load_i3d_pretrained(device=torch.device('cpu'), channels=400):
    if channels == 400:
        filename = pretrained_i3d_400_name
    elif channels == 600:
        filename = pretrained_i3d_600_name
    else:
        raise ValueError(f"Only 400 and 600 channels are supported, got {channels}.")
    from .pytorch_i3d import InceptionI3d
    i3d = InceptionI3d(channels, in_channels=3).to(device)
    filepath = os.path.join(default_pretrained_metrics_dir, filename)
    if not os.path.exists(filepath):
        download_pretrained_weights(ckpt_name=filename,
                                    save_dir=default_pretrained_metrics_dir,
                                    exist_ok=False)
    i3d.load_state_dict(torch.load(filepath, map_location=device))
    i3d.eval()
    return i3d
