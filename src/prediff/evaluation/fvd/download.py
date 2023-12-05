import requests
from tqdm import tqdm
import os
import torch

from ...utils.path import default_pretrained_metrics_dir

_I3D_PRETRAINED_ID = '1mQK8KD8G6UWRa5t87SRMm5PVXtlpneJT'


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


def download(id, fname, root=None):
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
    assert channels in [400, 600], f"Only 400 and 600 channels are supported, got {channels}."
    from .pytorch_i3d import InceptionI3d
    i3d = InceptionI3d(channels, in_channels=3).to(device)
    filepath = download(_I3D_PRETRAINED_ID, f"i3d_pretrained_{channels}.pt")  # google drive link not work now
    # filepath = os.path.join(default_pretrained_metrics_dir, f"i3d_pretrained_{channels}.pt")
    if not os.path.exists:
        raise FileNotFoundError(f"Pretrained I3D model not found at {filepath}. Run `./scripts/evaluation/convert_tf_pretrained.py` to generate it.")
    i3d.load_state_dict(torch.load(filepath, map_location=device))
    i3d.eval()
    return i3d
