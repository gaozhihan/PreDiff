import os
import warnings


pretrained_sevirlr_vae_name = "pretrained_sevirlr_vae_8x8x64_v1.pt"
pretrained_sevirlr_earthformerunet_name = "pretrained_sevirlr_earthformerunet_v1.pt"
pretrained_sevirlr_alignment_name = "pretrained_sevirlr_alignment_avg_x_cuboid_v1.pt"

file_id_dict = {
    pretrained_sevirlr_vae_name: "10OicEQuOPzSKDp5WYF3zDHsL-COywe98",
    pretrained_sevirlr_earthformerunet_name: "1cVB0Sm2V4OMTLxNNEXlb2__ONqSAjUDJ",
    pretrained_sevirlr_alignment_name: "1CzrzNVDVTyc8ivnWqEOiJnrN6wraY6fy",
}


def download_pretrained_weights(ckpt_name, save_dir=None, exist_ok=False):
    r"""
    Download pretrained weights from Google Drive.

    Parameters
    ----------
    ckpt_name:  str
    save_dir:   str
    exist_ok:   bool
    """
    if save_dir is None:
        from .path import default_pretrained_dir
        save_dir = default_pretrained_dir
    ckpt_path = os.path.join(save_dir, ckpt_name)
    if os.path.exists(ckpt_path) and not exist_ok:
        warnings.warn(f"Checkpoint file {ckpt_path} already exists!")
    else:
        os.makedirs(save_dir, exist_ok=True)
        file_id = file_id_dict[ckpt_name]
        os.system(f"wget --load-cookies /tmp/cookies.txt "
                  f"\"https://docs.google.com/uc?export=download&confirm="
                  f"$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "
                  f"'https://docs.google.com/uc?export=download&id={file_id}'"
                  f" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')"
                  f"&id={file_id}\" -O {ckpt_path} && rm -rf /tmp/cookies.txt")
