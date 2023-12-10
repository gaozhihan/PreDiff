import os
import argparse
from prediff.utils.download import (
    download_pretrained_weights,
    pretrained_sevirlr_vae_name,
    pretrained_sevirlr_earthformerunet_name,
    pretrained_sevirlr_alignment_name,
    pretrained_i3d_400_name,
    pretrained_i3d_600_name,
)
from prediff.utils.path import (
    default_pretrained_vae_dir,
    default_pretrained_earthformerunet_dir,
    default_pretrained_alignment_dir,
    default_pretrained_metrics_dir,
)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='all', type=str)
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.model in ["vae", "all"]:
        os.makedirs(default_pretrained_vae_dir, exist_ok=True)
        download_pretrained_weights(ckpt_name=pretrained_sevirlr_vae_name,
                                    save_dir=default_pretrained_vae_dir,
                                    exist_ok=False)
    if args.model in ["earthformerunet", "all"]:
        os.makedirs(default_pretrained_earthformerunet_dir, exist_ok=True)
        download_pretrained_weights(ckpt_name=pretrained_sevirlr_earthformerunet_name,
                                    save_dir=default_pretrained_earthformerunet_dir,
                                    exist_ok=False)
    if args.model in ["alignment", "all"]:
        os.makedirs(default_pretrained_alignment_dir, exist_ok=True)
        download_pretrained_weights(ckpt_name=pretrained_sevirlr_alignment_name,
                                    save_dir=default_pretrained_alignment_dir,
                                    exist_ok=False)
    if args.model in ["i3d400", "all"]:
        os.makedirs(default_pretrained_metrics_dir, exist_ok=True)
        download_pretrained_weights(ckpt_name=pretrained_i3d_400_name,
                                    save_dir=default_pretrained_metrics_dir,
                                    exist_ok=False)
    if args.model in ["i3d600", "all"]:
        os.makedirs(default_pretrained_metrics_dir, exist_ok=True)
        download_pretrained_weights(ckpt_name=pretrained_i3d_600_name,
                                    save_dir=default_pretrained_metrics_dir,
                                    exist_ok=False)
