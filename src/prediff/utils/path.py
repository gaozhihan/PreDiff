import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

default_exps_dir = os.path.abspath(os.path.join(root_dir, "experiments"))

default_dataset_dir = os.path.abspath(os.path.join(root_dir, "datasets"))
# SEVIR
default_dataset_sevir_dir = os.path.abspath(os.path.join(default_dataset_dir, "sevir"))
default_dataset_sevirlr_dir = os.path.abspath(os.path.join(default_dataset_dir, "sevirlr"))
# HKO
default_dataset_hko_dir = os.path.abspath(os.path.join(default_dataset_dir, "hko"))
# default_pd_path = os.path.join(default_dataset_hko_dir, "hko7_all.pkl")
default_pd_path = os.path.join(default_dataset_hko_dir, "hko7_all_avg_int.pkl")  # one more column "avg_int" than hko7_all.pkl
default_exclude_mask_path = os.path.join(default_dataset_hko_dir, 'mask_dat.npz')
default_png_file_dir = os.path.join(default_dataset_hko_dir, 'radarPNG')
default_mask_file_dir = os.path.join(default_dataset_hko_dir, 'radarPNG_mask')
# pretrained
default_pretrained_dir = os.path.abspath(os.path.join(root_dir, "pretrained"))
default_pretrained_metrics_dir = os.path.abspath(os.path.join(default_pretrained_dir, "metrics"))
default_pretrained_vae_dir = os.path.abspath(os.path.join(default_pretrained_dir, "vae"))
default_pretrained_earthformerunet_dir = os.path.abspath(os.path.join(default_pretrained_dir, "earthformerunet"))
default_pretrained_alignment_dir = os.path.abspath(os.path.join(default_pretrained_dir, "alignment"))
