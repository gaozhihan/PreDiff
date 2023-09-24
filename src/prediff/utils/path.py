import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

default_exps_dir = os.path.abspath(os.path.join(root_dir, "experiments"))

default_dataset_dir = os.path.abspath(os.path.join(root_dir, "datasets"))
default_dataset_sevir_dir = os.path.abspath(os.path.join(default_dataset_dir, "sevir"))
default_dataset_sevirlr_dir = os.path.abspath(os.path.join(default_dataset_dir, "sevirlr"))

default_pretrained_dir = os.path.abspath(os.path.join(root_dir, "pretrained"))
default_pretrained_metrics_dir = os.path.abspath(os.path.join(default_pretrained_dir, "metrics"))
default_pretrained_vae_dir = os.path.abspath(os.path.join(default_pretrained_dir, "vae"))
default_pretrained_earthformerunet_dir = os.path.abspath(os.path.join(default_pretrained_dir, "earthformerunet"))
default_pretrained_alignment_dir = os.path.abspath(os.path.join(default_pretrained_dir, "alignment"))
