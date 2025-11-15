import os

# root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# default_exps_dir = os.path.abspath(os.path.join(root_dir, "experiments"))

# default_dataset_dir = os.path.abspath(os.path.join(root_dir, "datasets"))
# default_dataset_sevir_dir = os.path.abspath(os.path.join(default_dataset_dir, "sevir"))
# default_dataset_sevirlr_dir = os.path.abspath(os.path.join(default_dataset_dir, "sevirlr"))

# default_pretrained_dir = os.path.abspath(os.path.join(root_dir, "pretrained"))
# default_pretrained_metrics_dir = os.path.abspath(os.path.join(default_pretrained_dir, "metrics"))
# default_pretrained_vae_dir = os.path.abspath(os.path.join(default_pretrained_dir, "vae"))
# default_pretrained_earthformerunet_dir = os.path.abspath(os.path.join(default_pretrained_dir, "earthformerunet"))
# default_pretrained_alignment_dir = os.path.abspath(os.path.join(default_pretrained_dir, "alignment"))

#==============换成绝对路径=================
# ==============================
# 固定绝对路径版本配置
# ==============================

ROOT_DIR = "/data/25fall_nowcasting/ly/PreDiff-25fall"

# ---- Experiments ----
default_exps_dir = os.path.join(ROOT_DIR, "experiments1016")

ROOT_DIR_pub = "/data/25fall_nowcasting/25fall_aiclass/lesson_resource/data/prediff"
# ---- Datasets ----
default_dataset_dir = os.path.join(ROOT_DIR_pub, "datasets")
default_dataset_sevir_dir = os.path.join(default_dataset_dir, "sevir")
default_dataset_sevirlr_dir = os.path.join(default_dataset_dir, "sevirlr")

# ---- Pretrained Models ----
default_pretrained_dir = os.path.join(ROOT_DIR_pub, "pretrained")
default_pretrained_metrics_dir = os.path.join(default_pretrained_dir, "metrics")
default_pretrained_vae_dir = os.path.join(default_pretrained_dir, "vae")
default_pretrained_earthformerunet_dir = os.path.join(default_pretrained_dir, "earthformerunet")
default_pretrained_alignment_dir = os.path.join(default_pretrained_dir, "alignment")
