from shutil import copyfile
from prediff.datasets.sevir.sevir_dataloader import (
    SEVIRDataLoader, SEVIR_LR_DATA_DIR,
    SEVIR_CATALOG, SEVIR_LR_CATALOG,)


if __name__ == '__main__':
    downsample_dict = {'vil': (2, 3, 3)}
    batch_size = 32

    copyfile(SEVIR_CATALOG, SEVIR_LR_CATALOG)
    sevir_dataloader = SEVIRDataLoader(data_types=['vil', ], sample_mode='sequent', batch_size=batch_size)
    sevir_dataloader.save_downsampled_dataset(
        save_dir=SEVIR_LR_DATA_DIR,
        downsample_dict=downsample_dict,
        verbose=True)
