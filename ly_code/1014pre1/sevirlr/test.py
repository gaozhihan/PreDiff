from prediff.datasets.sevir.sevir_dataloader import NPYSEVIRDataLoader
from prediff.datasets.sevir.sevir_torch_wrap import SEVIRTorchDataset
from torch.utils.data import DataLoader

np_dir = "/home/user01/25fall_aiclass/lesson_resource/data/prediff/datasets/sevirlr/data_npy"
loader = NPYSEVIRDataLoader(npy_dir=np_dir, seq_len=25, raw_seq_len=25, stride=1)
dataset = SEVIRTorchDataset(seq_len=25, raw_seq_len=25, sevir_dataloader=loader)
dl = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
for batch in dl:
    # batch shape (B, T, H, W) 或根据 layout 调整
    print(batch.shape)
    break

