import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from pathlib import Path
import torchaudio
from torch.utils.data import random_split
import torch.nn.functional as F

sample_rate = 44100
data_length = 2**16


class DrumDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.root = Path(data_dir)
        self.files = []
        hats = self.root / "Hats"
        kicks = self.root / "Kicks"
        snares = self.root / "Snares"
        self.files.extend(hats.glob("*.wav"))
        self.files.extend(hats.glob("*.aif"))
        self.files.extend(kicks.glob("*.wav"))
        self.files.extend(kicks.glob("*.aif"))
        self.files.extend(snares.glob("*.wav"))
        self.files.extend(snares.glob("*.aif"))

        self.labels = [f.parent.name for f in self.files]
        self.labels = [
            0 if l == "Hats" else 1 if l == "Kicks" else 2 for l in self.labels
        ]

        shuffle_indices = torch.randperm(len(self.files))
        self.files = [self.files[i] for i in shuffle_indices]
        self.labels = [self.labels[i] for i in shuffle_indices]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data, sr = torchaudio.load(self.files[idx])
        data = torchaudio.functional.resample(data, sr, sample_rate)
        # Sum to mono
        if data.shape[0] > 1:
            data = torch.sum(data, dim=0, keepdim=True)
        # Pad or trim to 2**16
        if data.shape[1] < data_length:
            data = F.pad(data, (0, data_length - data.shape[1]))
        else:
            data = data[:, :data_length]
        return data, self.labels[idx]


class DrumDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def setup(self, stage=None):
        train_split = int(len(self.dataset) * 0.75)
        val_split = len(self.dataset) - train_split
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_split, val_split]
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )
