import pytorch_lightning as pl
from dataset import DrumDataModule, DrumDataset
from model import Classifier, Cnn14


def main():
    print("Hello World")
    network = Cnn14(n_fft=4096, num_classes=3, sample_rate=44100)
    model = Classifier(1e-4, 1e-3, 44100, network)
    dataset = DrumDataset("sss_free")
    batch_size = 32
    data = DrumDataModule(dataset, batch_size)

    trainer = pl.Trainer(max_epochs=10, accelerator="cpu")
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
