from model import Classifier, Cnn14
import sys
import torch
import torchaudio
import torch.nn.functional as F
import os
from pathlib import Path

sample_rate = 44100
data_length = 2**16


def main():
    root = Path(sys.argv[1])
    files = []
    if not os.path.isdir(sys.argv[1]):
        files = [root]
    else:
        files = root.glob("*.wav")
    files = list(files)
    dataset = []
    for file in files:
        print(file)
        audio, sr = torchaudio.load(file)
        data = torchaudio.functional.resample(audio, sr, sample_rate)
        # Sum to mono
        if data.shape[0] > 1:
            data = torch.sum(data, dim=0, keepdim=True)
        # Pad or trim to 2**16
        if data.shape[1] < data_length:
            data = F.pad(data, (0, data_length - data.shape[1]))
        else:
            data = data[:, :data_length]
        dataset.append(data)
    dataset = torch.stack(dataset)

    network = Cnn14(n_fft=4096, num_classes=3, sample_rate=44100)
    model = Classifier.load_from_checkpoint(
        "best.ckpt",
        network=network,
        lr=1e-4,
        lr_weight_decay=1e-3,
        sample_rate=sample_rate,
    )
    model.eval()
    with torch.no_grad():
        predictions = model(dataset)
    predictions = torch.argmax(predictions, dim=1)
    output = [
        "Hat" if prediction == 0 else "Kick" if prediction == 1 else "Snare"
        for prediction in predictions
    ]
    import pdb

    pdb.set_trace()
    for file, prediction in zip(files, output):
        print(f"{file}: {prediction}")


if __name__ == "__main__":
    main()
