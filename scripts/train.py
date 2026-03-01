import argparse
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader

from ballroom_dataset import BallroomDataset
from model import BeatNet


def collate(batch):
    max_frames = max(item[0].shape[0] for item in batch)
    mel_bins = batch[0][0].shape[1]
    features = np.zeros((len(batch), max_frames, mel_bins), dtype=np.float32)
    labels = np.zeros((len(batch), max_frames), dtype=np.float32)
    mask = np.zeros((len(batch), max_frames), dtype=np.float32)

    for i, (feat, lab) in enumerate(batch):
        frames = feat.shape[0]
        features[i, :frames, :] = feat
        labels[i, :frames] = lab
        mask[i, :frames] = 1.0

    return torch.from_numpy(features), torch.from_numpy(labels), torch.from_numpy(mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to ballroom dataset root")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", type=str, default="training/outputs")
    args = parser.parse_args()

    dataset = BallroomDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    model = BeatNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCELoss(reduction="none")

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for features, labels, mask in loader:
            output = model(features)
            preds = output["beat"]
            loss = bce(preds, labels)
            loss = (loss * mask).sum() / mask.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}")

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "beat_model.pt"
    torch.save(model.state_dict(), checkpoint)
    print(f"Saved {checkpoint}")


if __name__ == "__main__":
    main()
