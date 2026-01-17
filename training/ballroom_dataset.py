import pathlib
from typing import List, Tuple

import librosa
import numpy as np


class BallroomDataset:
    def __init__(self, root: str, sample_rate: int = 16000, hop_length: int = 160, n_mels: int = 64):
        self.root = pathlib.Path(root)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.items = self._scan()

    def _scan(self) -> List[Tuple[pathlib.Path, pathlib.Path]]:
        audio_dir = self.root / "audio"
        anno_dir = self.root / "annotations"
        items = []
        for audio_path in sorted(audio_dir.glob("*.wav")):
            stem = audio_path.stem
            anno_path = anno_dir / f"{stem}.beats"
            if anno_path.exists():
                items.append((audio_path, anno_path))
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        audio_path, anno_path = self.items[idx]
        audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        beat_times = np.loadtxt(anno_path, dtype=np.float32)
        if beat_times.ndim == 0:
            beat_times = np.array([beat_times], dtype=np.float32)

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=512,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
        )
        log_mel = np.log1p(mel).astype(np.float32)
        log_mel = log_mel.T  # (frames, mel)

        labels = np.zeros(log_mel.shape[0], dtype=np.float32)
        frame_times = np.arange(log_mel.shape[0]) * (self.hop_length / self.sample_rate)
        tolerance = 0.05
        for bt in beat_times:
            idx = np.argmin(np.abs(frame_times - bt))
            if abs(frame_times[idx] - bt) <= tolerance:
                labels[idx] = 1.0

        return log_mel, labels
