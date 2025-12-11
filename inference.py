"""
Utility to track valence, arousal, and dominance over time using
``audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim``.

Example usage:
    python inference.py --audio path/to/audio.wav --output predictions.csv

This script splits audio into chunks, runs the emotion model on each chunk,
and writes a CSV or JSON file with time-aligned scores.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import librosa
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoProcessor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
DEFAULT_SAMPLE_RATE = 16_000


class RegressionHead(nn.Module):
    """Regression head for arousal, dominance, valence."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    """Speech emotion regressor: arousal, dominance, valence."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        # input_values: (B, T)
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]  # (B, T, D)
        hidden_states = torch.mean(hidden_states, dim=1)  # (B, D)
        logits = self.classifier(hidden_states)  # (B, 3) [A, D, V]
        # Return (embeddings, logits)
        return hidden_states, logits


@dataclass
class SegmentResult:
    """Container for a single chunk prediction."""

    start_time: float
    end_time: float
    scores: dict


@dataclass
class InferenceConfig:
    """Configuration for model inference."""

    model_id: str = MODEL_ID
    sample_rate: int = DEFAULT_SAMPLE_RATE
    chunk_seconds: float = 3.0
    hop_seconds: float | None = None
    device: str = "cpu"

    def resolved_hop_seconds(self) -> float:
        return self.chunk_seconds if self.hop_seconds is None else self.hop_seconds


class EmotionTracker:
    """Predict valence, arousal, and dominance for audio chunks."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.processor = AutoProcessor.from_pretrained(config.model_id)
        self.model = EmotionModel.from_pretrained(config.model_id)
        self.model.to(config.device)
        self.model.eval()
        self.labels: List[str] = []
        self._load_labels()

    def _load_labels(self) -> None:
        id2label = getattr(self.model.config, "id2label", None)
        if not id2label:
            raise ValueError("Model config does not contain id2label mapping.")
        # Ensure ordering by index keys 0,1,2 → ["arousal", "dominance", "valence"]
        self.labels = [id2label[i] for i in range(len(id2label))]

    def predict_chunks(self, chunks: List[tuple[float, float, np.ndarray]]) -> List[SegmentResult]:
        """
        Predict emotions for pre-generated audio chunks.
        
        Args:
            chunks: List of (start_time, end_time, chunk_audio) tuples
            
        Returns:
            List of SegmentResult with emotion predictions
        """
        results: List[SegmentResult] = []
        for start, end, chunk in chunks:
            scores = self._predict_chunk(chunk)
            results.append(SegmentResult(start_time=start, end_time=end, scores=scores))
        return results

    def predict(self, audio: np.ndarray) -> tuple[List[SegmentResult], List[tuple[float, float, np.ndarray]]]:
        """
        Predict emotions for audio chunks (generates chunks internally).
        
        Args:
            audio: Audio array to process
            
        Returns:
            Tuple of (emotion_results, chunks) where:
            - emotion_results: List of SegmentResult with emotion predictions
            - chunks: List of (start_time, end_time, chunk_audio) tuples
        """
        chunks = list(
            generate_chunks(
                audio=audio,
                sample_rate=self.config.sample_rate,
                chunk_seconds=self.config.chunk_seconds,
                hop_seconds=self.config.resolved_hop_seconds(),
            )
        )
        results = self.predict_chunks(chunks)
        return results, chunks

    def _predict_chunk(self, chunk: np.ndarray) -> dict:
        proc_out = self.processor(
            chunk,
            sampling_rate=self.config.sample_rate,
            return_tensors="pt",
        )
        input_values = proc_out["input_values"].to(self.config.device)

        with torch.no_grad():
            _, logits = self.model(input_values)

        values = logits.squeeze(0).cpu().numpy().tolist()
        return {label: float(score) for label, score in zip(self.labels, values)}


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return audio


def generate_chunks(
    audio: Sequence[float],
    sample_rate: int,
    chunk_seconds: float,
    hop_seconds: float,
) -> Iterable[tuple[float, float, np.ndarray]]:
    """Yield padded audio chunks with start/end time metadata."""

    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if hop_seconds <= 0:
        raise ValueError("hop_seconds must be positive")

    chunk_samples = int(round(chunk_seconds * sample_rate))
    hop_samples = int(round(hop_seconds * sample_rate))
    total_samples = len(audio)

    start_sample = 0
    while start_sample < total_samples:
        end_sample = start_sample + chunk_samples
        chunk = np.asarray(audio[start_sample:end_sample])
        # Pad last chunk if needed
        if len(chunk) < chunk_samples:
            pad_width = chunk_samples - len(chunk)
            chunk = np.pad(chunk, (0, pad_width))
        start_time = start_sample / sample_rate
        end_time = min(end_sample, total_samples) / sample_rate
        yield start_time, end_time, chunk
        start_sample += hop_samples


def write_csv(path: Path, results: List[SegmentResult], labels: Sequence[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_time", "end_time", *labels])
        for item in results:
            writer.writerow(
                [
                    f"{item.start_time:.3f}",
                    f"{item.end_time:.3f}",
                    *[
                        f"{item.scores.get(label, float('nan')):.6f}"
                        for label in labels
                    ],
                ]
            )


def write_json(path: Path, results: List[SegmentResult]) -> None:
    payload = [
        {
            "start_time": item.start_time,
            "end_time": item.end_time,
            "scores": item.scores,
        }
        for item in results
    ]
    path.write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True, help="Path to audio file")
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID,
        help="Hugging Face model id to load",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=5.0,
        help="Chunk length in seconds",
    )
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=None,
        help="Stride between chunks in seconds (defaults to chunk length)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Target sampling rate for the model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device, e.g., 'cpu' or 'cuda'",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save predictions (defaults to <audio>.csv)",
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "json"],
        default="csv",
        help="Output file format",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = InferenceConfig(
        model_id=args.model_id,
        sample_rate=args.sample_rate,
        chunk_seconds=args.chunk_seconds,
        hop_seconds=args.hop_seconds,
        device=args.device,
    )

    audio = load_audio(args.audio, config.sample_rate)
    tracker = EmotionTracker(config)
    results, _ = tracker.predict(audio)  # Unpack results and chunks (chunks unused in CLI)

    output_path = args.output or args.audio.with_suffix(f".{args.output_format}")

    if args.output_format == "csv":
        write_csv(output_path, results, tracker.labels)
    else:
        write_json(output_path, results)

    print(f"Wrote {len(results)} segments to {output_path}")


if __name__ == "__main__":
    main()
