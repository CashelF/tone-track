# tone-track

Track valence, arousal, and dominance over time using the
`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` model.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run inference

Pass any mono/stereo audio file; it will be resampled to 16 kHz automatically.

```bash
python inference.py \
  --audio path/to/audio.wav \
  --chunk-seconds 5.0 \
  --hop-seconds 2.5 \
  --output predictions.csv
```

* `--output-format` can be `csv` (default) or `json`.
* `--hop-seconds` defaults to the chunk size for non-overlapping windows.
* Use `--device cuda` if you have a GPU available.

The output CSV has `start_time`, `end_time`, and model scores (valence,
arousal, dominance) for each chunk.
