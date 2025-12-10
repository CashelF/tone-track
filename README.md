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

### Start Flask server:

```bash
python app.py
```

The server will start on `http://localhost:5001` by default (to avoid AirPlay conflict on macOS).

To use a different port:

```bash
PORT=8080 python app.py
```

### Process an audio file:

**Basic usage (uses inference.py defaults: 5.0s chunks, non-overlapping):**

```bash
curl -X POST http://localhost:5001/process_audio \
  -F "audio=@konrad_pt2.wav" \
  -o response.json
```

**With custom chunk settings:**

```bash
curl -X POST http://localhost:5001/process_audio \
  -F "audio=@path/to/your/audio.wav" \
  -F "chunk_seconds=5.0" \
  -F "hop_seconds=2.5" \
  -F "device=cpu" \
  -o response.json
```

### Output Files:

1. **Chunks CSV** (`*_chunks.csv`): Emotion scores and transcripts for each chunk
2. **Words CSV** (`*_words.csv`): Word-level alignment with emotion chunks
