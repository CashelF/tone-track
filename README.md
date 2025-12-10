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

### Process audio and analyze with Gemini (all-in-one):

**Process audio and immediately get Gemini analysis:**

```bash
curl -X POST http://localhost:5001/process_and_analyze_audio \
  -F "audio=@konrad_pt2.wav" \
  -F "api_key=YOUR_GEMINI_API_KEY" \
  -o response.json
```

**With custom settings:**

```bash
curl -X POST http://localhost:5001/process_and_analyze_audio \
  -F "audio=@path/to/your/audio.wav" \
  -F "chunk_seconds=5.0" \
  -F "hop_seconds=2.5" \
  -F "device=cpu" \
  -F "model=gemini-2.5-flash" \
  -F "api_key=YOUR_GEMINI_API_KEY" \
  -o response.json
```

**Or set GEMINI_API_KEY environment variable:**

```bash
export GEMINI_API_KEY=your_api_key_here
curl -X POST http://localhost:5001/process_and_analyze_audio \
  -F "audio=@konrad_pt2.wav" \
  -o response.json
```

This endpoint:
1. Processes the audio file (generates chunks and words CSV files)
2. Immediately analyzes the chunks CSV with Gemini
3. Returns both the CSV file paths AND the LLM analysis in one response
4. Saves the analysis to a text file in the outputs directory

**Response includes:**
- `chunks_csv`: Path to chunks CSV file
- `words_csv`: Path to words CSV file
- `chunk_count`: Number of emotion chunks
- `word_count`: Number of aligned words
- `analysis`: Full Gemini sentiment analysis text
- `analysis_txt`: Path to saved analysis text file

### Analyze existing chunks CSV with Gemini AI:

**Analyze emotion chunks for sentiment analysis (focuses on emotional dips):**

```bash
curl -X POST http://localhost:5001/analyze_chunks \
  -F "chunks_csv=outputs/konrad_pt2_chunks.csv" \
  -F "api_key=YOUR_GEMINI_API_KEY" \
  -o analysis.json
```

**Or set GEMINI_API_KEY environment variable:**

```bash
export GEMINI_API_KEY=your_api_key_here
curl -X POST http://localhost:5001/analyze_chunks \
  -F "chunks_csv=outputs/konrad_pt2_chunks.csv" \
  -o analysis.json
```

**Optional parameters:**
- `model`: Gemini model name (default: `gemini-2.5-flash`)
- `api_key`: Gemini API key (if not set, uses `GEMINI_API_KEY` env var)

The analysis provides:
- Overall sentiment trajectory
- Detailed analysis of emotional dips (valence, arousal, dominance)
- Emotional recovery patterns
- Key insights and recommendations

**Get your Gemini API key:** https://makersuite.google.com/app/apikey
