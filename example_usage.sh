#!/bin/bash
# Example usage of the Flask backend API
# Uses inference.py as the base for emotion tracking

# Start the Flask server (run this in one terminal)
# python app.py
# Default port is 5001 (to avoid AirPlay conflict on macOS)
# To use a different port: PORT=8080 python app.py

# Then in another terminal, use curl to send a WAV file:

# Option 1: Use the helper script (easiest)
# ./test_upload.sh /path/to/your/audio.wav

# Option 2: Use curl directly
# IMPORTANT: Replace 'audio.wav' with your actual file path!

# Basic usage (uses inference.py defaults: 5.0s chunks, non-overlapping):
curl -X POST http://localhost:5001/process_audio \
  -F "audio=@audio.wav" \
  -o response.json

# If you get "Failed to open/read local data" error:
# - Make sure the file path is correct
# - Use absolute path: -F "audio=@/full/path/to/audio.wav"
# - Or relative path: -F "audio=@./audio.wav"

# With custom chunk settings (e.g., overlapping chunks):
curl -X POST http://localhost:5001/process_audio \
  -F "audio=@path/to/your/audio.wav" \
  -F "chunk_seconds=5.0" \
  -F "hop_seconds=2.5" \
  -F "device=cpu" \
  -o response.json

# The response will contain paths to the generated CSV files:
# - emotion_csv: Same format as inference.py output
# - transcript_chunks_csv: Transcripts for each emotion chunk
# - word_alignment_csv: Words aligned with emotion chunks

