#!/bin/bash
# Helper script to test the Flask API with an audio file

# Check if file path was provided
if [ -z "$1" ]; then
    echo "Usage: ./test_upload.sh <path/to/audio.wav>"
    echo ""
    echo "Example:"
    echo "  ./test_upload.sh /path/to/my_audio.wav"
    echo "  ./test_upload.sh ./audio.wav"
    echo "  ./test_upload.sh audio.wav"
    exit 1
fi

AUDIO_FILE="$1"

# Check if file exists
if [ ! -f "$AUDIO_FILE" ]; then
    echo "Error: File '$AUDIO_FILE' does not exist"
    echo ""
    echo "Please provide a valid path to an audio file."
    echo "Current directory: $(pwd)"
    exit 1
fi

# Get absolute path
AUDIO_FILE=$(realpath "$AUDIO_FILE" 2>/dev/null || echo "$AUDIO_FILE")

# Check if server is running
if ! curl -s http://localhost:5001/health > /dev/null 2>&1; then
    echo "Error: Server is not running on port 5001"
    echo ""
    echo "Please start the server first:"
    echo "  python app.py"
    exit 1
fi

echo "Uploading: $AUDIO_FILE"
echo "To server: http://localhost:5001/process_audio"
echo ""

# Send the request
curl -X POST http://localhost:5001/process_audio \
  -F "audio=@$AUDIO_FILE" \
  -o response.json

# Check if request was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Request successful!"
    echo "Response saved to response.json"
    echo ""
    echo "Response content:"
    cat response.json | python3 -m json.tool 2>/dev/null || cat response.json
    echo ""
    echo "Output files are in: ./outputs/"
else
    echo "✗ Error: Request failed"
    exit 1
fi

