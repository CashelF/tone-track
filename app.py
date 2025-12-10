"""
Flask backend for emotion tracking with transcription and word alignment.

Example usage:
    python app.py
    curl -X POST http://localhost:5001/process_audio -F "audio=@audio.wav" -o response.json
"""

import csv
import os
from pathlib import Path
from typing import List

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from asr import get_transcriber
from inference import (
    DEFAULT_SAMPLE_RATE,
    InferenceConfig,
    load_audio,
    generate_chunks,
    SegmentResult,
    EmotionTracker,
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

OUTPUT_DIR = Path(__file__).parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(OUTPUT_DIR)


def write_chunk_csv(emotion_results: List[SegmentResult], chunk_transcripts: List[dict], labels: List[str], output_path: Path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['chunk_index', 'start_time', 'end_time', 'transcript', *labels])
        
        for idx, (emotion_result, chunk_transcript) in enumerate(zip(emotion_results, chunk_transcripts)):
            writer.writerow([
                idx,
                f"{emotion_result.start_time:.3f}",
                f"{emotion_result.end_time:.3f}",
                chunk_transcript.get('transcript', ''),
                *[f"{emotion_result.scores.get(label, float('nan')):.6f}" for label in labels]
            ])


def align_words_with_chunks(transcript_result: dict, emotion_results: List[SegmentResult], audio_duration: float) -> List[dict]:
    words = []
    
    if 'chunks' in transcript_result:
        for chunk in transcript_result['chunks']:
            word = chunk.get('text', '').strip()
            start = chunk.get('timestamp', [0, 0])[0]
            end = chunk.get('timestamp', [0, 0])[1]
            
            if word:
                chunk_idx = None
                for i, emotion_chunk in enumerate(emotion_results):
                    if not (end < emotion_chunk.start_time or start > emotion_chunk.end_time):
                        chunk_idx = i
                        break
                
                words.append({
                    'word': word,
                    'start_time': start,
                    'end_time': end,
                    'chunk_idx': chunk_idx if chunk_idx is not None else -1
                })
    else:
        text = transcript_result.get('text', '')
        word_list = text.split()
        
        if word_list:
            words_per_second = len(word_list) / audio_duration if audio_duration > 0 else 0
            
            for i, word in enumerate(word_list):
                start = i / words_per_second if words_per_second > 0 else 0
                end = (i + 1) / words_per_second if words_per_second > 0 else audio_duration
                
                chunk_idx = None
                for j, emotion_chunk in enumerate(emotion_results):
                    if not (end < emotion_chunk.start_time or start > emotion_chunk.end_time):
                        chunk_idx = j
                        break
                
                words.append({
                    'word': word,
                    'start_time': start,
                    'end_time': end,
                    'chunk_idx': chunk_idx if chunk_idx is not None else -1
                })
    
    return words


def write_word_csv(words: List[dict], output_path: Path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'chunk_index', 'start_time', 'end_time'])
        for word_data in words:
            chunk_idx = word_data.get('chunk_idx', -1)
            if chunk_idx >= 0:
                writer.writerow([
                    word_data['word'],
                    chunk_idx,
                    f"{word_data['start_time']:.3f}",
                    f"{word_data['end_time']:.3f}"
                ])


@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    file.save(input_path)
    
    try:
        input_path_obj = Path(input_path)
        
        chunk_seconds = float(request.form.get('chunk_seconds', 5.0))
        hop_seconds = request.form.get('hop_seconds')
        hop_seconds = float(hop_seconds) if hop_seconds else None
        device = request.form.get('device', 'cpu')
        
        emotion_config = InferenceConfig(
            model_id="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
            sample_rate=DEFAULT_SAMPLE_RATE,
            chunk_seconds=chunk_seconds,
            hop_seconds=hop_seconds,
            device=device
        )
        
        audio = load_audio(input_path_obj, emotion_config.sample_rate)
        audio_duration = len(audio) / emotion_config.sample_rate
        
        # Generate chunks once, then process with both emotion tracker and transcriber
        chunks = list(
            generate_chunks(
                audio=audio,
                sample_rate=emotion_config.sample_rate,
                chunk_seconds=emotion_config.chunk_seconds,
                hop_seconds=emotion_config.resolved_hop_seconds(),
            )
        )
        
        tracker = EmotionTracker(emotion_config)
        emotion_results = tracker.predict_chunks(chunks)
        
        base_path = input_path_obj.stem
        output_dir = Path(app.config['UPLOAD_FOLDER'])
        
        # Initialize ASR transcriber
        transcriber = get_transcriber(device=device)
        chunk_transcripts = transcriber.transcribe_chunks(audio, chunks, emotion_config.sample_rate)
        
        chunks_csv_path = output_dir / f"{base_path}_chunks.csv"
        write_chunk_csv(emotion_results, chunk_transcripts, tracker.labels, chunks_csv_path)
        
        transcript_result = transcriber.transcribe_audio(input_path, emotion_config.sample_rate)
        aligned_words = align_words_with_chunks(transcript_result, emotion_results, audio_duration)
        
        words_csv_path = output_dir / f"{base_path}_words.csv"
        write_word_csv(aligned_words, words_csv_path)
        
        return jsonify({
            'status': 'success',
            'chunks_csv': str(chunks_csv_path.absolute()),
            'words_csv': str(words_csv_path.absolute()),
            'chunk_count': len(emotion_results),
            'word_count': len([w for w in aligned_words if w.get('chunk_idx', -1) >= 0])
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
