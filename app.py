"""
Flask backend for emotion tracking with transcription and word alignment.

Example usage:
    python app.py
    curl -X POST http://localhost:5001/process_audio -F "audio=@audio.wav" -o response.json
"""

import csv
import logging
import os
from pathlib import Path
from typing import List

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from asr import get_transcriber
from inference import (
    DEFAULT_SAMPLE_RATE,
    InferenceConfig,
    load_audio,
    generate_chunks,
    SegmentResult,
    EmotionTracker,
)
from llm_analysis import analyze_chunks_with_gemini

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# Enable CORS for all routes
CORS(app)

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


def read_csv_to_dict(csv_path: Path) -> List[dict]:
    """
    Read CSV file and return as list of dictionaries.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        List of dictionaries with CSV data
    """
    data = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


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
        
        # Read CSV data for frontend
        chunks_csv_data = read_csv_to_dict(chunks_csv_path)
        words_csv_data = read_csv_to_dict(words_csv_path)
        
        return jsonify({
            'status': 'success',
            'chunks_csv_data': chunks_csv_data,
            'words_csv_data': words_csv_data,
            'file_paths': {
                'chunks_csv': str(chunks_csv_path.absolute()),
                'words_csv': str(words_csv_path.absolute())
            },
            'metadata': {
                'chunk_count': len(emotion_results),
                'word_count': len([w for w in aligned_words if w.get('chunk_idx', -1) >= 0])
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/process_and_analyze_audio', methods=['POST'])
def process_and_analyze_audio():
    """
    Process audio file and immediately analyze with Gemini API.
    
    This endpoint combines /process_audio and /analyze_chunks functionality.
    It processes the audio, generates CSV files, then immediately calls Gemini
    to analyze the chunks and returns both CSV data and LLM analysis in the response.
    
    Accepts either:
    - File upload: 'audio' file in form data
    - File path: 'audio_path' parameter pointing to existing WAV file
    
    Expected form data:
    - audio (optional): Audio file to upload
    - audio_path (optional): Path to existing WAV file (relative to outputs/ or absolute)
    - chunk_seconds (optional): Chunk length in seconds (default: 5.0)
    - hop_seconds (optional): Stride between chunks (default: same as chunk_seconds)
    - device (optional): Device to use 'cpu' or 'cuda' (default: 'cpu')
    - api_key (optional): Gemini API key (if not provided, uses GEMINI_API_KEY env var)
    - model (optional): Gemini model name (default: gemini-2.5-flash)
    
    Returns:
    - chunks_csv_data: Array of chunk data objects
    - words_csv_data: Array of word data objects
    - analysis: LLM sentiment analysis text
    - file_paths: Paths to saved files (for reference)
    """
    logger.info("=== /process_and_analyze_audio endpoint called ===")
    logger.debug(f"Request form data: {dict(request.form)}")
    logger.debug(f"Request files: {dict(request.files)}")
    
    # Handle file path or file upload
    input_path = None
    
    # Check for file path first (for frontend convenience)
    audio_path = request.form.get('audio_path')
    if audio_path:
        logger.info(f"Using audio_path parameter: {audio_path}")
        input_path_obj = Path(audio_path)
        
        # Handle relative paths
        if not input_path_obj.is_absolute():
            # Try relative to outputs directory
            path_str = str(input_path_obj).replace('outputs/', '').lstrip('/')
            input_path_obj = OUTPUT_DIR / path_str
            
            # If not found, try relative to project root
            if not input_path_obj.exists():
                input_path_obj = Path(__file__).parent / audio_path
        
        if not input_path_obj.exists():
            logger.error(f"Audio file not found: {input_path_obj}")
            return jsonify({'error': f'Audio file not found: {input_path_obj}'}), 404
        
        input_path = str(input_path_obj)
        logger.info(f"Resolved audio path: {input_path}")
    
    # Fall back to file upload
    elif 'audio' in request.files:
        file = request.files['audio']
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        file.save(input_path)
        logger.info(f"Saved uploaded audio file to: {input_path}")
    else:
        logger.error("No audio file or audio_path provided")
        return jsonify({'error': 'Either provide audio file upload or audio_path parameter'}), 400
    
    try:
        input_path_obj = Path(input_path)
        
        chunk_seconds = float(request.form.get('chunk_seconds', 5.0))
        hop_seconds = request.form.get('hop_seconds')
        hop_seconds = float(hop_seconds) if hop_seconds else None
        device = request.form.get('device', 'cpu')
        
        logger.info(f"Processing audio with chunk_seconds={chunk_seconds}, hop_seconds={hop_seconds}, device={device}")
        
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
        logger.info(f"Created chunks CSV: {chunks_csv_path}")
        
        transcript_result = transcriber.transcribe_audio(input_path, emotion_config.sample_rate)
        aligned_words = align_words_with_chunks(transcript_result, emotion_results, audio_duration)
        
        words_csv_path = output_dir / f"{base_path}_words.csv"
        write_word_csv(aligned_words, words_csv_path)
        logger.info(f"Created words CSV: {words_csv_path}")
        
        # Now analyze with Gemini
        logger.info("Starting Gemini analysis...")
        api_key = request.form.get('api_key') or None
        model_name = request.form.get('model', 'gemini-2.5-flash')
        
        try:
            analysis = analyze_chunks_with_gemini(
                csv_path=chunks_csv_path,
                api_key=api_key,
                model_name=model_name
            )
            logger.info(f"Gemini analysis completed, length: {len(analysis)} characters")
            
            # Save analysis to text file
            csv_stem = chunks_csv_path.stem
            if csv_stem.endswith('_chunks'):
                base_name = csv_stem[:-7]
            else:
                base_name = csv_stem
            
            analysis_txt_path = OUTPUT_DIR / f"{base_name}_analysis.txt"
            analysis_txt_path.write_text(analysis, encoding='utf-8')
            logger.info(f"Saved analysis to: {analysis_txt_path}")
            
            # Read CSV data for frontend
            chunks_csv_data = read_csv_to_dict(chunks_csv_path)
            words_csv_data = read_csv_to_dict(words_csv_path)
            
            return jsonify({
                'status': 'success',
                'chunks_csv_data': chunks_csv_data,
                'words_csv_data': words_csv_data,
                'analysis': analysis,
                'file_paths': {
                    'chunks_csv': str(chunks_csv_path.absolute()),
                    'words_csv': str(words_csv_path.absolute()),
                    'analysis_txt': str(analysis_txt_path.absolute())
                },
                'metadata': {
                    'chunk_count': len(emotion_results),
                    'word_count': len([w for w in aligned_words if w.get('chunk_idx', -1) >= 0])
                }
            }), 200
            
        except ImportError as e:
            logger.error(f"ImportError in Gemini analysis: {str(e)}")
            # Read CSV data for frontend
            chunks_csv_data = read_csv_to_dict(chunks_csv_path)
            words_csv_data = read_csv_to_dict(words_csv_path)
            # Return CSV results even if Gemini fails
            return jsonify({
                'status': 'partial_success',
                'chunks_csv_data': chunks_csv_data,
                'words_csv_data': words_csv_data,
                'analysis': None,
                'file_paths': {
                    'chunks_csv': str(chunks_csv_path.absolute()),
                    'words_csv': str(words_csv_path.absolute())
                },
                'metadata': {
                    'chunk_count': len(emotion_results),
                    'word_count': len([w for w in aligned_words if w.get('chunk_idx', -1) >= 0])
                },
                'error': 'Gemini analysis failed',
                'gemini_error': str(e),
                'hint': 'Install google-generativeai: pip install google-generativeai'
            }), 200
        except Exception as e:
            logger.error(f"Error in Gemini analysis: {str(e)}", exc_info=True)
            # Read CSV data for frontend
            chunks_csv_data = read_csv_to_dict(chunks_csv_path)
            words_csv_data = read_csv_to_dict(words_csv_path)
            # Return CSV results even if Gemini fails
            return jsonify({
                'status': 'partial_success',
                'chunks_csv_data': chunks_csv_data,
                'words_csv_data': words_csv_data,
                'analysis': None,
                'file_paths': {
                    'chunks_csv': str(chunks_csv_path.absolute()),
                    'words_csv': str(words_csv_path.absolute())
                },
                'metadata': {
                    'chunk_count': len(emotion_results),
                    'word_count': len([w for w in aligned_words if w.get('chunk_idx', -1) >= 0])
                },
                'error': 'Gemini analysis failed',
                'gemini_error': str(e)
            }), 200
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/analyze_chunks', methods=['POST'])
def analyze_chunks():
    """
    Analyze chunks CSV using Gemini API for sentiment analysis.
    
    Expected form data:
    - chunks_csv: Path to the chunks CSV file (relative to outputs/ or absolute path)
    - api_key (optional): Gemini API key (if not provided, uses GEMINI_API_KEY env var)
    - model (optional): Gemini model name (default: gemini-2.5-flash)
    """
    logger.info("=== /analyze_chunks endpoint called ===")
    logger.debug(f"Request method: {request.method}")
    logger.debug(f"Request form data: {dict(request.form)}")
    logger.debug(f"Request files: {dict(request.files)}")
    
    try:
        chunks_csv_path = request.form.get('chunks_csv')
        logger.debug(f"Received chunks_csv_path: {chunks_csv_path}")
        
        if not chunks_csv_path:
            logger.error("Missing chunks_csv parameter")
            return jsonify({'error': 'chunks_csv parameter is required'}), 400
        
        # Handle both relative and absolute paths
        csv_path = Path(chunks_csv_path)
        logger.debug(f"Initial csv_path: {csv_path}")
        logger.debug(f"Is absolute: {csv_path.is_absolute()}")
        logger.debug(f"OUTPUT_DIR: {OUTPUT_DIR}")
        
        if csv_path.is_absolute():
            # Use absolute path as-is
            logger.debug("Using absolute path as-is")
            pass
        else:
            # Handle relative paths
            # Strip "outputs/" prefix if present to avoid doubling
            path_str = str(csv_path).replace('outputs/', '').lstrip('/')
            logger.debug(f"Stripped path_str: {path_str}")
            
            # Try relative to OUTPUT_DIR
            csv_path = OUTPUT_DIR / path_str
            logger.debug(f"Final csv_path: {csv_path}")
        
        logger.info(f"Looking for CSV file at: {csv_path}")
        logger.debug(f"File exists: {csv_path.exists()}")
        
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            logger.debug(f"OUTPUT_DIR contents: {list(OUTPUT_DIR.iterdir()) if OUTPUT_DIR.exists() else 'OUTPUT_DIR does not exist'}")
            return jsonify({
                'error': f'CSV file not found: {csv_path}',
                'debug': {
                    'requested_path': chunks_csv_path,
                    'resolved_path': str(csv_path),
                    'output_dir': str(OUTPUT_DIR),
                    'output_dir_exists': OUTPUT_DIR.exists(),
                    'output_dir_contents': [str(p.name) for p in OUTPUT_DIR.iterdir()] if OUTPUT_DIR.exists() else []
                }
            }), 404
        
        # Get optional parameters
        api_key = request.form.get('api_key') or None
        model_name = request.form.get('model', 'gemini-2.5-flash')
        
        logger.info(f"API key provided: {bool(api_key)}")
        logger.info(f"Using model: {model_name}")
        
        if not api_key:
            env_api_key = os.environ.get('GEMINI_API_KEY')
            logger.debug(f"GEMINI_API_KEY env var set: {bool(env_api_key)}")
        
        # Analyze with Gemini
        logger.info("Calling analyze_chunks_with_gemini...")
        try:
            analysis = analyze_chunks_with_gemini(
                csv_path=csv_path,
                api_key=api_key,
                model_name=model_name
            )
            logger.info(f"Analysis received, length: {len(analysis)} characters")
            logger.debug(f"Analysis preview (first 200 chars): {analysis[:200]}...")
        except Exception as e:
            logger.error(f"Error in analyze_chunks_with_gemini: {str(e)}", exc_info=True)
            raise
        
        # Save analysis to text file in outputs directory
        # Extract base name from CSV (e.g., "konrad_pt2_chunks.csv" -> "konrad_pt2_chunks")
        csv_stem = csv_path.stem
        logger.debug(f"CSV stem: {csv_stem}")
        
        # Remove "_chunks" suffix if present to get base name
        if csv_stem.endswith('_chunks'):
            base_name = csv_stem[:-7]  # Remove "_chunks"
        else:
            base_name = csv_stem
        
        logger.debug(f"Base name: {base_name}")
        
        analysis_txt_path = OUTPUT_DIR / f"{base_name}_analysis.txt"
        logger.info(f"Saving analysis to: {analysis_txt_path}")
        
        try:
            analysis_txt_path.write_text(analysis, encoding='utf-8')
            logger.info(f"Successfully saved analysis to {analysis_txt_path}")
        except Exception as e:
            logger.error(f"Error saving analysis file: {str(e)}", exc_info=True)
            raise
        
        response_data = {
            'status': 'success',
            'analysis': analysis,
            'csv_path': str(csv_path),
            'analysis_txt': str(analysis_txt_path.absolute())
        }
        
        logger.info("=== /analyze_chunks endpoint completed successfully ===")
        return jsonify(response_data), 200
        
    except ImportError as e:
        logger.error(f"ImportError: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'hint': 'Install google-generativeai: pip install google-generativeai'
        }), 500
    except ValueError as e:
        logger.error(f"ValueError: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'error_type': type(e).__name__
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
