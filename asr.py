"""
Automatic Speech Recognition (ASR) module using Whisper.

Provides transcription functionality for full audio files and audio chunks.
"""

from typing import List, Tuple

import librosa
import numpy as np
import torch
from transformers import pipeline


class ASRTranscriber:
    """Handles ASR transcription using Whisper model."""
    
    def __init__(self, model_id: str = "openai/whisper-base", device: str | None = None):
        """
        Initialize the ASR transcriber.
        
        Args:
            model_id: Hugging Face model identifier for Whisper
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_id = model_id
        self._device = device
        self._pipeline = None
    
    @property
    def device(self) -> int:
        """Get the device index for the pipeline."""
        if self._device is None:
            return 0 if torch.cuda.is_available() else -1
        return 0 if self._device == 'cuda' else -1
    
    @property
    def pipeline(self):
        """Lazy-load the ASR pipeline."""
        if self._pipeline is None:
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device=self.device
            )
        return self._pipeline
    
    def transcribe_audio(self, audio_path: str, sample_rate: int) -> dict:
        """
        Transcribe a full audio file with word-level timestamps.
        
        Args:
            audio_path: Path to the audio file
            sample_rate: Target sample rate for the audio
            
        Returns:
            Dictionary with 'text' and 'chunks' (word-level timestamps if available)
        """
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        
        # Try to get word-level timestamps, fallback to chunk-level, then text-only
        try:
            result = self.pipeline(
                {"raw": audio, "sampling_rate": sr}, 
                return_timestamps="word"
            )
        except Exception:
            try:
                result = self.pipeline(
                    {"raw": audio, "sampling_rate": sr}, 
                    return_timestamps=True
                )
            except Exception:
                result = self.pipeline({"raw": audio, "sampling_rate": sr})
                
                # Normalize result format
                if isinstance(result, dict) and 'text' in result:
                    result = {'text': result['text'], 'chunks': []}
                else:
                    result = {
                        'text': result if isinstance(result, str) else str(result), 
                        'chunks': []
                    }
        
        return result
    
    def transcribe_chunks(
        self, 
        audio: np.ndarray, 
        chunks: List[Tuple[float, float, np.ndarray]], 
        sample_rate: int
    ) -> List[dict]:
        """
        Transcribe multiple audio chunks.
        
        Args:
            audio: Full audio array (unused, kept for API compatibility)
            chunks: List of (start_time, end_time, chunk_audio) tuples
            sample_rate: Sample rate of the audio
            
        Returns:
            List of dictionaries with 'start_time', 'end_time', and 'transcript'
        """
        chunk_transcripts = []
        
        for start_time, end_time, chunk_audio in chunks:
            try:
                result = self.pipeline(
                    {"raw": chunk_audio, "sampling_rate": sample_rate}
                )
                transcript = result.get('text', '') if isinstance(result, dict) else str(result)
                chunk_transcripts.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'transcript': transcript.strip()
                })
            except Exception:
                chunk_transcripts.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'transcript': ''
                })
        
        return chunk_transcripts


# Global instance for convenience (lazy-loaded)
_transcriber = None


def get_transcriber(model_id: str = "openai/whisper-base", device: str | None = None) -> ASRTranscriber:
    """
    Get or create the global ASR transcriber instance.
    
    Args:
        model_id: Hugging Face model identifier
        device: Device to use ('cuda', 'cpu', or None for auto-detection)
        
    Returns:
        ASRTranscriber instance
    """
    global _transcriber
    if _transcriber is None:
        _transcriber = ASRTranscriber(model_id=model_id, device=device)
    return _transcriber

