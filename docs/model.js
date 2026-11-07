export const DIMENSIONS = ['valence', 'arousal', 'dominance'];
export const MAX_BYTES = 50 * 1024 * 1024;
export const API_BASE = 'https://cashel-tone-track.hf.space';

export function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char]));
}

export function formatTime(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
}

export function normalizeAnalysis(data) {
  if (!['success', 'partial_success'].includes(data?.status) || !Array.isArray(data.chunks_csv_data)) {
    throw new Error('The service could not analyze this recording. Try a shorter audio file.');
  }
  const chunks = data.chunks_csv_data.map(row => {
    const chunk = {start_time:Number(row.start_time), end_time:Number(row.end_time), transcript:String(row.transcript || '')};
    for (const dimension of DIMENSIONS) chunk[dimension] = Number(row[dimension]);
    if (![chunk.start_time, chunk.end_time, ...DIMENSIONS.map(key => chunk[key])].every(Number.isFinite)
      || chunk.start_time < 0 || chunk.end_time <= chunk.start_time
      || DIMENSIONS.some(key => row[key] == null || row[key] === '')) {
      throw new Error('The service returned incomplete emotion scores. Please try again.');
    }
    return chunk;
  }).sort((a, b) => a.start_time - b.start_time);
  if (!chunks.length) throw new Error('No speech segments were returned. Try a recording with clear speech.');
  const words = (Array.isArray(data.words_csv_data) ? data.words_csv_data : []).map(row => ({
    word:String(row.word || '').trim(), start_time:Number(row.start_time), end_time:Number(row.end_time)
  })).filter(row => row.word && Number.isFinite(row.start_time) && Number.isFinite(row.end_time)
    && row.start_time >= 0 && row.end_time >= row.start_time).sort((a,b) => a.start_time - b.start_time);
  return {chunks, words, analysis: typeof data.analysis === 'string' ? data.analysis : '', partial: data.status === 'partial_success', duration:Math.max(...chunks.map(row => row.end_time))};
}

export function average(chunks, dimension) {
  return chunks.reduce((total, row) => total + row[dimension], 0) / chunks.length;
}

export function segmentAt(chunks, seconds) {
  let nearest = chunks[0];
  for (const row of chunks) {
    if (Math.abs((row.start_time + row.end_time) / 2 - seconds) < Math.abs((nearest.start_time + nearest.end_time) / 2 - seconds)) nearest = row;
  }
  return nearest;
}

export function transcriptSegments(recording) {
  if (!recording.words.length) return recording.chunks.filter(row => row.transcript.trim()).map(row => ({...row, text:row.transcript}));
  const segments = [];
  let current = null;
  for (const word of recording.words) {
    if (!current || (word.start_time - current.start_time >= 4) || (/[.!?]$/.test(current.text) && word.start_time - current.start_time >= 1.5)) {
      current = {start_time:word.start_time, end_time:word.end_time, text:word.word};
      segments.push(current);
    } else {
      current.text += ` ${word.word}`;
      current.end_time = word.end_time;
    }
  }
  return segments;
}

export function validateFile(file) {
  if (!file || !file.size) return 'Choose an audio file that contains a recording.';
  if (file.size > MAX_BYTES) return 'This file is larger than 50 MB. Please choose a smaller recording.';
  if (!file.type?.startsWith('audio/') && !/\.(wav|mp3|m4a|ogg|flac|webm|mp4|aac|aiff|aif|opus)$/i.test(file.name)) {
    return 'Please choose an audio file, such as WAV, MP3, M4A, FLAC, or WebM.';
  }
  return '';
}

function csvCell(value) {
  let text = String(value ?? '');
  // Spreadsheet programs interpret leading formula characters even in quoted cells.
  if (typeof value === 'string' && /^[\s]*[=+\-@]/.test(text)) text = "'" + text;
  return `"${text.replace(/"/g, '""')}"`;
}

export function toCsv(recording) {
  const columns = ['start_time', 'end_time', ...DIMENSIONS, 'transcript'];
  return [columns.join(','), ...recording.chunks.map(row => columns.map(key => csvCell(row[key])).join(','))].join('\r\n');
}

export async function requestAnalysis(path, form, signal) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, {method:'POST', body:form, signal, credentials:'omit'});
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new Error('The analysis service could not be reached. It may be waking up. Please try again shortly.');
  }
  if (!response.ok) {
    if (response.status === 422) {
      const detail = await response.json().catch(() => ({}));
      if (detail.code === 'invalid_audio') throw new Error('This file could not be decoded. Try recording again, or upload a WAV or MP3 file.');
    }
    if (response.status === 413) throw new Error('The recording is too large for the analysis service. Try a smaller file.');
    if (response.status === 404 && path === '/generate_overviews') throw new Error('No AI summaries are available yet. Analyze a recording with AI notes first.');
    if (response.status === 429) throw new Error('The service is busy. Please wait a moment before trying again.');
    throw new Error(`The analysis service returned an error (${response.status}). Please try again shortly.`);
  }
  try { return await response.json(); }
  catch { throw new Error('The service returned an unexpected response. Please try again shortly.'); }
}
