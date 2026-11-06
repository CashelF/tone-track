import {DIMENSIONS, escapeHtml as esc, formatTime, normalizeAnalysis, average, segmentAt, transcriptSegments, validateFile, toCsv, requestAnalysis} from './model.js';

const $ = id => document.getElementById(id);
const icon = name => `<svg class="icon" aria-hidden="true"><use href="#i-${name}"/></svg>`;
const colors = {valence:'#53844e', arousal:'#c0804b', dominance:'#6388b0'};
const sessionId = crypto.randomUUID();
const recordings = [];
const visibleSeries = new Set(DIMENSIONS);
const audio = $('audio');
let selected = null, selectedFile = null, previewUrl = null;
let recorder = null, stream = null, recordTimer = null, recordingStartedAt = 0, discardRecording = false;
let processingController = null, processingTimer = null, overviewBusy = false, overviewController = null;
let waveform = null, waveVersion = 0, transcript = [], toastTimer = null, mode = 'upload', chartWidth = 960;

function toast(message) {
  $('toast').textContent = message;
  $('toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, 4200);
}

function notice(message = '') { $('notice').textContent = message; $('notice').hidden = !message; }
function openDialog(id) { if (!$(id).open) $(id).showModal(); }
document.querySelectorAll('[data-close]').forEach(button => button.addEventListener('click', () => $(button.dataset.close).close()));
document.querySelectorAll('[data-action]').forEach(button => button.addEventListener('click', () => {
  if (button.dataset.action === 'about') openDialog('about-dialog');
  else openUpload(button.dataset.action === 'record' ? 'record' : 'upload');
}));
document.querySelectorAll('dialog').forEach(dialog => dialog.addEventListener('click', event => {
  if (event.target !== dialog) return;
  const rect = dialog.getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
}));

function renderSessions() {
  $('session-count').textContent = String(recordings.length).padStart(2, '0');
  $('sessions').innerHTML = recordings.map(recording => `<button class="session ${selected?.id === recording.id ? 'selected' : ''}" data-recording="${esc(recording.id)}" aria-pressed="${selected?.id === recording.id}"><span class="session-icon">${icon('wave')}</span><span class="session-copy"><strong>${esc(recording.name)}</strong><small>${formatTime(recording.duration)} · ${recording.sample ? 'Sample recording' : 'Your recording'}</small></span>${icon('chevron')}</button>`).join('');
  $('sessions').querySelectorAll('[data-recording]').forEach(button => button.addEventListener('click', () => selectRecording(recordings.find(item => item.id === button.dataset.recording))));
  $('overview-button').disabled = !recordings.some(item => !item.sample && item.analysis);
}

function selectRecording(recording) {
  if (!recording) return;
  audio.pause();
  selected = recording;
  audio.src = recording.audioUrl;
  audio.playbackRate = 1;
  $('speed-button').textContent = '1×';
  $('speed-button').setAttribute('aria-label', 'Playback speed 1 times');
  $('recording-name').textContent = recording.name;
  $('sample-badge').hidden = !recording.sample;
  $('recording-duration').textContent = `${formatTime(recording.duration)} recording`;
  $('recording-detail').textContent = recording.sample ? 'JFK · Inaugural address, 1961' : `${recording.chunks.length} analyzed segments`;
  $('total-time').textContent = formatTime(recording.duration);
  $('seek').max = recording.duration;
  $('seek').disabled = false;
  for (const id of ['play-button','rewind-button','speed-button','export-button']) $(id).disabled = false;
  notice(recording.partial ? 'Emotion scores and transcript are ready. AI notes are unavailable for this recording; you can still listen, explore, and export your results.' : '');
  renderSessions();
  renderMetrics();
  renderChart();
  renderTranscript();
  renderInsights();
  updatePosition(0);
  waveform = null;
  drawWaveform();
  decodeWaveform(recording);
}

function renderMetrics() {
  for (const key of DIMENSIONS) {
    $(key + '-score').textContent = average(selected.chunks, key).toFixed(2);
    const values = selected.chunks.map(chunk => Math.max(0, Math.min(1, chunk[key])));
    const line = values.map((value, index) => `${index ? 'L' : 'M'}${(index / Math.max(1, values.length - 1) * 128 + 2).toFixed(1)},${(36 - value * 31).toFixed(1)}`).join(' ');
    $(key + '-spark').innerHTML = `<path d="${line} L130 40 L2 40 Z" fill="currentColor" opacity=".07"/><path d="${line}" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/>`;
  }
}

function chartX(seconds) { return 36 + Math.min(selected.duration, Math.max(0, seconds)) / selected.duration * (chartWidth-54); }
function chartY(value) { return 195 - Math.max(0, Math.min(1, value)) * 173; }
function renderChart() {
  if (!selected) return;
  chartWidth = Math.max(200,$('emotion-chart').getBoundingClientRect().width);
  $('emotion-chart').setAttribute('viewBox',`0 0 ${chartWidth} 230`);
  const grid = [0,.25,.5,.75,1].map(value => `<line class="grid-line" x1="36" x2="${chartWidth-18}" y1="${chartY(value)}" y2="${chartY(value)}"/><text class="chart-label" x="27" y="${chartY(value)+3}" text-anchor="end">${value.toFixed(2)}</text>`).join('');
  const labels = Array.from({length:6}, (_, index) => `<text class="chart-label" x="${chartX(selected.duration * index/5)}" y="220" text-anchor="${index===0?'start':index===5?'end':'middle'}">${formatTime(selected.duration*index/5)}</text>`).join('');
  const lines = DIMENSIONS.filter(key => visibleSeries.has(key)).map(key => {
    const line = selected.chunks.map((chunk, index) => `${index?'L':'M'}${chartX((chunk.start_time+chunk.end_time)/2).toFixed(2)},${chartY(chunk[key]).toFixed(2)}`).join(' ');
    if (selected.chunks.length === 1) return `<circle cx="${chartX(selected.duration/2)}" cy="${chartY(selected.chunks[0][key])}" r="4" fill="${colors[key]}"/>`;
    return `${key === 'valence' ? `<path d="${line} L${chartX((selected.chunks.at(-1).start_time+selected.chunks.at(-1).end_time)/2)} 195 L${chartX((selected.chunks[0].start_time+selected.chunks[0].end_time)/2)} 195 Z" fill="${colors[key]}" opacity=".04"/>` : ''}<path class="series-path" d="${line}" stroke="${colors[key]}"/>`;
  }).join('');
  $('emotion-chart').innerHTML = `<title id="chart-title">Emotion scores over time</title><desc id="chart-description">Scores from zero to one. Use the audio position slider to explore, or download exact scores using Export.</desc>${grid}${labels}${lines}<g id="chart-cursor"><line class="playhead" y1="16" y2="195" x1="43" x2="43"/>${DIMENSIONS.filter(key => visibleSeries.has(key)).map(key => `<circle data-dot="${key}" r="4" stroke="#fff" stroke-width="2" fill="${colors[key]}"/>`).join('')}</g>`;
  updatePosition(audio.currentTime || 0);
}

document.querySelectorAll('[data-series]').forEach(button => button.addEventListener('click', () => {
  const key = button.dataset.series;
  if (visibleSeries.has(key)) {
    if (visibleSeries.size === 1) { toast('Keep at least one dimension visible.'); return; }
    visibleSeries.delete(key);
  } else visibleSeries.add(key);
  button.setAttribute('aria-pressed', String(visibleSeries.has(key)));
  renderChart();
}));

function secondsFromChart(event) {
  const rect = $('emotion-chart').getBoundingClientRect();
  return Math.max(0, Math.min(selected.duration, (((event.clientX-rect.left)/rect.width*chartWidth)-36)/(chartWidth-54)*selected.duration));
}
$('emotion-chart').addEventListener('click', event => { if(selected) seek(secondsFromChart(event)); });
$('emotion-chart').addEventListener('pointermove', event => {
  if (!selected) return;
  const time = secondsFromChart(event), chunk = segmentAt(selected.chunks,time);
  const tip = $('chart-tooltip'), rect = $('emotion-chart').getBoundingClientRect();
  tip.innerHTML = `<strong>${formatTime(chunk.start_time)} – ${formatTime(chunk.end_time)}</strong>${DIMENSIONS.filter(key => visibleSeries.has(key)).map(key => `<div><span style="color:${colors[key]}">${key[0].toUpperCase()+key.slice(1)}</span> ${chunk[key].toFixed(2)}</div>`).join('')}`;
  tip.style.left = `${Math.max(0, Math.min(rect.width-125, event.clientX-rect.left+12))}px`;
  tip.hidden = false;
});
$('emotion-chart').addEventListener('pointerleave', () => { $('chart-tooltip').hidden = true; });

function renderTranscript() {
  transcript = transcriptSegments(selected);
  $('transcript').innerHTML = transcript.length ? transcript.map((segment,index) => `<button class="transcript-row" data-segment="${index}" aria-label="Jump to ${formatTime(segment.start_time)}: ${esc(segment.text)}"><span class="transcript-time">${formatTime(segment.start_time)}</span><span class="transcript-text">${esc(segment.text)}</span></button>`).join('') : '<p class="empty-copy">No speech was transcribed in this recording.</p>';
  $('transcript').querySelectorAll('[data-segment]').forEach(button => button.addEventListener('click', () => seek(transcript[Number(button.dataset.segment)].start_time)));
}

function observations(recording) {
  const peak = recording.chunks.reduce((best, chunk) => chunk.arousal > best.arousal ? chunk : best);
  const start = recording.chunks[0], end = recording.chunks.at(-1);
  const delta = end.valence - start.valence;
  return [
    {title:'Where the energy peaks', text:`The highest arousal estimate is ${peak.arousal.toFixed(2)}, in the ${formatTime(peak.start_time)}–${formatTime(peak.end_time)} window. Listen to that moment in context.`, time:peak.start_time},
    {title:'From beginning to end', text:`Valence ${Math.abs(delta)<.04?'stays similar':delta>0?'rises':'falls'} from ${start.valence.toFixed(2)} in the first window to ${end.valence.toFixed(2)} in the last. These scores describe the model’s reading of the voice.`},
    {title:'Keep the full picture', text:'Emotion scores are estimates. Compare the curves with the words and the recording before drawing a conclusion.'}
  ];
}

function renderInsights() {
  if (selected.analysis) {
    $('insight-badge').textContent = 'AI NOTES';
    // Text only: model output and transcripts never become executable HTML.
    $('insights').innerHTML = selected.analysis.split(/\n\s*\n/).map(part => `<p>${esc(part).replace(/\n/g,'<br>')}</p>`).join('');
  } else {
    $('insight-badge').textContent = 'FROM THE SCORES';
    $('insights').innerHTML = observations(selected).map((item,index) => `<div class="insight-item"><span class="insight-number">0${index+1}</span><div><strong>${esc(item.title)}</strong><p>${esc(item.text)}</p>${item.time !== undefined ? `<button class="text-button" data-listen="${item.time}">Listen to this moment ${icon('arrow')}</button>` : ''}</div></div>`).join('');
    $('insights').querySelectorAll('[data-listen]').forEach(button => button.addEventListener('click', async () => { seek(Number(button.dataset.listen)); await play(); }));
  }
}

function updatePosition(time) {
  if (!selected) return;
  time = Math.max(0, Math.min(time, selected.duration));
  $('seek').value = time;
  $('seek').setAttribute('aria-valuetext', `${formatTime(time)} of ${formatTime(selected.duration)}`);
  $('elapsed').textContent = formatTime(time);
  const chunk = segmentAt(selected.chunks,time), cursor = $('chart-cursor');
  if (cursor) {
    cursor.querySelector('line').setAttribute('x1',chartX(time));
    cursor.querySelector('line').setAttribute('x2',chartX(time));
    cursor.querySelectorAll('[data-dot]').forEach(dot => { dot.setAttribute('cx',chartX(time)); dot.setAttribute('cy',chartY(chunk[dot.dataset.dot])); });
  }
  $('current-moment').textContent = `Window ${formatTime(chunk.start_time)}–${formatTime(chunk.end_time)} · ${chunk.valence.toFixed(2)} valence`;
  let current = transcript.findLastIndex(segment => segment.start_time <= time);
  $('transcript').querySelectorAll('[data-segment]').forEach(button => {
    const isCurrent = Number(button.dataset.segment) === current;
    button.classList.toggle('current',isCurrent);
    if (isCurrent) button.setAttribute('aria-current','true'); else button.removeAttribute('aria-current');
  });
  drawWaveform();
}

function seek(time) {
  if (!selected) return;
  const next = Math.max(0,Math.min(time,Number.isFinite(audio.duration) ? audio.duration : selected.duration));
  try { audio.currentTime = next; } catch { /* Metadata may still be loading. */ }
  updatePosition(next);
}
async function play() { try { await audio.play(); } catch { toast('Audio could not play. Try downloading or reselecting the recording.'); } }
$('play-button').addEventListener('click', () => { if (audio.paused) play(); else audio.pause(); });
$('rewind-button').addEventListener('click', () => seek(audio.currentTime-5));
$('seek').addEventListener('input', event => seek(Number(event.target.value)));
$('speed-button').addEventListener('click', () => {
  const speeds = [1,1.25,1.5,2,.75], speed = speeds[(speeds.indexOf(audio.playbackRate)+1)%speeds.length];
  audio.playbackRate = speed;
  $('speed-button').textContent = `${speed}×`;
  $('speed-button').setAttribute('aria-label',`Playback speed ${speed} times`);
});
audio.addEventListener('timeupdate', () => updatePosition(audio.currentTime));
for (const type of ['play','pause','ended']) audio.addEventListener(type, () => {
  $('play-button').innerHTML = icon(audio.paused ? 'play':'pause');
  $('play-button').classList.toggle('is-playing',!audio.paused);
  $('play-button').setAttribute('aria-label', audio.paused ? 'Play recording':'Pause recording');
});
audio.addEventListener('error', () => { if(selected) toast('This browser could not play the audio. The analysis and export are still available.'); });

async function decodeWaveform(recording) {
  const version = ++waveVersion;
  let context;
  try {
    context = new AudioContext();
    const bytes = recording.file ? await recording.file.arrayBuffer() : await (await fetch(recording.audioUrl)).arrayBuffer();
    const buffer = await context.decodeAudioData(bytes);
    const samples = buffer.getChannelData(0), buckets = 220, step = Math.max(1, Math.floor(samples.length/buckets));
    const peaks = Array.from({length:buckets}, (_, index) => {
      let sum = 0, count = 0;
      for(let j=index*step;j<Math.min(samples.length,(index+1)*step);j+=Math.max(1,Math.floor(step/120))) { sum+=samples[j]*samples[j]; count++; }
      return Math.sqrt(sum/Math.max(1,count));
    });
    const maximum = Math.max(...peaks,.001);
    if (version === waveVersion && selected?.id === recording.id) { waveform = peaks.map(value => value/maximum); drawWaveform(); }
  } catch { /* A flat track remains a usable seek control when decoding is unsupported. */ }
  finally { if(context) await context.close().catch(()=>{}); }
}

function drawWaveform() {
  const canvas = $('waveform'), rect = canvas.getBoundingClientRect();
  if (!rect.width) return;
  const ratio = Math.min(window.devicePixelRatio || 1,2);
  canvas.width = rect.width*ratio; canvas.height = rect.height*ratio;
  const ctx = canvas.getContext('2d');
  ctx.scale(ratio,ratio);
  const bars = Math.floor(rect.width/4), progress = selected ? Number($('seek').value)/selected.duration : 0;
  for(let i=0;i<bars;i++) {
    const peak = waveform ? waveform[Math.min(waveform.length-1,Math.floor(i/bars*waveform.length))] : .07;
    const height = Math.max(3,peak*(rect.height-4));
    ctx.fillStyle = i/bars <= progress ? '#86a75a' : '#d1dabe';
    ctx.fillRect(i*4,(rect.height-height)/2,2,height);
  }
}
new ResizeObserver(drawWaveform).observe($('waveform'));
new ResizeObserver(() => {
  if (selected && Math.abs($('emotion-chart').getBoundingClientRect().width-chartWidth)>1) renderChart();
}).observe($('emotion-chart'));

function setMode(next) {
  if (recorder?.state === 'recording') stopRecording(true);
  mode = next;
  $('upload-mode').hidden = next !== 'upload'; $('record-mode').hidden = next !== 'record';
  for (const key of ['upload','record']) { $(key+'-tab').classList.toggle('selected',next===key); $(key+'-tab').setAttribute('aria-pressed',String(next===key)); }
  clearSelectedFile();
  $('upload-error').hidden = true;
}

function openUpload(next) {
  setMode(next);
  $('recording-label').value = '';
  $('processing').hidden = true;
  openDialog('upload-dialog');
}
$('upload-tab').addEventListener('click',()=>setMode('upload'));
$('record-tab').addEventListener('click',()=>setMode('record'));
$('drop-zone').addEventListener('click',()=>$('file-input').click());
$('file-input').addEventListener('change',event=>chooseFile(event.target.files[0]));
for (const type of ['dragenter','dragover']) $('drop-zone').addEventListener(type,event=>{event.preventDefault();if(!processingController)$('drop-zone').classList.add('dragging');});
$('drop-zone').addEventListener('dragleave',()=> $('drop-zone').classList.remove('dragging'));
$('drop-zone').addEventListener('drop',event=>{event.preventDefault();$('drop-zone').classList.remove('dragging');if(!processingController)chooseFile(event.dataTransfer.files[0]);});

function clearSelectedFile() {
  selectedFile = null;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  $('audio-preview').pause(); $('audio-preview').removeAttribute('src'); $('audio-preview').load(); $('audio-preview').hidden=true;
  $('file-input').value = '';
  $('file-label').textContent = 'Drop a recording here';
  $('file-detail').textContent = 'or click to browse · up to 50 MB';
  $('analyze-button').disabled = true;
}
function formError(message) { $('upload-error').textContent=message; $('upload-error').hidden=!message; }
function chooseFile(file) {
  clearSelectedFile();
  const error = validateFile(file);
  formError(error);
  if(error) return;
  selectedFile = file;
  previewUrl = URL.createObjectURL(file);
  $('audio-preview').src = previewUrl; $('audio-preview').hidden=false;
  $('file-label').textContent = file.name;
  $('file-detail').textContent = `${(file.size/1024/1024).toFixed(1)} MB · click to choose another`;
  if(!$('recording-label').value.trim()) $('recording-label').value=file.name.replace(/\.[^.]+$/,'').slice(0,100);
  $('analyze-button').disabled=false;
}

function stopStream() { stream?.getTracks().forEach(track=>track.stop()); stream=null;clearInterval(recordTimer);recordTimer=null; }
function stopRecording(discard = false) {
  discardRecording = discard;
  if(recorder?.state === 'recording') recorder.stop();
  stopStream();
  $('record-toggle').classList.remove('recording');
  $('record-label').textContent = 'Record again';
}
$('record-toggle').addEventListener('click',async()=>{
  if(recorder?.state==='recording') { stopRecording(); return; }
  formError('');
  if(!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') { formError('Recording is not supported in this browser. Please upload an audio file instead.');return; }
  try {
    $('record-toggle').disabled = true;
    stream = await navigator.mediaDevices.getUserMedia({audio:true});
    if(!$('upload-dialog').open || mode !== 'record') { stopStream();return; }
    clearSelectedFile();
    discardRecording = false;
    const mime = ['audio/webm;codecs=opus','audio/mp4','audio/ogg;codecs=opus'].find(type=>MediaRecorder.isTypeSupported(type));
    recorder = new MediaRecorder(stream,mime?{mimeType:mime}:undefined);
    const parts=[];
    recorder.ondataavailable = event=>{if(event.data.size)parts.push(event.data);};
    recorder.onstop = ()=>{
      if(discardRecording || !$('upload-dialog').open || mode!=='record') return;
      const type = recorder.mimeType || parts[0]?.type || 'audio/webm';
      const ext = type.includes('mp4')?'m4a':type.includes('ogg')?'ogg':'webm';
      chooseFile(new File(parts,`Voice recording.${ext}`,{type}));
    };
    recorder.onerror = ()=>{stopRecording(true);formError('The recording was interrupted. Try again or upload a file.');};
    recorder.start(500);
    recordingStartedAt=Date.now();
    $('record-time').textContent='0:00';
    $('record-label').textContent='Stop recording';
    $('record-toggle').classList.add('recording');
    recordTimer=setInterval(()=>{
      const elapsed=(Date.now()-recordingStartedAt)/1000;
      $('record-time').textContent=formatTime(elapsed);
      if(elapsed>=120) {stopRecording();toast('Recording saved at the two-minute limit.');}
    },500);
  } catch { stopStream();formError('Microphone access was unavailable. Allow microphone access, or upload a recording instead.'); }
  finally { $('record-toggle').disabled=false; }
});

function setProcessing(busy) {
  for(const id of ['recording-label','file-input','drop-zone','record-toggle','upload-tab','record-tab']) $(id).disabled=busy;
  $('analyze-button').disabled=busy||!selectedFile;
  $('processing').hidden=!busy;
  $('cancel-upload').textContent=busy?'Stop waiting':'Cancel';
  $('audio-preview').pause();
}
$('upload-dialog').addEventListener('close',()=>{
  if(processingController) { processingController.abort();toast('Stopped waiting. The server may still finish processing.'); }
  stopRecording(true);
  clearSelectedFile();
});
$('upload-form').addEventListener('submit',async event=>{
  event.preventDefault();
  if(processingController || !selectedFile) return;
  const error = validateFile(selectedFile);
  if(error) {formError(error);return;}
  formError('');
  const file=selectedFile, name=$('recording-label').value.trim() || file.name.replace(/\.[^.]+$/,''), id=crypto.randomUUID();
  const form=new FormData(), ext=(file.name.match(/\.([a-zA-Z0-9]+)$/)?.[1] || 'webm').toLowerCase();
  // Unique server filenames keep recordings with the same user-visible name separate.
  form.append('audio',file,`${id}.${ext}`);
  form.append('session_id',sessionId);
  form.append('chunk_seconds','3');
  form.append('hop_seconds','3');
  const controller=new AbortController();
  processingController=controller;
  setProcessing(true);
  $('processing-title').textContent='Listening closely…';
  $('processing-detail').textContent='Transcribing and mapping the emotional arc. This can take a few minutes.';
  const start=Date.now();
  const timeout=setTimeout(()=>controller.abort('timeout'),600000);
  processingTimer=setInterval(()=>{
    $('processing-title').textContent=`Listening closely… ${formatTime((Date.now()-start)/1000)}`;
    if(Date.now()-start>60000) $('processing-detail').textContent='Still processing. The audio models can take several minutes, especially on the first recording.';
  },1000);
  try {
    const raw=await requestAnalysis('/process_and_analyze_audio',form,controller.signal);
    const result=normalizeAnalysis(raw);
    const recording={...result,id,name,sample:false,file,audioUrl:URL.createObjectURL(file),createdAt:new Date().toISOString()};
    recordings.unshift(recording);
    processingController=null;
    selectRecording(recording);
    $('upload-dialog').close();
    toast(result.partial?'Timeline ready. AI notes are temporarily unavailable.':'Your recording is ready for a closer listen.');
  } catch(error) {
    if(controller.signal.aborted) { if(controller.signal.reason==='timeout')formError('The service took longer than ten minutes. It may still be processing. Try a shorter recording later.'); }
    else formError(error.message);
  } finally {
    clearTimeout(timeout);clearInterval(processingTimer);processingTimer=null;processingController=null;setProcessing(false);
  }
});

$('export-button').addEventListener('click',()=>openDialog('export-dialog'));
function download(content, extension, type) {
  const url=URL.createObjectURL(new Blob([content],{type}));
  const link=document.createElement('a');
  link.href=url;link.download=`${selected.name.replace(/[^a-zA-Z0-9_-]/g,'_').slice(0,80)||'tone-track'}.${extension}`;
  document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),2000);
  $('export-dialog').close();toast('Analysis downloaded.');
}
$('export-csv').addEventListener('click',()=>{if(selected)download('\ufeff'+toCsv(selected),'csv','text/csv;charset=utf-8');});
$('export-json').addEventListener('click',()=>{
  if(!selected)return;
  const {name,duration,chunks,words,analysis,sample,createdAt}=selected;
  download(JSON.stringify({name,duration,chunks,words,analysis,sample,createdAt,source:'Tone Track',model:'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'},null,2),'json','application/json');
});

async function generateOverview() {
  if(overviewBusy)return;
  openDialog('overview-dialog');overviewBusy=true;
  $('overview-content').textContent='Looking for patterns across your recordings…';
  $('overview-retry').hidden=true;
  const controller=new AbortController();overviewController=controller;
  const timeout=setTimeout(()=>controller.abort('timeout'),120000);
  try {
    const form=new FormData();form.append('session_id',sessionId);
    const result=await requestAnalysis('/generate_overviews',form,controller.signal);
    if(result.status!=='success'||!result.overview)throw new Error('No overview was returned. Please try again.');
    $('overview-content').textContent=result.overview;
  } catch(error) {
    $('overview-content').textContent=controller.signal.aborted?'The overview request stopped before completing. You can try again.':error.message;
    $('overview-retry').hidden=false;
  } finally {clearTimeout(timeout);overviewBusy=false;overviewController=null;}
}
$('overview-button').addEventListener('click',generateOverview);
$('overview-retry').addEventListener('click',generateOverview);
$('overview-dialog').addEventListener('close',()=>overviewController?.abort());
window.addEventListener('pagehide',()=>{stopStream();processingController?.abort();overviewController?.abort();});

async function loadSample() {
  try {
    const response=await fetch('./sample.json');
    if(!response.ok)throw new Error('Sample unavailable');
    const raw=await response.json();
    const recording={...normalizeAnalysis(raw),id:'sample',name:'A call to a generation',sample:true,audioUrl:'./assets/jfk.flac',createdAt:null};
    recordings.push(recording);
    if(!selected)selectRecording(recording);else renderSessions();
  } catch {
    $('recording-name').textContent='Your next recording';
    $('recording-detail').textContent='Upload audio to get started';
    $('sample-badge').hidden=true;
    notice('The example could not load. You can still upload or record audio to create your own analysis.');
  }
}
loadSample();
