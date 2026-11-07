import test from 'node:test';
import assert from 'node:assert/strict';
import {normalizeAnalysis, transcriptSegments, segmentAt, toCsv, validateFile, escapeHtml, formatTime, requestAnalysis} from '../docs/model.js';

const response = status => ({status, chunks_csv_data:[
  {start_time:'0',end_time:'3',valence:'0.2',arousal:'0.7',dominance:'0.6',transcript:'A beginning'},
  {start_time:'3',end_time:'6',valence:'0.8',arousal:'0.5',dominance:'0.4',transcript:'An ending'}
]});

test('AI failure preserves valid audio results and does not fabricate notes',()=>{
  const value=normalizeAnalysis(response('partial_success'));
  assert.equal(value.partial,true);assert.equal(value.analysis,'');assert.equal(value.duration,6);assert.equal(value.chunks[1].valence,.8);
});
test('malformed responses cannot produce a misleading chart',()=>{
  for(const data of [{status:'success'}, {status:'success',chunks_csv_data:[]}, {status:'error',chunks_csv_data:response('success').chunks_csv_data}])assert.throws(()=>normalizeAnalysis(data));
  for(const invalid of ['NaN',null,'',Infinity]){const data=response('success');data.chunks_csv_data[0].valence=invalid;assert.throws(()=>normalizeAnalysis(data));}
  const data=response('success');data.chunks_csv_data[0].end_time='-1';assert.throws(()=>normalizeAnalysis(data));
});
test('seeking chooses nearest analysis window, including boundaries',()=>{
  const {chunks}=normalizeAnalysis(response('success'));
  assert.equal(segmentAt(chunks,0),chunks[0]);assert.equal(segmentAt(chunks,6),chunks[1]);assert.equal(segmentAt(chunks,4),chunks[1]);
});
test('transcript uses aligned words instead of duplicating overlapping chunks',()=>{
  const data=response('success');data.words_csv_data=[{word:'Hello.',start_time:0,end_time:2},{word:'Again.',start_time:4,end_time:5}];
  assert.deepEqual(transcriptSegments(normalizeAnalysis(data)),[{text:'Hello.',start_time:0,end_time:2},{text:'Again.',start_time:4,end_time:5}]);
  assert.equal(transcriptSegments(normalizeAnalysis(response('success')))[1].text,'An ending');
});
test('CSV quotes transcripts and neutralizes spreadsheet formula injection',()=>{
  const recording=normalizeAnalysis(response('success'));
  recording.chunks[0].transcript='=HYPERLINK("https://example.com")';
  const csv=toCsv(recording);
  assert.match(csv, /"'=HYPERLINK\(""https:\/\/example.com""\)"/);
  assert.match(csv,/start_time,end_time,valence,arousal,dominance,transcript/);
});
test('file validation accepts browser recordings and rejects invalid or oversized files',()=>{
  assert.equal(validateFile({name:'recording.webm',type:'audio/webm;codecs=opus',size:100}), '');
  assert.equal(validateFile({name:'voice.FLAC',type:'',size:100}), '');
  assert.ok(validateFile({name:'file.html',type:'text/html',size:100}));
  assert.ok(validateFile({name:'file.wav',type:'audio/wav',size:51*1024*1024}));
  assert.ok(validateFile({name:'file.wav',type:'audio/wav',size:0}));
});
test('untrusted text is escaped and time formatting handles invalid inputs',()=>{
  assert.equal(escapeHtml('<img src=x onerror="alert(1)">'),'&lt;img src=x onerror=&quot;alert(1)&quot;&gt;');
  assert.equal(formatTime(61.9),'1:01');assert.equal(formatTime(-4),'0:00');assert.equal(formatTime(NaN),'0:00');
});
test('invalid audio has a useful error without exposing server internals',async t=>{
  t.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({code:'invalid_audio',error:'/private/server/path'}), {status:422}));
  await assert.rejects(requestAnalysis('/process_and_analyze_audio',new FormData()), /could not be decoded.*WAV or MP3/);
});
