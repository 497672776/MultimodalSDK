'use strict';

const $ = (selector) => document.querySelector(selector);

const state = {
  files: { image: null, video: null, audio: null, scc: null, vlm: null },
};

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

function metrics(pairs) {
  const items = pairs
    .filter(([, value]) => value !== undefined && value !== null && value !== '')
    .map(([key, value]) => `<div>${escapeHtml(key)}：<b>${escapeHtml(value)}</b></div>`)
    .join('');
  return `<div class="metrics">${items}</div>`;
}

function ms(value) {
  return value === undefined || value === null ? undefined : `${Number(value).toFixed(1)} ms`;
}

function figure(src, caption) {
  return `<figure class="figure"><img src="${src}" alt="${escapeHtml(caption)}" /><figcaption>${escapeHtml(caption)}</figcaption></figure>`;
}

function showError(target, message) {
  $(target).innerHTML = `<div class="error">${escapeHtml(message)}</div>`;
}

function showBusy(target, text) {
  $(target).innerHTML = `<div class="muted"><span class="spinner"></span> ${escapeHtml(text)}</div>`;
}

async function postForm(url, formData) {
  const response = await fetch(url, { method: 'POST', body: formData });
  const text = await response.text();
  let payload;
  try {
    payload = JSON.parse(text);
  } catch (e) {
    throw new Error(`${response.status} ${text.slice(0, 300)}`);
  }
  if (!response.ok) {
    throw new Error(payload.detail || `${response.status} ${response.statusText}`);
  }
  return payload;
}

/* ------------------------------------------------------------------ tabs */
document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t === tab));
    document.querySelectorAll('.panel').forEach((panel) => {
      panel.classList.toggle('active', panel.id === `panel-${tab.dataset.panel}`);
    });
  });
});

/* -------------------------------------------------------------- uploads */
function wireDrop(key, inputSelector, dropSelector, preview) {
  const input = $(inputSelector);
  const drop = $(dropSelector);

  const adopt = (file) => {
    state.files[key] = file;
    input.value = '';
    drop.classList.add('selected');
    if (preview && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => { drop.innerHTML = `<img src="${e.target.result}" alt="preview" />`; };
      reader.readAsDataURL(file);
    } else {
      drop.innerHTML = `<span>${escapeHtml(file.name)}<br /><span class="muted">${(file.size / 1024).toFixed(1)} KiB</span></span>`;
    }
    const button = drop.closest('.card').querySelector('button.primary');
    if (button) button.disabled = false;
  };

  input.addEventListener('change', () => { if (input.files[0]) adopt(input.files[0]); });
  drop.addEventListener('dragover', (event) => { event.preventDefault(); drop.classList.add('dragover'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('dragover'));
  drop.addEventListener('drop', (event) => {
    event.preventDefault();
    drop.classList.remove('dragover');
    if (event.dataTransfer.files[0]) adopt(event.dataTransfer.files[0]);
  });
}

wireDrop('image', '#file-image', '#drop-image', true);
wireDrop('video', '#file-video', '#drop-video', false);
wireDrop('audio', '#file-audio', '#drop-audio', false);
wireDrop('scc', '#file-scc', '#drop-scc', true);
wireDrop('vlm', '#file-vlm', '#drop-vlm', true);

/* --------------------------------------------------------------- status */
async function loadStatus() {
  const box = $('#status');
  try {
    const response = await fetch('/api/status');
    const data = await response.json();
    const sdk = data.sdk;
    const runtime = data.runtime;
    const llama = data.llama;
    $('#deploy-path').textContent = sdk.path;
    const pills = [
      `<span class="pill ok">acc: ${escapeHtml(sdk.acc_backend)}</span>`,
      `<span class="pill ${runtime.torch ? 'ok' : 'bad'}">torch: ${runtime.torch ? '已安装' : '未安装'}</span>`,
      `<span class="pill ${runtime.ffmpeg ? 'ok' : 'bad'}">ffmpeg: ${runtime.ffmpeg ? '可用' : '缺失'}</span>`,
      `<span class="pill ${llama.online ? 'ok' : 'bad'}">llama-server: ${llama.online ? escapeHtml(llama.model || '在线') : '离线'}</span>`,
      `<span class="pill">${escapeHtml(runtime.machine)} · py${escapeHtml(runtime.python)}</span>`,
    ];
    box.innerHTML = pills.join('');
  } catch (error) {
    box.innerHTML = `<span class="pill bad">状态获取失败：${escapeHtml(error.message)}</span>`;
  }
}

/* ---------------------------------------------------------------- image */
$('#run-image').addEventListener('click', async () => {
  const form = new FormData();
  form.append('file', state.files.image);
  form.append('target', $('#image-target').value);
  form.append('size', $('#image-size').value);
  showBusy('#result-image', '正在预处理…');
  try {
    const data = await postForm('/api/image/preprocess', form);
    $('#result-image').innerHTML = [
      '<div class="grid">',
      figure(data.original.preview, `mm.Image 解码（${escapeHtml(data.decode_source)}）`),
      figure(data.preview, `归一化后预览 ${data.target.width}×${data.target.height}`),
      '</div>',
      metrics([
        ['解码耗时', ms(data.decode_ms)],
        ['预处理耗时', ms(data.preprocess_ms)],
        ['原图', `${data.original.size[0]}×${data.original.size[1]} · ${data.original.format} · ${data.original.dtype}`],
        ['目标尺寸', `${data.target.width}×${data.target.height}（${data.target.patches} patches）`],
        ['输出 tensor', `${(data.tensor.shape || []).join('×')} · ${data.tensor.dtype}`],
        ['数值范围', data.tensor.min !== undefined ? `${data.tensor.min.toFixed(3)} … ${data.tensor.max.toFixed(3)}（mean ${data.tensor.mean.toFixed(3)}）` : undefined],
        ['注意', data.warning],
      ]),
    ].join('');
  } catch (error) {
    showError('#result-image', error.message);
  }
});

/* ---------------------------------------------------------------- video */
$('#run-video').addEventListener('click', async () => {
  const form = new FormData();
  form.append('file', state.files.video);
  form.append('sample_num', $('#video-sample').value);
  form.append('frame_indices', $('#video-indices').value);
  showBusy('#result-video', '正在解码视频…');
  try {
    const data = await postForm('/api/video/decode', form);
    const thumbs = data.frames.map((frame) =>
      `<figure><img src="${frame.preview}" alt="frame ${frame.index}" /><figcaption>#${frame.index} · ${frame.size[0]}×${frame.size[1]}</figcaption></figure>`
    ).join('');
    $('#result-video').innerHTML = [
      metrics([
        ['解码帧数', data.decoded],
        ['总耗时', ms(data.decode_ms)],
        ['单帧耗时', ms(data.per_frame_ms)],
        ['请求', data.requested.frame_indices.length ? `帧号 ${data.requested.frame_indices.join(',')}` : `均匀抽 ${data.requested.sample_num} 帧`],
      ]),
      `<div class="thumbs">${thumbs}</div>`,
    ].join('');
  } catch (error) {
    showError('#result-video', error.message);
  }
});

/* ---------------------------------------------------------------- audio */
$('#run-audio').addEventListener('click', async () => {
  const form = new FormData();
  form.append('file', state.files.audio);
  form.append('sample_rate', $('#audio-sr').value);
  showBusy('#result-audio', '正在加载音频…');
  try {
    const data = await postForm('/api/audio/load', form);
    $('#result-audio').innerHTML = [
      metrics([
        ['采样率', `${data.sample_rate} Hz`],
        ['样本数', data.samples],
        ['时长', `${data.duration_s} s`],
        ['dtype', data.dtype],
        ['峰值/RMS', `${data.peak.toFixed(4)} / ${data.rms.toFixed(4)}`],
        ['加载耗时', ms(data.load_ms)],
      ]),
      `<canvas class="wave" id="wave-canvas" width="1000" height="120"></canvas>`,
    ].join('');
    drawWave(data.waveform);
  } catch (error) {
    showError('#result-audio', error.message);
  }
});

function drawWave(waveform) {
  const canvas = $('#wave-canvas');
  if (!canvas || !waveform.length) return;
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  const half = waveform.length / 2;
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = '#5ad1a8';
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  for (let x = 0; x < width; x += 1) {
    const index = Math.floor((x / width) * half);
    const max = waveform[index];
    const min = waveform[index + half];
    ctx.moveTo(x + 0.5, (1 - max) * height / 2);
    ctx.lineTo(x + 0.5, (1 - min) * height / 2);
  }
  ctx.stroke();
}

/* ------------------------------------------------------------------ scc */
$('#run-scc').addEventListener('click', async () => {
  const form = new FormData();
  form.append('file', state.files.scc);
  form.append('ratio', $('#scc-ratio').value);
  form.append('tau', $('#scc-tau').value);
  form.append('epsilon', $('#scc-eps').value);
  showBusy('#result-scc', '正在执行 SCC 压缩…');
  try {
    const data = await postForm('/api/scc/compress', form);
    $('#result-scc').innerHTML = [
      '<div class="grid">',
      figure(data.features_heatmap, `压缩前特征 ${data.tokens}×${data.feature_dim}`),
      figure(data.compressed_heatmap, `压缩后特征 ${data.output_shape[0]}×${data.output_shape[1]}`),
      '</div>',
      metrics([
        ['patch 网格', `${data.patch_grid[0]}×${data.patch_grid[1]}`],
        ['token 数', `${data.tokens} → ${data.target_tokens}`],
        ['保留比例', `${(data.keep_ratio * 100).toFixed(1)}%`],
        ['是否触发 SCC', data.should_run ? '是' : '否（超过 8192 token 阈值）'],
        ['tau / epsilon', `${data.tau} / ${data.epsilon}`],
        ['压缩耗时', ms(data.scc_ms)],
      ]),
    ].join('');
  } catch (error) {
    showError('#result-scc', error.message);
  }
});

/* ------------------------------------------------------------------ vlm */
$('#run-vlm').addEventListener('click', async () => {
  const form = new FormData();
  form.append('file', state.files.vlm);
  form.append('question', $('#vlm-question').value);
  form.append('max_tokens', $('#vlm-maxtokens').value);
  form.append('temperature', $('#vlm-temp').value);
  form.append('max_side', $('#vlm-maxside').value);
  showBusy('#result-vlm', 'llama-server 正在推理…（4B 模型在 K3 上通常需要数十秒）');
  try {
    const data = await postForm('/api/vlm/chat', form);
    $('#result-vlm').innerHTML = [
      figure(data.preview, '发送给模型的图像'),
      `<div class="answer">${escapeHtml(data.answer)}</div>`,
      metrics([
        ['mm 预处理', ms(data.prepare_ms)],
        ['推理耗时', ms(data.infer_ms)],
        ['生成 token', data.usage.completion_tokens],
        ['提示 token', data.usage.prompt_tokens],
        ['吞吐', data.tokens_per_s ? `${data.tokens_per_s} tok/s` : undefined],
        ['结束原因', data.finish_reason],
        ['服务端', data.llama_url],
      ]),
    ].join('');
  } catch (error) {
    showError('#result-vlm', error.message);
  }
});

loadStatus();
