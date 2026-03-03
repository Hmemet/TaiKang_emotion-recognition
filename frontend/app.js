const inputText = document.getElementById('inputText');
const batchText = document.getElementById('batchText');
const topK = document.getElementById('topK');
const predictBtn = document.getElementById('predictBtn');
const batchBtn = document.getElementById('batchBtn');
const healthBtn = document.getElementById('healthBtn');
const clearBtn = document.getElementById('clearBtn');
const exampleBtns = document.querySelectorAll('.example-btn');
const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');
const historyEl = document.getElementById('history');
const chartEl = document.getElementById('chart');
const detailMetaEl = document.getElementById('detailMeta');
const detailTitleEl = document.getElementById('detailTitle');

const tabPredict = document.getElementById('tabPredict');
const tabHistory = document.getElementById('tabHistory');
const modeSingle = document.getElementById('modeSingle');
const modeBatch = document.getElementById('modeBatch');
const backFromDetail = document.getElementById('backFromDetail');
const clearHistoryBtn = document.getElementById('clearHistory');
const exportDetailBtn = document.getElementById('exportDetail');

const predictPage = document.getElementById('predictPage');
const historyPage = document.getElementById('historyPage');
const detailPage = document.getElementById('detailPage');
const singleMode = document.getElementById('singleMode');
const batchMode = document.getElementById('batchMode');

const API_BASE = 'http://127.0.0.1:8000';
const HISTORY_KEY = 'emotion_prediction_history';
const HISTORY_STORAGE = window.sessionStorage;

let currentPage = 'predict';
let lastPageBeforeDetail = 'predict';
let currentDetailRecord = null;
let historyRecords = loadHistory();

function setLoadingState(loading) {
  predictBtn.disabled = loading;
  batchBtn.disabled = loading;
  healthBtn.disabled = loading;
}

function loadHistory() {
  try {
    const raw = HISTORY_STORAGE.getItem(HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (_) {
    return [];
  }
}

function saveHistory() {
  HISTORY_STORAGE.setItem(HISTORY_KEY, JSON.stringify(historyRecords.slice(0, 100)));
}

function clearHistory() {
  historyRecords = [];
  HISTORY_STORAGE.removeItem(HISTORY_KEY);
  renderHistory();
  setStatus('历史记录已清空');
}

function switchMainPage(page) {
  currentPage = page;
  const showPredict = page === 'predict';
  const showHistory = page === 'history';
  const showDetail = page === 'detail';

  predictPage.classList.toggle('hidden', !showPredict);
  historyPage.classList.toggle('hidden', !showHistory);
  detailPage.classList.toggle('hidden', !showDetail);

  tabPredict.classList.toggle('active', showPredict);
  tabHistory.classList.toggle('active', showHistory);
}

function switchMode(mode) {
  const isSingle = mode === 'single';
  singleMode.classList.toggle('hidden', !isSingle);
  batchMode.classList.toggle('hidden', isSingle);
  modeSingle.classList.toggle('active', isSingle);
  modeBatch.classList.toggle('active', !isSingle);
}

function normalizeTopK() {
  const k = Number(topK.value || 5);
  if (Number.isNaN(k)) {
    topK.value = 5;
    return 5;
  }
  const normalized = Math.max(1, Math.min(28, Math.floor(k)));
  topK.value = normalized;
  return normalized;
}

async function safeJson(response) {
  try {
    return await response.json();
  } catch (_) {
    return {};
  }
}

function setStatus(text) {
  statusEl.textContent = text;
}

function createList(title, data) {
  const wrapper = document.createElement('div');
  wrapper.className = 'result-block';

  const titleEl = document.createElement('div');
  titleEl.className = 'result-title';
  titleEl.textContent = title;
  wrapper.appendChild(titleEl);

  const ul = document.createElement('ul');
  ul.className = 'result-list';

  if (!Array.isArray(data) || data.length === 0) {
    const li = document.createElement('li');
    li.textContent = '无';
    ul.appendChild(li);
  } else {
    data.forEach(item => {
      const li = document.createElement('li');
      li.textContent = `${item.label}: ${item.score}`;
      ul.appendChild(li);
    });
  }

  wrapper.appendChild(ul);
  return wrapper;
}

function openDetail(record) {
  lastPageBeforeDetail = currentPage === 'detail' ? 'predict' : currentPage;
  currentDetailRecord = record;
  detailTitleEl.textContent = `预测详情 - ${record.mode === 'batch' ? `第 ${record.index + 1} 条` : '单条'}`;
  detailMetaEl.textContent = `文本：${record.text_processed || ''}`;
  chartEl.innerHTML = '';

  const scores = [...(record.all_scores || [])].sort((a, b) => b.score - a.score);
  scores.forEach(item => {
    const row = document.createElement('div');
    row.className = 'bar-row';

    const label = document.createElement('div');
    label.className = 'bar-label';
    label.textContent = item.label;

    const barWrap = document.createElement('div');
    barWrap.className = 'bar-wrap';

    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.style.width = `${Math.max(2, Math.round(item.score * 100))}%`;

    barWrap.appendChild(bar);

    const value = document.createElement('span');
    value.className = 'bar-value-out';
    value.textContent = item.score.toFixed(4);

    row.appendChild(label);
    row.appendChild(barWrap);
    row.appendChild(value);
    chartEl.appendChild(row);
  });

  switchMainPage('detail');
}

function exportCurrentDetail() {
  if (!currentDetailRecord) {
    setStatus('暂无可导出的详情数据');
    return;
  }

  const blob = new Blob([JSON.stringify(currentDetailRecord, null, 2)], { type: 'application/json;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  const stamp = new Date().toISOString().replace(/[.:]/g, '-');
  anchor.href = url;
  anchor.download = `emotion_detail_${stamp}.json`;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
  setStatus('详情 JSON 已导出');
}

function addHistory(record) {
  historyRecords.unshift({
    ...record,
    created_at: new Date().toLocaleString()
  });
  historyRecords = historyRecords.slice(0, 100);
  saveHistory();
  renderHistory();
}

function renderHistory() {
  historyEl.innerHTML = '';
  if (!historyRecords.length) {
    historyEl.textContent = '暂无历史记录';
    historyEl.className = 'result-empty';
    return;
  }

  historyEl.className = '';
  historyRecords.forEach(record => {
    const block = document.createElement('div');
    block.className = 'result-block';

    const title = document.createElement('div');
    title.className = 'result-title';
    title.textContent = `${record.created_at} | ${record.mode === 'batch' ? `批量第 ${record.index + 1} 条` : '单条预测'}`;
    block.appendChild(title);

    const textLine = document.createElement('div');
    textLine.innerHTML = `<strong>文本：</strong>${record.text_processed || ''}`;
    block.appendChild(textLine);

    const topLine = document.createElement('div');
    const top = (record.summary_scores || [])[0];
    topLine.innerHTML = `<strong>最高情绪：</strong>${top ? `${top.label} (${top.score})` : '无'}`;
    block.appendChild(topLine);

    const btnRow = document.createElement('div');
    btnRow.className = 'row';
    const detailBtn = document.createElement('button');
    detailBtn.className = 'secondary';
    detailBtn.textContent = '预测详情';
    detailBtn.addEventListener('click', () => openDetail(record));
    btnRow.appendChild(detailBtn);
    block.appendChild(btnRow);

    historyEl.appendChild(block);
  });
}

function renderResult(payload) {
  resultEl.innerHTML = '';

  const basic = document.createElement('div');
  basic.className = 'result-block';
  basic.innerHTML = `<div><strong>清洗后文本：</strong>${payload.text_processed || ''}</div><div><strong>耗时：</strong>${payload.cost_ms ?? '-'} ms</div>`;

  resultEl.appendChild(basic);
  resultEl.appendChild(createList('命中情绪', payload.detected_emotions));
  const requestedTopK = normalizeTopK();
  const summaryScores = (payload.top_k_scores || []).slice(0, requestedTopK);
  resultEl.appendChild(createList(`Top ${requestedTopK} 分数`, summaryScores));

  const detailRecord = {
    mode: 'single',
    index: 0,
    text_processed: payload.text_processed || '',
    all_scores: payload.top_k_scores || [],
    summary_scores: summaryScores
  };

  const row = document.createElement('div');
  row.className = 'row';
  const detailBtn = document.createElement('button');
  detailBtn.className = 'secondary';
  detailBtn.textContent = '预测详情';
  detailBtn.addEventListener('click', () => openDetail(detailRecord));
  row.appendChild(detailBtn);
  resultEl.appendChild(row);

  addHistory(detailRecord);
}

async function doPredict() {
  const text = inputText.value.trim();
  normalizeTopK();

  if (!text) {
    setStatus('请输入文本后再预测。');
    return;
  }

  setLoadingState(true);
  setStatus('预测中...');

  try {
    const response = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, top_k: 28 })
    });

    const data = await safeJson(response);
    if (!response.ok) {
      throw new Error(data.detail || '预测失败');
    }

    setStatus('预测成功');
    renderResult(data);
  } catch (error) {
    setStatus(`预测失败: ${error.message}`);
  } finally {
    setLoadingState(false);
  }
}

async function doHealthCheck() {
  setLoadingState(true);
  setStatus('健康检查中...');

  try {
    const response = await fetch(`${API_BASE}/api/health`);
    const data = await safeJson(response);

    if (!response.ok) {
      throw new Error(data.detail || '健康检查失败');
    }

    setStatus(`健康状态: ${data.status} | 设备: ${data.device}`);
  } catch (error) {
    setStatus(`健康检查失败: ${error.message}`);
  } finally {
    setLoadingState(false);
  }
}

function renderBatchResult(payload) {
  resultEl.innerHTML = '';

  const summary = document.createElement('div');
  summary.className = 'result-block';
  summary.innerHTML = `<div><strong>总条数：</strong>${payload.count ?? 0}</div><div><strong>耗时：</strong>${payload.cost_ms ?? '-'} ms</div>`;
  resultEl.appendChild(summary);

  const requestedTopK = normalizeTopK();

  (payload.results || []).forEach(item => {
    const block = document.createElement('div');
    block.className = 'result-block';

    const title = document.createElement('div');
    title.className = 'result-title';
    title.textContent = `第 ${item.index + 1} 条（${item.status}）`;
    block.appendChild(title);

    const textLine = document.createElement('div');
    textLine.innerHTML = `<strong>清洗后文本：</strong>${item.text_processed || ''}`;
    block.appendChild(textLine);

    block.appendChild(createList('命中情绪', item.detected_emotions));
    const summaryScores = (item.top_k_scores || []).slice(0, requestedTopK);
    block.appendChild(createList(`Top ${requestedTopK} 分数`, summaryScores));

    const detailRecord = {
      mode: 'batch',
      index: item.index,
      text_processed: item.text_processed || '',
      all_scores: item.top_k_scores || [],
      summary_scores: summaryScores
    };

    const row = document.createElement('div');
    row.className = 'row';
    const detailBtn = document.createElement('button');
    detailBtn.className = 'secondary';
    detailBtn.textContent = '预测详情';
    detailBtn.addEventListener('click', () => openDetail(detailRecord));
    row.appendChild(detailBtn);
    block.appendChild(row);

    addHistory(detailRecord);
    resultEl.appendChild(block);
  });
}

async function doBatchPredict() {
  let lines = (batchText.value || '')
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(Boolean);

  if (lines.length === 0 && inputText.value.trim()) {
    lines = [inputText.value.trim()];
  }

  normalizeTopK();

  if (lines.length === 0) {
    setStatus('请在批量输入框中每行输入一条文本（或在单条文本框输入后再点批量）。');
    return;
  }

  setLoadingState(true);
  setStatus('批量预测中...');

  try {
    const response = await fetch(`${API_BASE}/api/predict/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts: lines, top_k: 28 })
    });

    const data = await safeJson(response);
    if (!response.ok) {
      throw new Error(data.detail || '批量预测失败');
    }

    const failed = (data.results || []).filter(item => item.status !== 'success').length;
    setStatus(failed > 0 ? `批量预测完成：成功 ${data.count - failed} 条，失败 ${failed} 条` : '批量预测成功');
    renderBatchResult(data);
  } catch (error) {
    setStatus(`批量预测失败: ${error.message}`);
  } finally {
    setLoadingState(false);
  }
}

predictBtn.addEventListener('click', doPredict);
batchBtn.addEventListener('click', doBatchPredict);
healthBtn.addEventListener('click', doHealthCheck);
tabPredict.addEventListener('click', () => switchMainPage('predict'));
tabHistory.addEventListener('click', () => {
  renderHistory();
  switchMainPage('history');
});
modeSingle.addEventListener('click', () => switchMode('single'));
modeBatch.addEventListener('click', () => switchMode('batch'));
backFromDetail.addEventListener('click', () => {
  switchMainPage(lastPageBeforeDetail === 'history' ? 'history' : 'predict');
});
clearHistoryBtn.addEventListener('click', clearHistory);
exportDetailBtn.addEventListener('click', exportCurrentDetail);

// 清除内容按钮
clearBtn.addEventListener('click', () => {
  inputText.value = '';
  setStatus('已清除输入内容');
});

// 示例句子点击填充
exampleBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    inputText.value = btn.dataset.text;
    setStatus('已填充示例文本');
  });
});

switchMainPage('predict');
switchMode('single');
renderHistory();
