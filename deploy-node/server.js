// 零依赖 Node.js 20 后端：接口与 Python 版完全一致
// 启动：node server.js   （监听 0.0.0.0:PORT）
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');

const PORT = Number(process.env.PORT || 8000);
const BASE_URL =
  process.env.LLM_BASE_URL || 'https://dashscope.aliyuncs.com/compatible-mode/v1';
const DEFAULT_MODEL = process.env.LLM_MODEL || 'deepseek-v4.1-flash';
const MODELS = (process.env.LLM_MODELS || 'deepseek-v4.1-flash,qwen-turbo,qwen-plus')
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean);
const EMBED_MODEL = process.env.EMBED_MODEL || 'text-embedding-v3';

const ROOT = __dirname;
const PUBLIC_DIR = path.join(ROOT, 'public');
const DATA_DIR = path.join(ROOT, 'data');

const MAX_CHUNK = 300;

const SYSTEM_PROMPT =
  '你是一名耐心的公司制度助手。回答简洁，只依据给定资料；资料里没有的内容直接说不知道，不要编造。';

const AGENT_PROMPT =
  '你是一名公司制度助手。需要制度信息时，先调用 search_policy 工具查询，' +
  '再根据查到的原文回答；查不到就直说没有相关规定，不要编造。';

const TOOLS = [
  {
    type: 'function',
    function: {
      name: 'search_policy',
      description:
        '在公司制度文档里检索相关条款，例如 住宿标准、报销时限、审批流程、餐补、账号申请',
      parameters: {
        type: 'object',
        properties: {
          keyword: { type: 'string', description: '要检索的主题词，尽量简短，例如 住宿' },
        },
        required: ['keyword'],
      },
    },
  },
];

// ---------- 基础工具 ----------
function sendJson(res, code, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(code, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body),
  });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let raw = '';
    req.on('data', (c) => {
      raw += c;
      if (raw.length > 1e6) reject(new Error('请求体过大'));
    });
    req.on('end', () => {
      try {
        resolve(raw ? JSON.parse(raw) : {});
      } catch {
        reject(new Error('请求体不是合法 JSON'));
      }
    });
    req.on('error', reject);
  });
}

function apiKeyOf(req) {
  const key = String(req.headers['x-api-key'] || '').trim();
  if (!key) throw new Error('请先在页面顶部填入你的 API Key');
  return key;
}

function authHeaders(key) {
  return { Authorization: 'Bearer ' + key, 'Content-Type': 'application/json' };
}

// ---------- 递归切块 ----------
function hardSplit(text, size = MAX_CHUNK) {
  const out = [];
  for (let i = 0; i < text.length; i += size) out.push(text.slice(i, i + size));
  return out;
}

function splitSentences(paragraph) {
  const pieces = paragraph
    .split(/(?<=[。！？；])|\n/)
    .map((s) => s.trim())
    .filter(Boolean);

  const out = [];
  for (const piece of pieces) {
    if (piece.length > MAX_CHUNK) out.push(...hardSplit(piece));
    else out.push(piece);
  }
  return out;
}

function packSentences(sentences, maxLen = MAX_CHUNK) {
  const chunks = [];
  let buf = '';
  for (const sentence of sentences) {
    if (buf && buf.length + sentence.length > maxLen) {
      chunks.push(buf);
      buf = sentence;
    } else {
      buf += sentence;
    }
  }
  if (buf) chunks.push(buf);
  return chunks;
}

function splitText(text) {
  const paragraphs = text
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter(Boolean);
  return packSentences(paragraphs.flatMap(splitSentences));
}

// ---------- RAG ----------
let indexCache = null;

function loadChunks() {
  const chunks = [];
  for (const name of fs.readdirSync(DATA_DIR).sort()) {
    if (!name.endsWith('.txt')) continue;
    const text = fs.readFileSync(path.join(DATA_DIR, name), 'utf8');
    for (const piece of splitText(text)) {
      chunks.push({ text: piece, source: name });
    }
  }
  return chunks;
}

async function embed(key, texts) {
  const resp = await fetch(BASE_URL + '/embeddings', {
    method: 'POST',
    headers: authHeaders(key),
    body: JSON.stringify({ model: EMBED_MODEL, input: texts }),
  });
  if (!resp.ok) {
    throw new Error('向量化失败 ' + resp.status + '：' + (await resp.text()).slice(0, 300));
  }
  const data = await resp.json();
  return data.data.map((d) => d.embedding);
}

async function buildIndex(key) {
  if (indexCache) return indexCache;
  const chunks = loadChunks();
  const vectors = await embed(key, chunks.map((c) => c.text));
  indexCache = { chunks, vectors };
  return indexCache;
}

function cosine(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return na && nb ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}

async function search(key, query, topK = 2) {
  const { chunks, vectors } = await buildIndex(key);
  const qv = (await embed(key, [query]))[0];
  return chunks
    .map((c, i) => ({ score: Number(cosine(qv, vectors[i]).toFixed(3)), ...c }))
    .sort((x, y) => y.score - x.score)
    .slice(0, topK);
}

// ---------- 对话 ----------
function sse(res, obj) {
  res.write('data: ' + JSON.stringify(obj) + '\n\n');
}

async function handleChat(req, res) {
  const key = apiKeyOf(req);
  const body = await readBody(req);
  const q = String(body.q || '').trim();
  const model = body.model || DEFAULT_MODEL;
  const history = Array.isArray(body.history) ? body.history.slice(-10) : [];
  const useKb = Boolean(body.use_kb);

  res.writeHead(200, {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });

  try {
    const messages = [{ role: 'system', content: SYSTEM_PROMPT }];
    for (const m of history) {
      if (m && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string') {
        messages.push({ role: m.role, content: m.content });
      }
    }

    if (useKb) {
      sse(res, { type: 'status', text: '正在检索资料…' });
      const hits = await search(key, q, 2);
      sse(res, {
        type: 'sources',
        sources: hits.map((h) => ({ source: h.source, score: h.score, text: h.text })),
      });
      messages.push({
        role: 'system',
        content:
          '以下是检索到的公司资料，回答必须依据它：\n\n' +
          hits.map((h) => h.text).join('\n\n'),
      });
    }

    messages.push({ role: 'user', content: q });

    const upstream = await fetch(BASE_URL + '/chat/completions', {
      method: 'POST',
      headers: authHeaders(key),
      body: JSON.stringify({ model, messages, stream: true, temperature: 0.3 }),
    });

    if (!upstream.ok) {
      sse(res, {
        type: 'error',
        text: '模型调用失败 ' + upstream.status + '：' + (await upstream.text()).slice(0, 300),
      });
      res.end();
      return;
    }

    const reader = upstream.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      let nl;
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (payload === '[DONE]') {
          res.write('data: [DONE]\n\n');
          res.end();
          return;
        }
        try {
          const piece = JSON.parse(payload).choices[0].delta.content;
          if (piece) sse(res, { type: 'delta', text: piece });
        } catch {
          // 跳过无法解析的片段
        }
      }
    }

    res.write('data: [DONE]\n\n');
    res.end();
  } catch (err) {
    sse(res, { type: 'error', text: err.message || String(err) });
    res.end();
  }
}

// ---------- Agent ----------
async function chatOnce(key, messages, model, tools) {
  const payload = { model, messages, temperature: 0 };
  if (tools) payload.tools = tools;

  const resp = await fetch(BASE_URL + '/chat/completions', {
    method: 'POST',
    headers: authHeaders(key),
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    throw new Error('模型调用失败 ' + resp.status + '：' + (await resp.text()).slice(0, 300));
  }
  const data = await resp.json();
  return data.choices[0].message;
}

async function handleAgent(req, res) {
  const key = apiKeyOf(req);
  const body = await readBody(req);
  const question = String(body.q || '').trim();
  const model = body.model || DEFAULT_MODEL;

  const messages = [
    { role: 'system', content: AGENT_PROMPT },
    { role: 'user', content: question },
  ];
  const steps = [];
  const sources = [];

  for (let i = 0; i < 5; i++) {
    const msg = await chatOnce(key, messages, model, TOOLS);

    if (!msg.tool_calls || !msg.tool_calls.length) {
      return sendJson(res, 200, { answer: msg.content || '', steps, sources });
    }

    messages.push(msg);

    for (const call of msg.tool_calls) {
      const name = call.function.name;
      let args = {};
      try {
        args = JSON.parse(call.function.arguments || '{}');
      } catch {
        args = {};
      }

      let result = '没有这个工具：' + name;
      let hits = [];
      if (name === 'search_policy') {
        hits = await search(key, args.keyword || '', 3);
        result = JSON.stringify(hits);
        for (const h of hits) {
          if (!sources.some((s) => s.text === h.text)) sources.push(h);
        }
      }

      steps.push({ tool: name, args, result: result.slice(0, 200) });
      messages.push({ role: 'tool', tool_call_id: call.id, content: result });
    }
  }

  sendJson(res, 200, { answer: '（思考步数超过了上限，先回答到这里）', steps, sources });
}

// ---------- 静态文件 ----------
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
};

function serveStatic(res, urlPath) {
  const rel = urlPath === '/' ? 'index.html' : urlPath.replace(/^\/+/, '');
  const file = path.join(PUBLIC_DIR, rel);
  if (!file.startsWith(PUBLIC_DIR)) {
    res.writeHead(403).end('Forbidden');
    return;
  }
  fs.readFile(file, (err, buf) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' }).end('Not Found');
      return;
    }
    res.writeHead(200, {
      'Content-Type': MIME[path.extname(file)] || 'application/octet-stream',
      'Cache-Control': 'no-cache',
    });
    res.end(buf);
  });
}

// ---------- 路由 ----------
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');
  const route = url.pathname;

  try {
    if (req.method === 'GET' && route === '/api/health') {
      return sendJson(res, 200, { ok: true });
    }
    if (req.method === 'GET' && route === '/api/config') {
      return sendJson(res, 200, { models: MODELS, default: DEFAULT_MODEL });
    }
    if (req.method === 'POST' && route === '/api/chat') {
      return await handleChat(req, res);
    }
    if (req.method === 'POST' && route === '/api/agent') {
      return await handleAgent(req, res);
    }
    if (req.method === 'GET' || req.method === 'HEAD') {
      return serveStatic(res, route);
    }
    sendJson(res, 405, { error: 'Method Not Allowed' });
  } catch (err) {
    if (!res.headersSent) {
      sendJson(res, 400, { error: err.message || String(err) });
    } else {
      res.end();
    }
  }
});

server.listen(PORT, '0.0.0.0', () => {
  console.log('policy-agent listening on 0.0.0.0:' + PORT);
});
