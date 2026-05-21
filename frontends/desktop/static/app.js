// GenericAgent 桌面版 —— bridge 适配 + 业务 UI（HTTP 命令 / WS 状态 / i18n）。
// 文案全部走 i18n：静态用 data-i18n / data-i18n-ph / data-i18n-title，
// 动态用 t(key)。dev 标注层与发给 agent 的预设 prompt 不进 UI 字典。
'use strict';

/* ═══════════════ 进程状态 store ═══════════════ */
const _serviceById = {};
const _serviceListeners = new Set();

function _serviceList() {
  return Object.values(_serviceById).sort((a, b) => String(a.id).localeCompare(String(b.id)));
}

function _serviceNotify() {
  const items = _serviceList();
  for (const cb of _serviceListeners) {
    try { cb(items, _serviceById); } catch (e) { console.error('[service-store]', e); }
  }
}

const gaServiceStore = {
  applySnapshot(services) {
    for (const k of Object.keys(_serviceById)) delete _serviceById[k];
    for (const s of services || []) {
      if (s && s.id) _serviceById[s.id] = s;
    }
    _serviceNotify();
  },
  applyChanged(service) {
    if (service && service.id) _serviceById[service.id] = service;
    _serviceNotify();
  },
  onServices(cb) {
    _serviceListeners.add(cb);
    cb(_serviceList(), _serviceById);
    return () => _serviceListeners.delete(cb);
  },
  list: _serviceList,
  get: (id) => _serviceById[id],
};

/* ═══════════════ Bridge 适配（HTTP 命令 + WS 状态） ═══════════════ */
(function initGaBridge() {
  const listeners = new Map();
  let ws = null;
  let cachedBridgeReady = null;
  const bridgeBase = `${location.protocol}//${location.hostname}:14168`;
  const wsUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.hostname}:14168/ws`;

  function on(channel, cb) {
    if (typeof cb !== 'function') return () => {};
    if (!listeners.has(channel)) listeners.set(channel, new Set());
    listeners.get(channel).add(cb);
    if (channel === 'bridge-ready' && cachedBridgeReady) {
      try { cb(cachedBridgeReady); } catch (err) { console.error('[ga bridge] replay bridge-ready', err); }
    }
    return () => listeners.get(channel)?.delete(cb);
  }

  function emit(channel, payload) {
    if (channel === 'bridge-ready') cachedBridgeReady = payload;
    const set = listeners.get(channel);
    if (!set) return;
    for (const cb of Array.from(set)) {
      try { cb(payload); } catch (err) { console.error('[ga bridge]', channel, err); }
    }
  }

  function handleServiceWs(msg) {
    if (msg.type === 'services.snapshot') gaServiceStore.applySnapshot(msg.services);
    else if (msg.type === 'service.changed') gaServiceStore.applyChanged(msg.service);
    emit('service-state', msg);
  }

  async function http(path, options = {}) {
    const headers = Object.assign({}, options.headers || {});
    const init = Object.assign({}, options, { headers });
    if (init.body && typeof init.body !== 'string') {
      headers['Content-Type'] = headers['Content-Type'] || 'application/json';
      init.body = JSON.stringify(init.body);
    }
    const res = await fetch(`${bridgeBase}${path}`, init);
    const text = await res.text();
    let data = null;
    try { data = text ? JSON.parse(text) : {}; } catch (_) { data = { raw: text }; }
    if (!res.ok) {
      const err = new Error((data && (data.error || data.message)) || `${res.status} ${res.statusText}`);
      err.status = res.status;
      err.data = data;
      throw err;
    }
    return data;
  }

  function connectWs() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
    try {
      ws = new WebSocket(wsUrl);
      ws.addEventListener('open', () => emit('bridge-log', 'WS connected'));
      ws.addEventListener('message', (ev) => {
        let msg;
        try { msg = JSON.parse(ev.data); } catch (_) { return; }
        if (msg.type === 'bridge-ready') emit('bridge-ready', msg);
        else if (msg.type === 'services.snapshot' || msg.type === 'service.changed') handleServiceWs(msg);
        else if (msg.type === 'session-state') emit('bridge-notification', msg);
        else if (msg.type === 'bridge-log') emit('bridge-log', msg.payload || msg);
        else if (msg.type === 'bridge-error') emit('bridge-error', msg.payload || msg);
      });
      ws.addEventListener('close', () => emit('bridge-closed', { reason: 'ws-closed' }));
      ws.addEventListener('error', () => emit('bridge-error', { type: 'ws-error', message: 'WebSocket error' }));
    } catch (err) {
      emit('bridge-error', { type: 'ws-error', message: err.message || String(err) });
    }
  }

  async function rpc(method, params = {}) {
    switch (method) {
      case 'app/status': return http('/status');
      case 'app/config/get': return http('/config');
      case 'app/config/save': return http('/config', { method: 'POST', body: params || {} });
      case 'get/model-profiles': return http('/model-profiles');
      case 'session/new': return http('/session/new', { method: 'POST', body: params || {} });
      case 'session/prompt': {
        const sid = params.sessionId || params.id || params.bridgeSessionId;
        if (!sid) throw new Error('session/prompt missing sessionId');
        return http(`/session/${encodeURIComponent(sid)}/prompt`, { method: 'POST', body: params || {} });
      }
      case 'session/poll': {
        const sid = params.sessionId || params.id || params.bridgeSessionId;
        if (!sid) throw new Error('session/poll missing sessionId');
        const after = params.afterId ?? params.after ?? 0;
        const limit = params.limit ?? 200;
        return http(`/session/${encodeURIComponent(sid)}/messages?after=${encodeURIComponent(after)}&limit=${encodeURIComponent(limit)}`);
      }
      case 'session/cancel': {
        const sid = params.sessionId || params.id || params.bridgeSessionId;
        if (!sid) throw new Error('session/cancel missing sessionId');
        return http(`/session/${encodeURIComponent(sid)}/cancel`, { method: 'POST', body: params || {} });
      }
      case 'app/path/open': return http('/path/open', { method: 'POST', body: params || {} });
      case 'services/start': {
        const id = params.id;
        if (!id) throw new Error('services/start missing id');
        return http('/services/start', { method: 'POST', body: { id } });
      }
      case 'services/stop': {
        const id = params.id;
        if (!id) throw new Error('services/stop missing id');
        return http('/services/stop', { method: 'POST', body: { id } });
      }
      case 'services/logs': {
        const id = params.id;
        if (!id) throw new Error('services/logs missing id');
        const tail = params.tail ?? 200;
        return http(`/services/logs?id=${encodeURIComponent(id)}&tail=${encodeURIComponent(tail)}`);
      }
      case 'services/panel': return http('/services/panel');
      case 'app/path/selectGaRoot': return http('/config');
      case 'list_continuable_sessions': return { sessions: [] };
      case 'restore_session': throw new Error('restore_session is not implemented in web2 bridge');
      default: throw new Error(`Unknown RPC method: ${method}`);
    }
  }

  async function startService(id) {
    try {
      const res = await rpc('services/start', { id });
      if (res.service) gaServiceStore.applyChanged(res.service);
      return res;
    } catch (e) {
      if (e.data && e.data.service) gaServiceStore.applyChanged(e.data.service);
      throw e;
    }
  }

  async function stopService(id) {
    const res = await rpc('services/stop', { id });
    if (res.service) gaServiceStore.applyChanged(res.service);
    return res;
  }

  window.ga = {
    platform: navigator.platform.toLowerCase().includes('mac') ? 'darwin' : 'win32',
    startBridge: async () => { connectWs(); return http('/status'); },
    stopBridge: async () => ({ ok: true }),
    checkStatus: () => rpc('app/status', {}),
    getConfig: () => rpc('app/config/get', {}),
    saveConfig: (cfg) => rpc('app/config/save', cfg || {}),
    getModelProfiles: () => rpc('get/model-profiles', {}),
    selectGaRoot: () => rpc('app/path/selectGaRoot', {}),
    openMykeyTemplate: () => rpc('app/path/open', { kind: 'mykeyTemplate' }),
    openMykey: () => rpc('app/path/open', { kind: 'mykey' }),
    startService,
    stopService,
    getServiceLogs: (id, tail = 200) => rpc('services/logs', { id, tail }),
    getServicePanel: () => rpc('services/panel', {}),
    pollSession: (sessionId, afterId = 0) => rpc('session/poll', { sessionId, afterId }),
    rpc,
    onBridgeMessage: (cb) => on('bridge-message', cb),
    onBridgeNotification: (cb) => on('bridge-notification', cb),
    onBridgeError: (cb) => on('bridge-error', cb),
    onBridgeClosed: (cb) => on('bridge-closed', cb),
    onBridgeReady: (cb) => on('bridge-ready', cb),
    onBridgeLog: (cb) => on('bridge-log', cb),
    onServiceState: (cb) => on('service-state', cb),
    onOpenSearch: (cb) => on('open-search', cb),
  };

  connectWs();
  http('/status').then(status => emit('bridge-ready', status))
    .catch(err => emit('bridge-error', { type: 'http-error', message: err.message || String(err) }));
})();

/* ═══════════════ i18n ═══════════════ */
const I18N = {
  zh: {
    'app.title': 'GenericAgent 桌面版',
    'brand.sub': '桌面终端',
    'nav.chat': '聊天', 'nav.channels': '消息通道', 'nav.status': '状态面板',
    'nav.collab': '协作动态', 'nav.token': 'Token 统计',
    'foot.settings': '配置', 'foot.ver': 'GenericAgent · 桌面版',
    'chat.startTitle': '开始对话', 'chat.startSub': '直接输入，或点预设功能一键启动',
    'preset.goal.t': 'Goal 模式', 'preset.goal.d': '设定目标，自主完成',
    'preset.explore.t': '自主探索', 'preset.explore.d': '自动浏览并周期汇总',
    'preset.hive.t': 'Hive 协作', 'preset.hive.d': '多 worker 协同攻坚',
    'preset.review.t': '深度复核', 'preset.review.d': '挑刺式质量把关',
    'preset.mine.t': '我的·周报', 'preset.mine.d': '自定义：抓本周提交并写周报',
    'preset.add.t': '自定义', 'preset.add.d': '任意一句话存为功能',
    'composer.placeholder': '输入消息… (Enter 发送, Shift+Enter 换行)',
    'search.placeholder': '搜索会话…', 'conv.new': '新对话',
    'ctx.pin': '置顶', 'ctx.del': '删除',
    'common.close': '关闭', 'common.more': '更多', 'common.save': '保存',
    'modal.preset': '预设功能', 'modal.addModel': '添加模型', 'modal.settings': '配置',
    'set.theme': '主题色', 'set.lang': '语言', 'set.model': '模型', 'set.addModel': '添加模型',
    'page.channels.title': '消息通道', 'page.channels.sub': '后台 IM 进程：列表、启停与日志（同 hub.pyw）',
    'page.status.title': '状态面板', 'page.status.sub': 'hub.pyw 管理的后台进程/服务，集中查看与启停',
    'page.collab.title': '协作动态', 'page.collab.sub': 'subagent / Hive worker 的实时状态与产出',
    'page.token.title': 'Token 统计', 'page.token.sub': '每会话与累计的 token 用量及估算成本',
    'status.connecting': '连接中…', 'status.ready': '就绪', 'status.running': '运行中',
    'status.disconnected': '未连接', 'status.stopped': '已停止', 'status.idle': '空闲',
    'conv.emptyList': '暂无会话，点「＋ 新对话」开始', 'conv.defaultTitle': '新对话',
    'err.bridge': 'bridge 未连接', 'err.newSession': '新建会话失败', 'err.poll': '轮询失败', 'err.stop': '停止失败',
    'sys.stopRequested': '已请求停止',
    'slash.help': '可用命令：\n/new 新会话  /clear 清屏  /stop 停止  /settings 设置',
    'slash.unknown': '未知命令',
    'upload.hint': '图片上传：粘贴图片到输入框即可（多模态接入中）',
    'fold.thinking': '思考', 'fold.tool': '工具调用', 'fold.toolResult': '工具结果', 'fold.llm': 'LLM Running',
    'model.auto': '自动选择',
    'ch.wechat': '微信', 'ch.wecom': '企业微信', 'ch.lark': '飞书', 'ch.dingtalk': '钉钉',
    'ch.qq': 'QQ', 'ch.telegram': 'Telegram', 'ch.discord': 'Discord',
    'ch.loading': '加载中…', 'ch.empty': '未发现 IM 进程脚本',
    'ch.logEmpty': '暂无日志',
    'err.channelLoad': '加载失败', 'err.channelStart': '启动失败', 'err.channelStop': '停止失败',
    'err.channelNotConfigured': '请先在 mykey.py 中配置该平台',
    'sys.channelStarted': '已启动', 'sys.channelStopped': '已停止',
    'modal.channelLogs': '进程日志',
    'st.starting': '启动中…', 'st.stopping': '停止中…',
    'st.online': '在线', 'st.offline': '离线', 'st.error': '错误', 'st.running': '运行', 'st.abnormal': '异常',
    'act.configure': '配置', 'act.logs': '日志', 'act.restart': '重启', 'act.stop': '停止', 'act.start': '启动',
    'proc.imbotWechat': 'imbot · 微信', 'proc.imbotDing': 'imbot · 钉钉', 'proc.scheduler': '定时任务调度',
    'cm.scheduling': '调度中', 'cm.running': '执行中', 'cm.idleSt': '空闲',
    'cm.master': '已派 3 子任务', 'cm.w1': '子任务：抓取数据', 'cm.w2': '子任务：复核结果', 'cm.sub': '等待派单',
    'tok.total': '累计 token', 'tok.cost': '估算成本', 'tok.today': '今日 token',
    'tok.colSession': '会话', 'tok.colIn': '输入', 'tok.colOut': '输出', 'tok.colCacheW': '缓存写入', 'tok.colCache': '缓存读取', 'tok.colCost': '成本',
    'tok.from': '从', 'tok.to': '到', 'tok.reset': '重置', 'tok.noData': '暂无记录',
    'presetPrompt.goal': '进入 Goal 模式：读 L3 goal mode SOP，自主达成我接下来描述的目标。',
    'presetPrompt.explore': '进入自主探索模式：自动浏览并定期向我汇总要点。',
    'presetPrompt.hive': '启动 Goal Hive 模式：按 hive SOP 拉起多个 worker 协同完成我接下来的目标。',
    'presetPrompt.review': '进入监察者模式：对刚才的产出严格挑刺、逐项复核并报告问题。',
    'presetPrompt.mine': '抓取本周的 git 提交并写一份周报。',
  },
  en: {
    'app.title': 'GenericAgent Desktop',
    'brand.sub': 'Desktop terminal',
    'nav.chat': 'Chat', 'nav.channels': 'Channels', 'nav.status': 'Status',
    'nav.collab': 'Collaboration', 'nav.token': 'Token usage',
    'foot.settings': 'Settings', 'foot.ver': 'GenericAgent · Desktop',
    'chat.startTitle': 'Start a conversation', 'chat.startSub': 'Type a message, or pick a preset',
    'preset.goal.t': 'Goal mode', 'preset.goal.d': 'Set a goal, run autonomously',
    'preset.explore.t': 'Auto explore', 'preset.explore.d': 'Browse & summarize periodically',
    'preset.hive.t': 'Hive', 'preset.hive.d': 'Multi-worker collaboration',
    'preset.review.t': 'Deep review', 'preset.review.d': 'Strict quality check',
    'preset.mine.t': 'My · Weekly', 'preset.mine.d': 'Custom: weekly report from commits',
    'preset.add.t': 'Custom', 'preset.add.d': 'Save any prompt as a function',
    'composer.placeholder': 'Type a message… (Enter to send, Shift+Enter for newline)',
    'search.placeholder': 'Search chats…', 'conv.new': 'New chat',
    'ctx.pin': 'Pin', 'ctx.del': 'Delete',
    'common.close': 'Close', 'common.more': 'More', 'common.save': 'Save',
    'modal.preset': 'Presets', 'modal.addModel': 'Add model', 'modal.settings': 'Settings',
    'set.theme': 'Theme color', 'set.lang': 'Language', 'set.model': 'Model', 'set.addModel': 'Add model',
    'page.channels.title': 'Channels', 'page.channels.sub': 'Background IM processes: list, start/stop, logs (hub.pyw style)',
    'page.status.title': 'Status', 'page.status.sub': 'Background processes/services managed by hub.pyw',
    'page.collab.title': 'Collaboration', 'page.collab.sub': 'Live state & output of subagents / Hive workers',
    'page.token.title': 'Token usage', 'page.token.sub': 'Per-session and total token usage & estimated cost',
    'status.connecting': 'Connecting…', 'status.ready': 'Ready', 'status.running': 'Running',
    'status.disconnected': 'Disconnected', 'status.stopped': 'Stopped', 'status.idle': 'Idle',
    'conv.emptyList': 'No chats yet — click “＋ New chat”', 'conv.defaultTitle': 'New chat',
    'err.bridge': 'Bridge not connected', 'err.newSession': 'Failed to create session', 'err.poll': 'Polling failed', 'err.stop': 'Stop failed',
    'sys.stopRequested': 'Stop requested',
    'slash.help': 'Commands:\n/new new chat  /clear clear  /stop stop  /settings settings',
    'slash.unknown': 'Unknown command',
    'upload.hint': 'Image upload: paste an image into the input box (multimodal WIP)',
    'fold.thinking': 'Thinking', 'fold.tool': 'Tool call', 'fold.toolResult': 'Tool result', 'fold.llm': 'LLM Running',
    'model.auto': 'Auto',
    'ch.wechat': 'WeChat', 'ch.wecom': 'WeCom', 'ch.lark': 'Lark', 'ch.dingtalk': 'DingTalk',
    'ch.qq': 'QQ', 'ch.telegram': 'Telegram', 'ch.discord': 'Discord',
    'ch.loading': 'Loading…', 'ch.empty': 'No IM process scripts found',
    'ch.logEmpty': 'No log output yet',
    'err.channelLoad': 'Failed to load', 'err.channelStart': 'Start failed', 'err.channelStop': 'Stop failed',
    'err.channelNotConfigured': 'Configure this platform in mykey.py first',
    'sys.channelStarted': 'Started', 'sys.channelStopped': 'Stopped',
    'modal.channelLogs': 'Process logs',
    'st.starting': 'Starting…', 'st.stopping': 'Stopping…',
    'st.online': 'Online', 'st.offline': 'Offline', 'st.error': 'Error', 'st.running': 'Running', 'st.abnormal': 'Error',
    'act.configure': 'Configure', 'act.logs': 'Logs', 'act.restart': 'Restart', 'act.stop': 'Stop', 'act.start': 'Start',
    'proc.imbotWechat': 'imbot · WeChat', 'proc.imbotDing': 'imbot · DingTalk', 'proc.scheduler': 'Scheduler',
    'cm.scheduling': 'Scheduling', 'cm.running': 'Running', 'cm.idleSt': 'Idle',
    'cm.master': 'Dispatched 3 subtasks', 'cm.w1': 'Subtask: fetch data', 'cm.w2': 'Subtask: review results', 'cm.sub': 'Waiting for tasks',
    'tok.total': 'Total tokens', 'tok.cost': 'Est. cost', 'tok.today': 'Today tokens',
    'tok.colSession': 'Session', 'tok.colIn': 'Input', 'tok.colOut': 'Output', 'tok.colCacheW': 'Cache write', 'tok.colCache': 'Cache read', 'tok.colCost': 'Cost',
    'tok.from': 'From', 'tok.to': 'To', 'tok.reset': 'Reset', 'tok.noData': 'No records',
    'presetPrompt.goal': 'Enter Goal mode: read the L3 goal-mode SOP and autonomously achieve the goal I describe next.',
    'presetPrompt.explore': 'Enter auto-explore mode: browse autonomously and periodically summarize key points to me.',
    'presetPrompt.hive': 'Start Goal Hive mode: per the hive SOP, spawn multiple workers to collaboratively achieve the goal I describe next.',
    'presetPrompt.review': 'Enter reviewer mode: strictly scrutinize the previous output, review item by item and report issues.',
    'presetPrompt.mine': 'Collect this week\'s git commits and write a weekly report.',
  },
};
let lang = (localStorage.getItem('ga_lang') === 'en') ? 'en' : 'zh';
function t(key) { return (I18N[lang] && I18N[lang][key]) || (I18N.zh[key]) || key; }
function applyI18n() {
  document.documentElement.lang = (lang === 'en') ? 'en' : 'zh-CN';
  document.title = t('app.title');
  document.querySelectorAll('[data-i18n]').forEach(el => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll('[data-i18n-ph]').forEach(el => { el.setAttribute('placeholder', t(el.dataset.i18nPh)); });
  document.querySelectorAll('[data-i18n-title]').forEach(el => { el.setAttribute('title', t(el.dataset.i18nTitle)); });
}

/* ═══════════════ 侧边栏导航 ═══════════════ */
const nav = document.getElementById('nav');
const pages = document.querySelectorAll('#pages .page');
nav.addEventListener('click', (e) => {
  const item = e.target.closest('.nav-item');
  if (!item) return;
  const key = item.dataset.page;
  nav.querySelectorAll('.nav-item').forEach(n => n.classList.toggle('active', n === item));
  pages.forEach(p => p.classList.toggle('active', p.dataset.page === key));
});

/* ═══════════════ 弹窗开关 ═══════════════ */
const openModal = (id) => { const m = document.getElementById(id); if (m) m.hidden = false; };
const closeModals = () => document.querySelectorAll('.modal').forEach(m => m.hidden = true);
const bindClick = (id, fn) => { const el = document.getElementById(id); if (el) el.addEventListener('click', fn); };
bindClick('add-model-btn', (e) => { e.stopPropagation(); openModal('add-model-modal'); });
bindClick('settings-btn',  (e) => { e.stopPropagation(); openModal('settings-modal'); });
bindClick('preset-btn',    (e) => { e.stopPropagation(); openModal('preset-modal'); });
document.querySelectorAll('.modal').forEach(m =>
  m.addEventListener('click', (e) => { if (e.target.closest('[data-close]')) m.hidden = true; }));
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModals(); });

/* ═══════════════ Markdown ═══════════════ */
if (typeof marked !== 'undefined') {
  marked.setOptions({ gfm: true, breaks: true, mangle: false, headerIds: false });
}
const ALLOWED_URI_RE = /^(https?:|mailto:|tel:|#|\/)/i;
function escapeHtml(s) {
  const d = document.createElement('div'); d.textContent = String(s == null ? '' : s); return d.innerHTML;
}
function sanitizeMarkdown(html) {
  const tpl = document.createElement('template');
  tpl.innerHTML = String(html);
  const blocked = new Set(['SCRIPT','STYLE','IFRAME','OBJECT','EMBED','LINK','META','BASE','FORM','INPUT','BUTTON']);
  const walker = document.createTreeWalker(tpl.content, NodeFilter.SHOW_ELEMENT);
  const rmv = [];
  while (walker.nextNode()) {
    const el = walker.currentNode;
    if (blocked.has(el.tagName)) { rmv.push(el); continue; }
    for (const attr of Array.from(el.attributes)) {
      const n = attr.name.toLowerCase(), v = attr.value.trim();
      if (n.startsWith('on') || n === 'srcdoc') { el.removeAttribute(attr.name); continue; }
      if ((n === 'href' || n === 'src' || n === 'xlink:href') && v && !ALLOWED_URI_RE.test(v)) el.removeAttribute(attr.name);
    }
    if (el.tagName === 'A') { el.setAttribute('rel','noopener noreferrer'); el.setAttribute('target','_blank'); }
  }
  rmv.forEach(el => el.remove());
  return tpl.innerHTML;
}
function renderMarkdown(text) {
  if (typeof marked === 'undefined') return escapeHtml(text).replace(/\n/g, '<br>');
  try { return sanitizeMarkdown(marked.parse(String(text || ''))); }
  catch (_) { return escapeHtml(text); }
}
function renderAssistant(text) {
  let s = String(text || '');
  const folds = [];
  const stash = (label, body) => { folds.push({ label, body }); return ` F${folds.length - 1} `; };
  s = s.replace(/<thinking>[\s\S]*?<\/thinking>/gi, m => stash(t('fold.thinking'), m.replace(/<\/?thinking>/gi, '')));
  s = s.replace(/<function_calls>[\s\S]*?<\/function_calls>/gi, m => stash(t('fold.tool'), m));
  s = s.replace(/<function_results>[\s\S]*?<\/function_results>/gi, m => stash(t('fold.toolResult'), m));
  s = s.replace(/(\**LLM Running \(Turn \d+\) \.\.\.\**)/g, m => stash(t('fold.llm'), m));
  let html = renderMarkdown(s);
  html = html.replace(/F(\d+)/g, (_, i) => {
    const f = folds[Number(i)];
    return `<details class="fold"><summary>${escapeHtml(f.label)}</summary><pre>${escapeHtml(f.body)}</pre></details>`;
  });
  return html;
}

/* ═══════════════ 状态 ═══════════════ */
const state = {
  sessions: new Map(), activeId: null, bridgeReady: false,
  llmNo: 0, modelProfiles: [], modelName: null,
  runtime: new Map(),
};
function rt(sess) {
  let r = state.runtime.get(sess.id);
  if (!r) { r = { polling:false, busy:false, lastId:0, seen:new Set(), draftEl:null, draftText:'' }; state.runtime.set(sess.id, r); }
  return r;
}
const activeSess = () => state.sessions.get(state.activeId) || null;
const isActive = (sess) => sess && sess.id === state.activeId;

/* ═══════════════ DOM refs ═══════════════ */
const chatPage   = document.querySelector('.page[data-page="chat"]');
const msgArea    = chatPage.querySelector('.msg-area');
const chatStart  = msgArea.querySelector('.chat-start');
const inputEl    = chatPage.querySelector('.input');
const sendBtn    = chatPage.querySelector('.send');
const runToggle  = document.getElementById('run-toggle');
const runLabel   = runToggle.querySelector('.rs-label');
const convListEl = document.querySelector('.conv-list');
const newConvBtn = document.querySelector('.new-conv');
const searchInput = document.querySelector('.search input');
const rpToggle   = document.getElementById('rp-toggle');
const rpResize   = document.getElementById('rp-resize');
const rpPanel    = document.getElementById('rightpanel');
const bodyEl     = document.querySelector('.body');
if (rpToggle) rpToggle.addEventListener('click', () => bodyEl.classList.toggle('rp-collapsed'));

if (rpResize && rpPanel) {
  let dragging = false, startX = 0, startW = 0;
  rpResize.addEventListener('mousedown', (e) => {
    dragging = true; startX = e.clientX; startW = rpPanel.offsetWidth;
    rpResize.classList.add('dragging');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    e.preventDefault();
  });
  document.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const w = Math.min(400, Math.max(160, startW + (startX - e.clientX)));
    rpPanel.style.width = w + 'px';
    rpPanel.style.flex = '0 0 ' + w + 'px';
  });
  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    rpResize.classList.remove('dragging');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  });
}
const modelChip  = document.getElementById('model-chip');
const modelNameEl= modelChip ? modelChip.querySelector('.model-name') : null;
const langSel    = document.getElementById('lang-select');

let msgsEl = null;
function ensureMsgs() {
  if (!msgsEl) { msgsEl = document.createElement('div'); msgsEl.className = 'msgs'; msgArea.appendChild(msgsEl); }
  return msgsEl;
}
function refreshEmptyState(sess) {
  const has = sess && sess.messages.length > 0;
  msgArea.classList.toggle('has-msgs', !!has);
  if (chatStart) chatStart.style.display = has ? 'none' : '';
  if (msgsEl) msgsEl.style.display = has ? '' : 'none';
}

/* ═══════════════ 消息渲染 ═══════════════ */
function msgNode(msg) {
  const el = document.createElement('div');
  el.className = 'msg ' + (msg.role || 'system');
  if (msg.role === 'user') el.innerHTML = `<div class="bubble">${escapeHtml(msg.content)}</div>`;
  else if (msg.role === 'assistant') el.innerHTML = `<div class="bubble md">${renderAssistant(msg.content)}</div>`;
  else if (msg.role === 'error') el.innerHTML = `<div class="bubble err">${escapeHtml(msg.content)}</div>`;
  else el.innerHTML = `<div class="bubble sys">${escapeHtml(msg.content)}</div>`;
  return el;
}
function renderAllMessages(sess) {
  const box = ensureMsgs(); box.innerHTML = '';
  for (const m of sess.messages) box.appendChild(msgNode(m));
  refreshEmptyState(sess); scrollBottom();
}
function appendMessage(sess, msg) {
  if (!isActive(sess)) return;
  ensureMsgs().appendChild(msgNode(msg));
  refreshEmptyState(sess); scrollBottom();
}
function scrollBottom() { requestAnimationFrame(() => { msgArea.scrollTop = msgArea.scrollHeight; }); }
function renderDraft(sess) {
  const r = rt(sess);
  if (!isActive(sess)) return;
  const box = ensureMsgs();
  if (!r.draftEl || r.draftEl.parentNode !== box) {
    r.draftEl = document.createElement('div'); r.draftEl.className = 'msg assistant'; box.appendChild(r.draftEl);
  }
  r.draftEl.innerHTML = `<div class="bubble md">${renderAssistant(r.draftText)}<span class="cursor"></span></div>`;
  refreshEmptyState(sess); scrollBottom();
}

/* ═══════════════ 运行状态 ═══════════════ */
function statusLabel() {
  const s = activeSess();
  if (s && rt(s).busy) return t('status.running');
  return state.bridgeReady ? t('status.ready') : t('status.disconnected');
}
function refreshStatusLabel() { if (!runToggle.classList.contains('stopped')) runLabel.textContent = statusLabel(); }
function setBusy(sess, busy) {
  const r = rt(sess); r.busy = busy;
  if (!isActive(sess)) return;
  runToggle.classList.remove('stopped');
  runToggle.classList.toggle('busy', busy);
  runLabel.textContent = busy ? t('status.running') : (state.bridgeReady ? t('status.ready') : t('status.disconnected'));
  sendBtn.disabled = busy;
}
runToggle.addEventListener('click', async () => {
  const sess = activeSess();
  if (sess && rt(sess).busy) {
    await cancelPrompt();
    runLabel.textContent = t('status.stopped');
    runToggle.classList.add('stopped');
  }
});

/* ═══════════════ 会话 ═══════════════ */
function isUntitled(x) { return !x || /^(new chat|新对话|新会话)$/i.test(String(x).trim()); }
function renderSessionList() {
  convListEl.innerHTML = '';
  const query = (searchInput ? searchInput.value : '').trim().toLowerCase();
  const all = [...state.sessions.values()];
  const filtered = query
    ? all.filter(s => {
        const title = (s.title || '').toLowerCase();
        const hasMsg = s.messages && s.messages.some(m => (m.text || '').toLowerCase().includes(query));
        return title.includes(query) || hasMsg;
      })
    : all;
  if (filtered.length === 0) {
    const e = document.createElement('div');
    e.className = 'conv-empty'; e.textContent = t('conv.emptyList');
    convListEl.appendChild(e); return;
  }
  for (const sess of filtered) {
    const r = state.runtime.get(sess.id);
    const busy = !!(r && r.busy);
    const item = document.createElement('div');
    item.className = 'conv-item' + (sess.id === state.activeId ? ' active' : '') + (busy ? '' : ' idle');
    item.dataset.id = sess.id;
    item.innerHTML =
      `<span class="ci-dot"></span><div class="ci-main">` +
      `<div class="ci-title">${escapeHtml(sess.title || t('conv.defaultTitle'))}</div>` +
      `<div class="ci-meta">${busy ? t('status.running') : t('status.idle')}</div></div>` +
      `<button class="ci-more" title="${escapeHtml(t('common.more'))}"><svg viewBox="0 0 24 24" fill="currentColor" stroke="none"><circle cx="12" cy="5" r="1.7"/><circle cx="12" cy="12" r="1.7"/><circle cx="12" cy="19" r="1.7"/></svg></button>`;
    convListEl.appendChild(item);
  }
}
if (searchInput) searchInput.addEventListener('input', () => renderSessionList());
async function ensureBridgeSession(sess) {
  if (sess.bridgeSessionId) return sess.bridgeSessionId;
  const res = await window.ga.rpc('session/new', { cwd: '', mcp_servers: [] });
  if (res?.error) throw new Error(res.error.message || res.error);
  sess.bridgeSessionId = res.sessionId || res.result?.sessionId;
  return sess.bridgeSessionId;
}
async function newSession() {
  const localId = 'local-' + Date.now() + '-' + Math.random().toString(16).slice(2);
  const sess = { id: localId, bridgeSessionId: null, title: t('conv.defaultTitle'), messages: [], untitled: true };
  state.sessions.set(localId, sess);
  try { await ensureBridgeSession(sess); } catch (e) { showError(t('err.newSession') + ': ' + (e.message || e)); }
  setActiveSession(localId);
  renderSessionList();
}
function setActiveSession(id) {
  state.activeId = id;
  const sess = state.sessions.get(id);
  if (!sess) return;
  if (msgsEl) msgsEl.innerHTML = '';
  rt(sess).draftEl = null;
  renderAllMessages(sess);
  setBusy(sess, rt(sess).busy);
  renderSessionList();
}
async function closeSession(id) {
  const sess = state.sessions.get(id);
  if (sess && sess.bridgeSessionId) {
    try { await window.ga.rpc('session/cancel', { sessionId: sess.bridgeSessionId }); } catch (_) {}
    fetch(`http://${location.hostname}:14168/session/${sess.bridgeSessionId}`, { method: 'DELETE' }).catch(() => {});
  }
  state.sessions.delete(id); state.runtime.delete(id);
  if (state.activeId === id) {
    const next = state.sessions.keys().next().value || null;
    if (next) setActiveSession(next);
    else { state.activeId = null; if (msgsEl) msgsEl.innerHTML = ''; refreshEmptyState(null); refreshStatusLabel(); }
  }
  renderSessionList();
}

const convMenu = document.getElementById('conv-menu');
let menuTargetId = null;
convListEl.addEventListener('click', (e) => {
  const more = e.target.closest('.ci-more');
  if (more) {
    e.stopPropagation();
    menuTargetId = more.closest('.conv-item').dataset.id;
    convMenu.hidden = false;
    const rect = more.getBoundingClientRect();
    convMenu.style.top = (rect.bottom + 4) + 'px';
    convMenu.style.left = (rect.right - convMenu.offsetWidth) + 'px';
    return;
  }
  const it = e.target.closest('.conv-item');
  if (it && it.dataset.id) setActiveSession(it.dataset.id);
});
convMenu.addEventListener('click', (e) => {
  e.stopPropagation();
  const act = e.target.closest('.ctx-item')?.dataset.act;
  const sess = menuTargetId && state.sessions.get(menuTargetId);
  if (sess && act === 'pin') {
    const m = new Map(); m.set(sess.id, sess);
    for (const [k, v] of state.sessions) if (k !== sess.id) m.set(k, v);
    state.sessions = m; renderSessionList();
  } else if (sess && act === 'del') {
    closeSession(sess.id);
  }
  convMenu.hidden = true;
});
document.addEventListener('click', () => { convMenu.hidden = true; });
newConvBtn.addEventListener('click', (e) => { e.preventDefault(); newSession(); });

/* ═══════════════ 轮询 + 流式 ═══════════════ */
function normalize(m) { return { id: Number(m.id || 0), role: m.role || 'system', content: m.content || '' }; }
function upsert(sess, raw, partial) {
  const m = normalize(raw); const r = rt(sess);
  if (partial && m.role === 'assistant') { r.draftText = m.content; if (isActive(sess)) renderDraft(sess); return; }
  if (!m.id || r.seen.has(m.id)) return;
  r.seen.add(m.id); r.lastId = Math.max(r.lastId, m.id);
  if (m.role === 'assistant' && r.draftEl) { r.draftEl.remove(); r.draftEl = null; r.draftText = ''; }
  sess.messages.push(m); appendMessage(sess, m);
}
async function pollSession(sess) {
  const r = rt(sess);
  if (r.polling) return;
  r.polling = true;
  try {
    do {
      const res = await window.ga.pollSession(sess.bridgeSessionId || sess.id, r.lastId || 0);
      if (res?.error) throw new Error(res.error.message || res.error);
      const result = res.result || res;
      for (const msg of (result.messages || [])) upsert(sess, msg, false);
      if (result.partial) upsert(sess, result.partial, true);
      const busy = result.status === 'running' || !!result.partial;
      setBusy(sess, busy);
      if (busy) await new Promise(z => setTimeout(z, 500));
      else { if (r.draftEl) { r.draftEl.remove(); r.draftEl = null; } break; }
    } while (true);
  } catch (e) {
    showError(t('err.poll') + ': ' + (e.message || e));
    setBusy(sess, false);
  } finally {
    r.polling = false; renderSessionList();
  }
}

/* ═══════════════ 发送 / 取消 ═══════════════ */
async function sendPrompt(text) {
  text = String(text || '').trim();
  if (!text) return;
  if (!state.bridgeReady) { showError(t('err.bridge')); return; }
  if (!state.activeId) { await newSession(); if (!state.activeId) return; }
  const sess = activeSess(); const r = rt(sess);
  if (r.busy) return;
  const userMsg = { role: 'user', content: text };
  sess.messages.push(userMsg); appendMessage(sess, userMsg);
  if (sess.untitled || isUntitled(sess.title)) {
    sess.title = text.slice(0, 40) + (text.length > 40 ? '…' : '');
    sess.untitled = false; renderSessionList();
  }
  setBusy(sess, true);
  try {
    const sid = await ensureBridgeSession(sess);
    const res = await window.ga.rpc('session/prompt', { sessionId: sid, prompt: text, images: [], llmNo: state.llmNo });
    if (res?.error) throw new Error(res.error.message || res.error);
    const uid = Number(res.userMessageId || res.result?.userMessageId || 0);
    if (uid) { r.seen.add(uid); r.lastId = Math.max(r.lastId, uid); }
    pollSession(sess);
  } catch (e) {
    const em = { role: 'error', content: e.message || String(e) };
    sess.messages.push(em); appendMessage(sess, em);
    setBusy(sess, false);
  }
}
async function cancelPrompt() {
  const sess = activeSess();
  if (!sess || !rt(sess).busy) return false;
  try {
    const res = await window.ga.rpc('session/cancel', { sessionId: sess.bridgeSessionId || sess.id });
    if (res?.error) throw new Error(res.error.message || res.error);
    return true;
  } catch (e) { showError(t('err.stop') + ': ' + (e.message || e)); return false; }
}

/* ═══════════════ 输入区 / slash / 预设 ═══════════════ */
function submitInput() {
  const text = inputEl.value;
  if (!text.trim()) return;
  inputEl.value = ''; inputEl.style.height = 'auto';
  if (text.trim().startsWith('/')) { handleSlash(text.trim()); return; }
  sendPrompt(text);
}
sendBtn.addEventListener('click', (e) => { e.preventDefault(); submitInput(); });
inputEl.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submitInput(); } });
inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 200) + 'px';
});
function showSystem(text) {
  const sess = activeSess(); if (!sess) return;
  const m = { role: 'system', content: text };
  sess.messages.push(m); appendMessage(sess, m);
}
function showError(text) {
  const sess = activeSess();
  if (sess) { const m = { role: 'error', content: text }; sess.messages.push(m); appendMessage(sess, m); }
  else console.error(text);
}
async function handleSlash(cmd) {
  const name = cmd.slice(1).split(/\s+/)[0];
  switch (name) {
    case 'help': showSystem(t('slash.help')); break;
    case 'new': await newSession(); break;
    case 'clear': { const s = activeSess(); if (s) { s.messages = []; renderAllMessages(s); } break; }
    case 'stop': if (await cancelPrompt()) showSystem(t('sys.stopRequested')); break;
    case 'settings': openModal('settings-modal'); break;
    default: showSystem(t('slash.unknown') + ': /' + name);
  }
}
// 预设卡：按 data-preset 解耦（与翻译后的标题无关）
document.querySelectorAll('.fcard').forEach(card => {
  card.addEventListener('click', () => {
    const key = card.dataset.preset;
    if (!key || key === 'add') { inputEl.focus(); closeModals(); return; }
    const prompt = I18N[lang]['presetPrompt.' + key] || I18N.zh['presetPrompt.' + key];
    closeModals();
    if (prompt) sendPrompt(prompt);
  });
});

/* ═══════════════ 模型档位 ═══════════════ */
function updateModelChip() {
  if (modelNameEl) modelNameEl.textContent = state.modelName || t('model.auto');
}
async function loadModelProfiles() {
  try {
    const res = await window.ga.getModelProfiles();
    const list = res?.profiles || res?.result?.profiles || [];
    state.modelProfiles = list;
    const active = list.find(p => p.active) || list[0];
    if (active) { state.llmNo = active.id ?? 0; state.modelName = active.name || null; }
    updateModelChip();
  } catch (_) {}
}
if (modelChip) modelChip.addEventListener('click', (e) => {
  e.preventDefault();
  const list = state.modelProfiles || [];
  if (!list.length) return;
  const idx = list.findIndex(p => (p.id ?? 0) === state.llmNo);
  const next = list[(idx + 1) % list.length];
  state.llmNo = next.id ?? 0; state.modelName = next.name || null;
  updateModelChip();
});

/* ═══════════════ 上传按钮（占位）═══════════════ */
const uploadBtn = chatPage.querySelector('.composer-top .ic-btn');
if (uploadBtn) uploadBtn.addEventListener('click', (e) => { e.preventDefault(); showSystem(t('upload.hint')); });

/* ═══════════════ 语言切换 ═══════════════ */
if (langSel) {
  langSel.value = lang;
  langSel.addEventListener('change', () => {
    lang = (langSel.value === 'en') ? 'en' : 'zh';
    localStorage.setItem('ga_lang', lang);
    applyI18n();
    renderSessionList();
    refreshStatusLabel();
    updateModelChip();
    if (document.querySelector('.page[data-page="channels"].active')) renderChannelList(gaServiceStore.list());
  });
}

/* ═══════════════ bridge 事件 ═══════════════ */
window.ga.onBridgeReady(() => {
  state.bridgeReady = true;
  if (!state.activeId) { refreshStatusLabel(); refreshEmptyState(null); }
  loadModelProfiles();
  if (document.querySelector('.page[data-page="channels"].active')) renderChannelList(gaServiceStore.list());
});
window.ga.onBridgeNotification((msg) => {
  if (msg && msg.type === 'session-state') {
    for (const sess of state.sessions.values()) {
      if (sess.bridgeSessionId === msg.sessionId) {
        if (msg.status === 'running' || msg.state === 'running') pollSession(sess);
        if (msg.state === 'idle' || msg.status === 'idle') tokPollBridge();
        renderSessionList();
        break;
      }
    }
  }
});
window.ga.onBridgeError((err) => { console.warn('[bridge error]', err); });
window.ga.onBridgeClosed(() => { state.bridgeReady = false; runLabel.textContent = t('status.disconnected'); });

/* ═══════════════ Token 统计页 ═══════════════ */
const tokTbody = document.getElementById('tok-tbody');
const tokPager = document.getElementById('tok-pager');
const tokSince = document.getElementById('tok-since');
const tokUntil = document.getElementById('tok-until');
const tokTotalN = document.getElementById('tok-total-n');
const tokTodayN = document.getElementById('tok-today-n');
const tokCostN = document.getElementById('tok-cost-n');
const TOK_PER_PAGE = 15;
let _tokPage = 0;
const TOK_STORE_KEY = 'ga_token_history';

// Model price table: $/M tokens [input, output]
const MODEL_PRICES = {
  'gpt-5.4':[2.50,15],'gpt-5':[1.25,10],'gpt-5-mini':[0.25,2],'gpt-4o':[2.50,10],'gpt-4o-mini':[0.15,0.60],
  'gpt-4.1':[2,8],'gpt-4.1-mini':[0.40,1.60],'gpt-4.1-nano':[0.10,0.40],'o4-mini':[1.10,4.40],
  'claude-opus-4-7':[5,25],'claude-opus-4-6':[5,25],'claude-sonnet-4-6':[3,15],'claude-sonnet-4-5':[3,15],'claude-haiku-4-5':[1,5],
  'deepseek-v4':[0.14,0.28],'deepseek-v4-pro':[0.55,2.19],'deepseek-chat':[0.14,0.28],'deepseek-reasoner':[0.55,2.19],
  'glm-5.1':[0.50,0.50],'minimax-m2.7':[0.50,0.50],'kimi-for-coding':[0.50,2],
};
const CNY_RATE = 7.2;
function estCost(inp, out, model, cacheRead, cacheCreate) {
  let p = [3,15];
  if (model) { const m = model.toLowerCase().replace(/\[.*\]/,''); p = MODEL_PRICES[m] || Object.entries(MODEL_PRICES).find(([k])=>m.includes(k))?.[1] || p; }
  const isClaudeOrDS = model && /claude|deepseek/i.test(model);
  const cacheReadRate = isClaudeOrDS ? 0.1 : 0.5;
  const cacheWriteRate = isClaudeOrDS ? 1.25 : 1.0;
  const cost = (inp*p[0] + out*p[1] + (cacheRead||0)*p[0]*cacheReadRate + (cacheCreate||0)*p[0]*cacheWriteRate) / 1e6 * CNY_RATE;
  return cost.toFixed(2);
}
function fmtTok(n) { return n>=1e6?(n/1e6).toFixed(2)+'M':n>=1e3?(n/1e3).toFixed(1)+'k':String(n); }
function fmtTime(ts) { return new Date(ts*1000).toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}); }

function tokLoadHistory() { try { return JSON.parse(localStorage.getItem(TOK_STORE_KEY)||'[]'); } catch(_) { return []; } }
function tokSaveHistory(h) { localStorage.setItem(TOK_STORE_KEY, JSON.stringify(h)); }

// Poll bridge and snapshot deltas into localStorage
const TOK_SNAP_KEY = 'ga_token_snap';
let _tokLastSnap = (() => { try { return JSON.parse(localStorage.getItem(TOK_SNAP_KEY)||'{}'); } catch(_) { return {}; } })();
let _tokPolling = false;
async function tokPollBridge() {
  if (_tokPolling) return;
  _tokPolling = true;
  try {
    const res = await fetch(`http://${location.hostname}:14168/token-stats`);
    const data = await res.json();
    const history = tokLoadHistory();
    for (const r of (data.records||[])) {
      const key = r.thread;
      const prev = _tokLastSnap[key] || {input:0,output:0,cacheCreate:0,cacheRead:0};
      const di = r.input-prev.input, do_ = r.output-prev.output, dc = r.cacheCreate-prev.cacheCreate, dr = r.cacheRead-prev.cacheRead;
      if (di>0||do_>0||dc>0||dr>0) {
        const sid = key.replace('GA-','');
        const sess = [...state.sessions.values()].find(s=>s.bridgeSessionId===sid);
        const title = sess?.title||sid;
        history.push({sessionId:sid, title:title, input:di, output:do_, cacheCreate:dc, cacheRead:dr, model:r.model||'', ts:Date.now()/1000});
        if(sess?.title) history.forEach(h=>{if(h.sessionId===sid&&(!h.title||h.title===sid))h.title=sess.title;});
      }
      _tokLastSnap[key] = {input:r.input, output:r.output, cacheCreate:r.cacheCreate, cacheRead:r.cacheRead};
    }
    localStorage.setItem(TOK_SNAP_KEY, JSON.stringify(_tokLastSnap));
    tokSaveHistory(history);
  } catch(_) {}
  _tokPolling = false;
}

function tokGetFiltered() {
  let records = tokLoadHistory();
  const since = tokSince?.value ? new Date(tokSince.value).getTime()/1000 : 0;
  const until = tokUntil?.value ? new Date(tokUntil.value).getTime()/1000 : 0;
  if (since) records = records.filter(r=>r.ts>=since);
  if (until) records = records.filter(r=>r.ts<=until);
  return records;
}

function tokRenderStats(filtered, all) {
  let total=0, cost=0;
  filtered.forEach(r=>{total+=(r.input||0)+(r.output||0); cost+=parseFloat(estCost(r.input||0,r.output||0,r.model,r.cacheRead||0,r.cacheCreate||0));});
  if(tokTotalN) tokTotalN.textContent=fmtTok(total);
  if(tokCostN) tokCostN.textContent='¥ '+cost.toFixed(1);
  const todayStart=new Date(); todayStart.setHours(0,0,0,0); const todayTs=todayStart.getTime()/1000;
  let todayT=0; all.filter(r=>r.ts>=todayTs).forEach(r=>{todayT+=(r.input||0)+(r.output||0);});
  if(tokTodayN) tokTodayN.textContent=fmtTok(todayT);
}

function tokRenderTable(records) {
  if(!tokTbody) return;
  const bySession=new Map();
  for(const r of records){
    const k=r.sessionId||'?';
    let title = r.title||k;
    if(!title||title===k){ const ss=[...state.sessions.values()].find(s=>s.bridgeSessionId===k); if(ss)title=ss.title; }
    if(!bySession.has(k)) bySession.set(k,{title:title,input:0,output:0,cacheCreate:0,cacheRead:0,lastTs:0,prompts:[]});
    const s=bySession.get(k); s.input+=r.input||0; s.output+=r.output||0; s.cacheCreate+=r.cacheCreate||0; s.cacheRead+=r.cacheRead||0;
    if(r.ts>s.lastTs){s.lastTs=r.ts; s.title=r.title||s.title;} s.prompts.push(r);
  }
  tokTbody.innerHTML='';
  if(bySession.size===0){tokTbody.innerHTML=`<tr><td colspan="6" style="color:var(--muted)">${t('tok.noData')}</td></tr>`;if(tokPager)tokPager.innerHTML='';return;}
  const sorted=[...bySession.values()].sort((a,b)=>b.lastTs-a.lastTs);
  const totalPages=Math.ceil(sorted.length/TOK_PER_PAGE);
  if(_tokPage>=totalPages)_tokPage=totalPages-1;
  const pageItems=sorted.slice(_tokPage*TOK_PER_PAGE,(_tokPage+1)*TOK_PER_PAGE);
  for(const s of pageItems){
    let sc=0; s.prompts.forEach(p=>{sc+=parseFloat(estCost(p.input||0,p.output||0,p.model,p.cacheRead||0,p.cacheCreate||0));});
    const tr=document.createElement('tr'); tr.className='tok-row-session';
    tr.innerHTML=`<td>${escapeHtml(s.title)}</td><td>${fmtTok(s.input)}</td><td>${fmtTok(s.output)}</td><td>${fmtTok(s.cacheCreate)}</td><td>${fmtTok(s.cacheRead)}</td><td>¥${sc.toFixed(2)}</td>`;
    tokTbody.appendChild(tr);
    const details=[]; s.prompts.sort((a,b)=>b.ts-a.ts);
    for(const p of s.prompts){
      const dr=document.createElement('tr'); dr.className='tok-detail'; dr.hidden=true;
      dr.innerHTML=`<td>${fmtTime(p.ts)}${p.model?' · '+escapeHtml(p.model):''}</td><td>${fmtTok(p.input||0)}</td><td>${fmtTok(p.output||0)}</td><td>${fmtTok(p.cacheCreate||0)}</td><td>${fmtTok(p.cacheRead||0)}</td><td>¥${estCost(p.input||0,p.output||0,p.model,p.cacheRead||0,p.cacheCreate||0)}</td>`;
      tokTbody.appendChild(dr); details.push(dr);
    }
    tr.addEventListener('click',()=>{const o=tr.classList.toggle('open');details.forEach(d=>d.hidden=!o);});
  }
  if(tokPager){tokPager.innerHTML='';if(totalPages>1)for(let i=0;i<totalPages;i++){const b=document.createElement('button');b.textContent=i+1;if(i===_tokPage)b.className='active';b.addEventListener('click',()=>{_tokPage=i;tokRenderTable(records);});tokPager.appendChild(b);}}
}

async function loadTokenPage(){await tokPollBridge();const f=tokGetFiltered();const all=tokLoadHistory();tokRenderStats(f,all);tokRenderTable(f);}
if(tokSince)tokSince.addEventListener('change',()=>{_tokPage=0;loadTokenPage();});
if(tokUntil)tokUntil.addEventListener('change',()=>{_tokPage=0;loadTokenPage();});
const tokResetBtn=document.getElementById('tok-reset');
if(tokResetBtn)tokResetBtn.addEventListener('click',()=>{if(tokSince)tokSince.value='';if(tokUntil)tokUntil.value='';_tokPage=0;loadTokenPage();});
nav.addEventListener('click',(e)=>{const item=e.target.closest('.nav-item');if(item&&item.dataset.page==='token')loadTokenPage();if(item&&item.dataset.page==='channels')renderChannelList(gaServiceStore.list());if(item&&item.dataset.page==='status')loadStatusPanel();});

/* ═══════════════ 消息通道（复用 gaServiceStore + WS 同步） ═══════════════ */
const CHAN_ICON = '<svg class="lr-ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>';
const CHAN_FILE_LABELS = {
  'qqapp.py': 'ch.qq',
  'wechatapp.py': 'ch.wechat',
  'wecomapp.py': 'ch.wecom',
  'dingtalkapp.py': 'ch.dingtalk',
  'tgapp.py': 'ch.telegram',
  'dcapp.py': 'ch.discord',
  'fsapp.py': 'ch.lark',
};
const chanListEl = document.getElementById('chan-list');
const chanEmptyEl = document.getElementById('chan-empty');
const chanLogModal = document.getElementById('chan-log-modal');
const chanLogPre = document.getElementById('chan-log-pre');
const chanLogTitle = document.getElementById('chan-log-title');
let _chanLogId = null;
let _chanBusy = false;
let _chanToastTimer = null;

function getToastRoot() {
  let root = document.getElementById('toast-root');
  if (!root) {
    root = document.createElement('div');
    root.id = 'toast-root';
    root.className = 'toast-root';
    root.setAttribute('aria-live', 'polite');
    document.body.appendChild(root);
  }
  return root;
}

function showChanToast(title, detail, kind) {
  if (!title) return;
  const root = getToastRoot();
  if (_chanToastTimer) clearTimeout(_chanToastTimer);
  root.innerHTML = '';
  const el = document.createElement('div');
  el.className = `toast toast-${kind === 'ok' ? 'ok' : 'err'}`;
  const tEl = document.createElement('span');
  tEl.className = 'toast-title';
  tEl.textContent = title;
  el.appendChild(tEl);
  if (detail) {
    const dEl = document.createElement('span');
    dEl.className = 'toast-detail';
    dEl.textContent = detail;
    el.appendChild(dEl);
  }
  root.appendChild(el);
  const show = () => el.classList.add('show');
  requestAnimationFrame(show);
  setTimeout(show, 16);
  _chanToastTimer = setTimeout(() => {
    el.classList.remove('show');
    setTimeout(() => el.remove(), 300);
  }, 3000);
}

function channelDisplayName(ch) {
  const file = (ch.name || ch.id || '').split('/').pop();
  const key = CHAN_FILE_LABELS[file];
  return key ? t(key) : (ch.name || ch.id || '');
}
function channelStatusClass(status) {
  if (status === 'running') return 'on';
  if (status === 'error') return 'err';
  return 'off';
}
function channelStatusLabel(status) {
  const map = {
    running: 'st.running', offline: 'st.offline', error: 'st.error',
    starting: 'st.starting', stopping: 'st.stopping',
  };
  return t(map[status] || 'st.offline');
}
function channelErrorMessage(code) {
  const map = { not_configured: 'err.channelNotConfigured' };
  return t(map[code] || code || 'err.channelStart');
}
function channelToastDetail(e) {
  const svc = e.data && e.data.service;
  if (svc && svc.lastError) return svc.lastError;
  const code = e.data && e.data.error;
  return channelErrorMessage(code || e.message);
}
function renderChannelList(channels) {
  if (!chanListEl) return;
  const rows = (channels || []).filter((ch) => (ch.id || '').startsWith('frontends/'));
  chanListEl.innerHTML = '';
  if (chanEmptyEl) chanEmptyEl.hidden = rows.length > 0;
  for (const ch of rows) {
    const row = document.createElement('div');
    row.className = 'list-row';
    row.dataset.channelId = ch.id;
    const stClass = channelStatusClass(ch.status || 'offline');
    const running = !!ch.running;
    row.innerHTML = `
      ${CHAN_ICON}
      <div class="chan-meta">
        <b class="chan-name"></b>
        <span class="kv chan-path"></span>
      </div>
      <span class="lr-st ${stClass} chan-status"></span>
      <span class="grow"></span>
      <button type="button" class="link-btn link sm" data-act="configure"></button>
      <button type="button" class="link-btn link sm" data-act="logs"></button>
      <button type="button" class="sw-mini${running ? ' on' : ''}" data-act="toggle" aria-pressed="${running}"><i></i></button>`;
    row.querySelector('.chan-name').textContent = channelDisplayName(ch);
    row.querySelector('.chan-path').textContent = ch.name || ch.id;
    row.querySelector('.chan-status').textContent = channelStatusLabel(ch.status || 'offline');
    row.querySelector('[data-act="configure"]').textContent = t('act.configure');
    row.querySelector('[data-act="logs"]').textContent = t('act.logs');
    chanListEl.appendChild(row);
  }
}
async function toggleChannel(id, running, toggleEl) {
  if (_chanBusy) return;
  _chanBusy = true;
  if (toggleEl) toggleEl.disabled = true;
  const label = channelDisplayName(gaServiceStore.get(id) || { id });
  try {
    if (running) {
      await window.ga.stopService(id);
      showChanToast(t('sys.channelStopped') + ' · ' + label, '', 'ok');
    } else {
      const res = await window.ga.startService(id);
      if (res && res.service && res.service.status === 'error') {
        throw Object.assign(new Error(res.service.lastError || 'start_failed'), { data: res });
      }
      showChanToast(t('sys.channelStarted') + ' · ' + label, '', 'ok');
    }
  } catch (e) {
    showChanToast(
      (running ? t('err.channelStop') : t('err.channelStart')) + ' · ' + label,
      channelToastDetail(e),
      'err'
    );
  } finally {
    _chanBusy = false;
    if (toggleEl) toggleEl.disabled = false;
  }
}
async function openChannelLogs(id) {
  if (!chanLogModal || !chanLogPre) return;
  _chanLogId = id;
  const ch = gaServiceStore.get(id) || { id };
  const titleName = id === '__bridge__' ? (ch.name || 'bridge') : statusDisplayName(ch);
  if (chanLogTitle) chanLogTitle.textContent = t('modal.channelLogs') + ' · ' + titleName;
  chanLogPre.textContent = t('ch.loading');
  openModal('chan-log-modal');
  try {
    const res = await window.ga.getServiceLogs(id, 200);
    const lines = res.lines || [];
    chanLogPre.textContent = lines.length ? lines.join('\n') : t('ch.logEmpty');
  } catch (e) {
    chanLogPre.textContent = t('err.channelLoad') + ': ' + (e.message || e);
  }
}
async function openChannelMykey() {
  try {
    await window.ga.openMykey();
  } catch (e) {
    showChanToast(t('err.channelLoad'), e.message || String(e), 'err');
  }
}

/* ═══════════════ 状态面板（复用 ServiceManager + 启停/日志） ═══════════════ */
const statusListEl = document.getElementById('status-list');

function statusDisplayName(s) {
  if (!s) return '';
  if (s.id === '__bridge__') return s.name || 'bridge';
  if (s.id === 'reflect/scheduler.py') return t('proc.scheduler');
  return channelDisplayName(s);
}
function fmtPid(pid) { return pid ? `PID ${pid}` : '—'; }
function fmtRes(s) {
  const cpu = s.cpuPct != null ? `${s.cpuPct}%` : '—';
  const mem = s.memMb != null ? `${s.memMb}MB` : '—';
  return `${cpu} / ${mem}`;
}

function renderStatusPanel(services) {
  if (!statusListEl) return;
  statusListEl.innerHTML = '';
  for (const s of services || []) {
    const row = document.createElement('div');
    row.className = 'list-row';
    row.dataset.serviceId = s.id;
    const stClass = channelStatusClass(s.status || 'offline');
    const running = !!s.running;
    const managed = s.managed !== false;
    let acts = `<button type="button" class="link-btn link sm" data-act="logs"></button>`;
    if (managed) {
      if (running) acts += `<button type="button" class="link-btn link sm" data-act="restart"></button>`;
      acts += `<button type="button" class="sw-mini${running ? ' on' : ''}" data-act="toggle" aria-pressed="${running}"><i></i></button>`;
    }
    row.innerHTML = `
      <b class="st-name"></b>
      <span class="lr-st ${stClass} st-status"></span>
      <span class="kv st-pid"></span>
      <span class="kv st-res"></span>
      <span class="grow"></span>
      ${acts}`;
    row.querySelector('.st-name').textContent = statusDisplayName(s);
    row.querySelector('.st-status').textContent = channelStatusLabel(s.status || 'offline');
    row.querySelector('.st-pid').textContent = fmtPid(s.pid);
    row.querySelector('.st-res').textContent = fmtRes(s);
    const logBtn = row.querySelector('[data-act="logs"]');
    if (logBtn) logBtn.textContent = t('act.logs');
    const rstBtn = row.querySelector('[data-act="restart"]');
    if (rstBtn) rstBtn.textContent = t('act.restart');
    statusListEl.appendChild(row);
  }
}

async function loadStatusPanel() {
  if (!statusListEl) return;
  const res = await window.ga.getServicePanel();
  renderStatusPanel(res.services || []);
}

async function restartService(id) {
  const label = statusDisplayName(gaServiceStore.get(id) || { id });
  await window.ga.stopService(id);
  const res = await window.ga.startService(id);
  if (res && res.service && res.service.status === 'error') {
    throw Object.assign(new Error(res.service.lastError || 'start_failed'), { data: res });
  }
  showChanToast(t('act.restart') + ' · ' + label, '', 'ok');
}

if (statusListEl) {
  statusListEl.addEventListener('click', async (e) => {
    const row = e.target.closest('.list-row');
    if (!row) return;
    const id = row.dataset.serviceId;
    const actEl = e.target.closest('[data-act]');
    if (!actEl || !id) return;
    const act = actEl.dataset.act;
    if (act === 'logs') {
      openChannelLogs(id);
      return;
    }
    if (act === 'restart') {
      if (_chanBusy) return;
      _chanBusy = true;
      try {
        await restartService(id);
        await loadStatusPanel();
      } catch (err) {
        showChanToast(t('act.restart') + ' · ' + statusDisplayName({ id }), err.message || String(err), 'err');
      } finally {
        _chanBusy = false;
      }
      return;
    }
    if (act === 'toggle') {
      if (actEl.disabled || _chanBusy) return;
      const running = actEl.classList.contains('on');
      await toggleChannel(id, running, actEl);
      if (document.querySelector('.page[data-page="status"].active')) loadStatusPanel();
    }
  });
}

gaServiceStore.onServices((list) => {
  if (document.querySelector('.page[data-page="channels"].active')) renderChannelList(list);
  if (document.querySelector('.page[data-page="status"].active')) loadStatusPanel();
});
if (chanListEl) {
  chanListEl.addEventListener('click', async (e) => {
    const row = e.target.closest('.list-row');
    if (!row) return;
    const id = row.dataset.channelId;
    const actEl = e.target.closest('[data-act]');
    if (!actEl || !id) return;
    const act = actEl.dataset.act;
    if (act === 'logs') {
      openChannelLogs(id);
      return;
    }
    if (act === 'configure') {
      openChannelMykey();
      return;
    }
    if (act === 'toggle') {
      if (actEl.disabled || _chanBusy) return;
      const running = actEl.classList.contains('on');
      await toggleChannel(id, running, actEl);
    }
  });
}

/* ═══════════════ 启动 ═══════════════ */
applyI18n();
updateModelChip();
renderSessionList();
refreshEmptyState(null);
runLabel.textContent = t('status.connecting');
window.ga.startBridge && window.ga.startBridge();
