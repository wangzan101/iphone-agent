/* iPhone Agent —— 界面逻辑。
 *
 * 没有构建步骤、没有框架。运行环境被锁死在 macOS 15+ 自带的浏览器上
 * （iPhone 镜像的下限），转译/打包/polyfill 这三件事一件都不需要 —— 而它们
 * 正是前端工具链存在的全部理由（docs/22 D3）。
 *
 * ⚠ 所有插进 DOM 的文本一律走 textContent，绝不 innerHTML。
 *   这里显示的东西有一半来自**模型的输出**和**手机屏幕上的文字** —— 那是
 *   不可信输入。用 innerHTML 等于把屏幕内容当代码执行。
 */
'use strict';

I18n.init();
const apiMessage = (data, field) => I18n.message(data[field + '_message'], data[field] || '');
const eventMessage = data => I18n.message(data.message, data.text || '');


const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const el = (cls, txt) => { const d = document.createElement('div'); d.className = cls;
                           if (txt !== undefined) I18n.bind(d, "textContent", txt); return d; };
const svg = (path, w = 2.4) =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="${w}"
        stroke-linecap="round" stroke-linejoin="round">${path}</svg>`;
const ICON = {
  ok: svg('<path d="M20 6 9 17l-5-5"/>', 3.4),
  x: svg('<path d="M18 6 6 18M6 6l12 12"/>', 3),
  warn: svg('<path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>', 1.8),
  info: svg('<circle cx="12" cy="12" r="9"/><path d="M12 16v-4M12 8h.01"/>', 1.7),
  lock: svg('<rect x="3" y="11" width="18" height="10" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>', 1.7),
  send: svg('<path d="M12 19V5M5 12l7-7 7 7"/>'),
  stop: svg('<rect x="7" y="7" width="10" height="10" rx="2"/>', 2),
  pause: svg('<path d="M9 5v14M15 5v14"/>', 2.4),
  chev: svg('<path d="m9 18 6-6-6-6"/>', 2.4),
  play: svg('<path d="M7 4.5v15l12-7.5z" fill="currentColor" stroke="none"/>', 0),
};

const api = {
  async get(p) {
    const r = await fetch(p), data = await r.json();
    if (!r.ok || data.error) {
      const error = new Error(data.error || 'HTTP ' + r.status);
      error.uiMessage = apiMessage(data, 'error') || error.message;
      throw error;
    }
    return data;
  },
  async post(p, body) {
    const r = await fetch(p, { method: 'POST', body: body === undefined ? '' : JSON.stringify(body) });
    return r.json();
  },
};

/* ══════════════════════════════════════════════════════════════════════
   1. 动作 → 人话
   只做机械直译，绝不推断意图（意图是 reason 的活，那是模型自己写的）。
   降级链：有目标文字 → 有 id → 有坐标 → 退回工具名。**最坏情况等于什么都不做。**
   876 个历史动作实测：真正落到 fallback 的 4 个，且全是模型幻觉出来的不存在工具。
   ══════════════════════════════════════════════════════════════════════ */
const Q = x => uiCat(uiCat('「', x), '」');
const cut = (x, k) => { x = String(x == null ? '' : x); return x.length > k ? uiCat(x.slice(0, k), '…') : x; };
const DIR = { up: L('dir.up'), down: L('dir.down'), left: L('dir.left'), right: L('dir.right') };

function humanize(s) {
  const a = s.args || {}, t = s.target_text;
  switch (s.name) {
    case 'tap': {
      if (t) {
        const target = Q(cut(t, 14));
        const key = { icon_above: 'action.tapIcon', row_right: 'action.tapRight', row_left: 'action.tapLeft' }[a.target];
        return L(key || 'action.tap', { target });
      }
      if (a.id !== undefined) return L('action.tapId', { id: a.id });
      if (a.x !== undefined) return L('action.tapCoord');
      return null;
    }
    case 'open_app': return a.name ? L('action.openApp', { app: Q(cut(a.name, 14)) }) : null;
    case 'scroll': {
      const direction = DIR[a.direction]; if (!direction) return null;
      if (a.direction === 'left' || a.direction === 'right') return L('action.page', { direction });
      return L('action.scroll', { direction, amount: L(a.amount === 'half' ? 'amount.half' : 'amount.full') });
    }
    case 'scroll_until': {
      const direction = DIR[a.direction]; if (!direction || !a.text) return null;
      return L('action.until', { direction, text: Q(cut(a.text, 12)) });
    }
    case 'collect': return L('action.collect');
    case 'type': return a.text ? L('action.type', { text: Q(cut(a.text, 16)) }) : null;
    case 'key': return { home: L('action.home'), app_switcher: L('action.switcher'), spotlight: L('action.spotlight') }[a.name] || null;
    case 'switch_ime': return L('action.ime');
    case 'wait': return a.seconds !== undefined ? L('action.wait', { seconds: a.seconds }) : null;
    case 'observe': return L('action.observe');
    case 'recall': return a.name ? L('action.recall', { name: Q(cut(a.name, 20)) }) : null;
    case 'recall_runs': return L('action.recallRuns');
    case 'done': return L(a.status === 'success' ? 'action.done' : 'action.failed');
    case 'start': return L('action.start');           // 回放补的第一帧：发任务时的画面（api.py run_detail）
    default: return null;
  }
}

/* 连续同类动作折叠成一条。只折叠本来就无副作用、且重复才有意义的那几个，
   **出错的一律不折叠** —— 压缩冗余，放大异常。 */
const FOLDABLE = new Set(['scroll', 'wait', 'observe']);
const gkey = s => (!FOLDABLE.has(s.name) || s.error) ? null
                : uiCat(uiCat(s.name, '|') + ((s.args || {}).direction || ''), '|') + ((s.args || {}).amount || '');
function groupLabel(items) {
  const s = items[0], a = s.args || {}, count = items.length;
  if (s.name === 'scroll') return L(
    a.direction === 'left' || a.direction === 'right' ? 'group.pages' : 'group.screens',
    { direction: DIR[a.direction], count });
  if (s.name === 'wait') return L('group.wait', { count });
  if (s.name === 'observe') return L('group.observe', { count });
  return uiCat(humanize(s) || s.name, ' ×', count);
}
/* 三态：出错 / 有变化 / 没变化。「没变化」不是错，但必须看得见。 */
const cls = s => s.error ? 'bad' : (s.changed === true ? 'ok' : 'nc');

/* ══════════════════════════════════════════════════════════════════════
   2. 对话
   ══════════════════════════════════════════════════════════════════════ */
const log = $('#log'), shot = $('#shot'), hud = $('#hud'), q = $('#q'), send = $('#send'), pauseBtn = $('#pause');
let paused = false;
let turn = null, tl = null, lastG = null, running = false;

function bottom() { log.scrollTop = log.scrollHeight; }

function showEmpty() {
  I18n.bind(log, "textContent", '');
  const box = el('');
  box.id = 'empty';
  const h = document.createElement('h2'); I18n.bind(h, "textContent", L("chat.prompt"));
  const p = document.createElement('p'); I18n.bind(p, "textContent", L("chat.intro"));
  box.append(h, p);
  const cards = el('cards');
  for (const [t, s] of [
    [L("example.version"), L("example.versionPath")],
    [L("example.battery"), L("example.batteryPath")],
    [L("example.screen"), L("example.screenNote")],
    [L("example.storage"), L("example.storagePath")],
  ]) {
    const b = document.createElement('button');
    b.className = 'card';
    const bb = document.createElement('b'); I18n.bind(bb, "textContent", t);
    const ss = document.createElement('span'); I18n.bind(ss, "textContent", s);
    b.append(bb, ss);
    b.onclick = () => { q.value = t; ask(); };
    cards.appendChild(b);
  }
  box.appendChild(cards);
  log.appendChild(box);
}

/* ⚠ 同一份内容，在两个时刻有两种职责：
     运行中 —— 过程就是内容（答案还不存在），所以只显示**当前这一步**；
     跑完后 —— 答案是内容，过程贬值成「出问题时才查的证据」，收成一行。
   三条状态规则：
     ① 展开是**用户意图**，存在 turn 上，任何一次重绘都不能把它抹掉；
     ② 展开的列表少的时候自然矮、多了到上限就停（max-height，不是死高度）；
     ③ 贴底才自动跟随；用户一往上翻就让位，给「↓ 回到最新」。
*/
function startTurn(text) {
  if ($('#empty')) I18n.bind(log, "textContent", '');
  lastG = null;
  turn = el('turn');
  turn.appendChild(el('ask', text));

  const box = el('live');
  const track = el('track');
  const head = el('livehead');
  const n = el('n'), dur = el(''), sp = el('sp');
  head.append(n, sp, dur);
  const now = el('livenow', L("state.starting"));
  const why = el('livewhy'), warn = el('livewarn');
  const foot = el('livefoot');
  const more = document.createElement('button');
  more.className = 'mini';
  I18n.bind(more, "textContent", L("steps.expand"));
  foot.append(el('sp'), more);
  box.append(track, head, now, why, warn, foot);

  // 过程是**一块**：横条（跑完才有）是它的头，列表接在下面，中间不留缝。
  const proc = el('proc');
  const listwrap = el('listwrap');
  listwrap.hidden = true;
  tl = el('tl capped');
  const jump = document.createElement('button');
  jump.className = 'jump';
  I18n.bind(jump, "textContent", L("steps.latest"));
  jump.hidden = true;
  listwrap.append(tl, jump);
  proc.appendChild(listwrap);
  proc.hidden = true;          // 运行中还没有横条、列表也收着 —— 别显示成一个空框
  // 回复和过程条装在同一个容器里，宽度才会一致（它俩本来就是一个整体）
  const reply = el('reply');
  reply.appendChild(proc);

  turn.append(box, reply);
  Object.assign(turn, {
    _live: box, _track: track, _n: n, _dur: dur, _now: now, _why: why, _warn: warn,
    _more: more, _list: tl, _wrap: listwrap, _proc: proc, _reply: reply, _jump: jump,
    _steps: [], _run: currentRun,
    _expanded: false,        // ① 用户意图
    _stick: true,            // ③ 还贴着底吗
  });

  // ⚠ 处理器里必须捕获**这一轮**，不能引用模块级的 turn ——
  //   下一轮开始之后，旧回合的滚动会跑去改新回合的状态。
  const t = turn;
  // ③ 用户一往上翻就停止跟随，别跟他抢滚动条 —— 否则想回头看第 3 步，
  //    每来一步就被拽回底部一次，比不自动滚还难受。
  tl.onscroll = () => {
    t._stick = tl.scrollHeight - tl.scrollTop - tl.clientHeight < 24;
    jump.hidden = t._stick;
  };
  jump.onclick = () => { t._stick = true; jump.hidden = true; tl.scrollTop = tl.scrollHeight; };
  more.onclick = () => setExpanded(t, !t._expanded);

  log.appendChild(turn);
  bottom();
}

function setExpanded(t, on) {
  t._expanded = on;
  t._wrap.hidden = !on;
  // 跑完之后横条一直在，所以 proc 只在「有横条 或 列表展开着」时才该露面
  if (t._proc) t._proc.hidden = !(on || t._sum);
  if (t._more) I18n.bind(t._more, "textContent", on ? L("steps.collapse") : L("steps.expand"));
  if (t._go) I18n.bind(t._go, "textContent", on ? L("process.hide") : L("process.show"));
  if (t._sum) t._sum.classList.toggle('open', on);
  if (on) { t._stick = true; t._jump.hidden = true; t._list.scrollTop = t._list.scrollHeight; }
}

function subRow(s) {
  const r = el('subrow');
  r.appendChild(el('n', uiCat('#', s.n)));
  const b = document.createElement('b'); I18n.bind(b, "textContent", humanize(s) || s.name);
  r.append(b, el('dur', frameHint(s)));
  return r;
}
const frameHint = s => s.frame ? String(s.frame).replace(/\.png$/, '') : '';

function addStep(s) {
  if (!turn) startTurn(L("chat.continued"));
  const k = gkey(s);
  // ① 能并进上一组就并进去，不新起一行
  if (k && lastG && lastG.k === k && lastG.items.length < 12) {
    lastG.items.push(s);
    I18n.bind(lastG.say, "textContent", groupLabel(lastG.items));
    lastG.sub.appendChild(subRow(s));
    if (lastG.items.length === 2) lastG.head.insertBefore(el('chev', '▸'), lastG.head.children[1]);
    markCurrent(lastG.node);
    bottom();
    return;
  }
  const c = cls(s);
  const row = el(uiCat('step ', c));
  const head = el('srow-h');
  const ic = el('ic');
  ic.innerHTML = c === 'bad' ? ICON.x : c === 'ok' ? ICON.ok : '';   // 常量图标，非用户数据
  const say = el('say', humanize(s) || s.name);                       // 翻不出来就退回工具名
  head.append(ic, say, el('tool', s.name || ''), el('dur', frameHint(s)));
  row.appendChild(head);
  if (s.reason) row.appendChild(el('why', s.reason));
  if (s.error) {
    const e = el('err');
    const b = document.createElement('b'); I18n.bind(b, "textContent", s.error);
    const sp = document.createElement('span'); I18n.bind(sp, "textContent", s.hint ? L('diagnostic.raw', { text: s.hint }) : '');
    e.append(b, sp);
    row.appendChild(e);
  } else if (s.changed === false && s.hint) {
    row.appendChild(el('why warn', s.hint));
  }
  row.appendChild(el('args', JSON.stringify(s.args || {}, null, 1)));
  const sub = el('sub'); row.appendChild(sub);
  row.onclick = () => {
    if (row.querySelector('.subrow')) row.classList.toggle('open');
    if (s.frame) showFrame(s.frame, s);
  };
  I18n.bind($('#railtxt'), "textContent", L('step.number', { count: s.n ?? '?' }));
  lastG = k ? { k, items: [s], say, head, sub, node: row } : null;
  tl.appendChild(row);
  markCurrent(row);
  bottom();
}

/* 运行中那一块：进度轨 + 当前这一步。⚠ 只更新，不重建 ——
   重建会把用户「展开」的选择和滚动位置一起抹掉。 */
function updateLive(t, s) {
  const cell = document.createElement('i');
  cell.className = cls(s);
  // 上一格从「正在做」变成它的实际结果
  const prev = t._track.lastElementChild;
  if (prev && prev.classList.contains('cur')) prev.className = prev.dataset.c || 'ok';
  cell.classList.add('cur');
  cell.dataset.c = cls(s);
  t._track.appendChild(cell);

  I18n.bind(t._n, "textContent", L('step.number', { count: s.n ?? '?' }));
  const ms = (s.exec_ms || 0) + (s.latency_ms || 0);
  I18n.bind(t._dur, "textContent", ms ? uiCat((ms / 1000).toFixed(1), 's') : '');
  I18n.bind(t._now, "textContent", humanize(s) || s.name || '');
  I18n.bind(t._why, "textContent", s.reason || '');
  I18n.bind(t._warn, "textContent", s.error ? (uiCat(uiCat('⚠ ', s.error), '：') + (s.hint || ''))
    : (s.changed === false && s.hint ? (uiCat('⚠ ', s.hint)) : ''));
}

function markCurrent(node) {
  $$('.step.cur').forEach(x => x.classList.remove('cur'));
  node.classList.add('cur');
  // ③ 贴底才跟随
  if (turn && turn._list && turn._stick) turn._list.scrollTop = turn._list.scrollHeight;
}

/* ══════════════════════════════════════════════════════════════════════
   富文本：模型的回复里带 markdown
   ══════════════════════════════════════════════════════════════════════
   实测：真实回复长这样 ——
     1. **22万赞** - "20分钟学会Codex！零基础终极教程"
   原来用 createTextNode 整段塞进去，星号原样显示、列表没有缩进。
   任务越复杂、回复越长，越难看。

   ⚠ **绝不用 innerHTML。** 回复里混着手机屏幕上的 OCR 文字，是不可信输入。
     所以这里自己解析一个最小子集（粗体、行内代码、有序/无序列表、段落），
     自己 createElement —— 好看和安全不用二选一。
   ⚠ 只认这几样。认不出来的一律当普通文字，**最坏情况等于什么都没做**。
*/
const INLINE_RE = /\*\*([^*\n]+)\*\*|`([^`\n]+)`/g;

function inlineInto(parent, text) {
  INLINE_RE.lastIndex = 0;
  let last = 0, m;
  while ((m = INLINE_RE.exec(text)) !== null) {
    if (m.index > last) parent.appendChild(uiNode(text.slice(last, m.index)));
    if (m[1] !== undefined) {
      const b = document.createElement('strong'); I18n.bind(b, "textContent", m[1]); parent.appendChild(b);
    } else {
      const c = document.createElement('code'); I18n.bind(c, "textContent", m[2]); parent.appendChild(c);
    }
    last = INLINE_RE.lastIndex;
  }
  if (last < text.length) parent.appendChild(uiNode(text.slice(last)));
}

function renderRich(host, text) {
  if (I18n.isMessage(text)) { uiText(host, text); return; }
  const lines = String(text == null ? '' : text).split('\n');
  let list = null, listTag = null, para = null;
  for (const raw of lines) {
    const line = raw.trim();
    if (!line) { para = null; list = null; listTag = null; continue; }   // 空行 = 分段
    const ol = line.match(/^(\d+)[.)]\s+(.+)$/);
    const ul = line.match(/^[-*•]\s+(.+)$/);
    if (ol || ul) {
      para = null;
      const want = ol ? 'ol' : 'ul';
      if (!list || listTag !== want) {
        list = document.createElement(want); list.className = 'md'; listTag = want;
        host.appendChild(list);
      }
      const li = document.createElement('li');
      inlineInto(li, ol ? ol[2] : ul[1]);
      list.appendChild(li);
      continue;
    }
    list = null; listTag = null;
    if (!para) { para = document.createElement('p'); para.className = 'md'; host.appendChild(para); }
    else para.appendChild(document.createElement('br'));                  // 段内换行
    inlineInto(para, line);
  }
}

const END_WORD = { stopped: L("end.stopped"), max_steps: L("end.maxSteps"), timeout: L("end.timeout"),
                   no_progress: L("end.noProgress"), interrupted: L("end.interrupted") };

function addAnswer(d) {
  if (!turn) return;
  const t = turn, good = d.end === 'done_success';

  // 跑完了，「当前这一步」这块就没意义了 —— 它的职责只存在于运行期间
  t._live.remove();

  const box = el(uiCat('answer', good ? '' : ' fail'));
  // ⚠ 成功不加标签。每条前面写「✓ 完成」等于每句话前加一句「我做完了」——
  //   成功是默认，内容自己会说话。只有失败才标，那时候标记才重新有信息量。
  if (!good) {
    const k = el('k');
    k.innerHTML = ICON.warn;                                          // 常量图标
    k.appendChild(uiNode(END_WORD[d.end] || L("end.failed")));
    box.appendChild(k);
  }
  renderRich(box, d.result || I18n.message(d.error_message, d.error) || '');

  // 摘要行：步数 + 用时 + **异常计数**。收起不等于隐藏 ——
  // 「一路顺畅」和「中间踉跄过」是两件事，这一行必须说出来。
  const nc = t._steps.filter(s => s.changed === false && !s.error).length;
  const er = t._steps.filter(s => s.error).length;
  const ms = t._steps.reduce((a, s) => a + (s.exec_ms || 0) + (s.latency_ms || 0), 0);
  const sum = el(uiCat('sum', good ? '' : ' bad'));
  sum.appendChild(el('sdot'));
  sum.appendChild(el('', uiJoin([L('step.count', { count: d.steps || t._steps.length }), humanTime(ms)].filter(Boolean), ' · ')));
  if (nc || er) {
    sum.appendChild(el('warnpill',
      uiJoin([nc ? L('step.unchangedCount', { count: nc }) : '', er ? L('step.rejectedCount', { count: er }) : ''].filter(Boolean), ' · ')));
  }
  sum.appendChild(el('sp'));

  /* ⚠ 展开必须有一个**长得像控件**的东西。原来整行可点却没有任何视觉信号，
       那是个隐形热区 —— 用户凭什么知道能点。
     ⚠ 而且原来这一行塞了两个动作：展开（就地、可逆、轻）和回放（全屏覆盖层、
       重），两个三角形 ▸ ▶ 还挤在一起，看不出哪个是哪个。
       现在合并：这一行只留「查看过程」，回放挪到展开之后的列表顶部 ——
       它俩本来就是同一件事的两个深度（文字轨迹 vs 画面轨迹）。 */
  /* 两个动作**并排放在同一横条上**。
     之前把回放挪到展开区顶部，结果那一行悬在中间，反而把回复和过程割开了。 */
  const rid = d.run || t._run;
  if (rid) {
    const rp = document.createElement('button');
    rp.className = 'go';
    rp.innerHTML = ICON.play;                         // 常量图标
    rp.appendChild(uiNode(L("replay.label")));
    // ⚠ 用 run id 而不是内存里那份：后端那份带真实耗时，节奏才对；刷新之后也放得出来。
    rp.onclick = e => { e.stopPropagation(); openZoom({ runId: rid, play: true }); };
    sum.appendChild(rp);
  }
  const go = document.createElement('button');
  go.className = 'go pri';
  const goLabel = document.createElement('span');
  I18n.bind(goLabel, "textContent", L("process.show"));
  const goChev = el('gochev');
  goChev.innerHTML = ICON.chev;                       // 常量图标
  go.append(goLabel, goChev);
  go.onclick = e => { e.stopPropagation(); setExpanded(t, !t._expanded); };
  sum.appendChild(go);
  sum.onclick = () => setExpanded(t, !t._expanded);   // 整行仍是大热区，但按钮才是视觉锚
  t._sum = sum;
  t._go = goLabel;
  t._more = null;

  // 成功 → 结果在上、过程收起；失败 → 过程默认展开，**那时候过程就是答案**。
  // 但运行中已经展开过的，一律保持展开 —— 用户明确说过「我要看过程」。
  const open = t._expanded || !good;
  t._proc.insertBefore(sum, t._wrap);
  t._proc.hidden = false;
  // 有回复了才收成 fit-content —— 跑的时候列表需要地方，别提前掐窄
  t._reply.classList.add('sized');
  if (good) {
    t._reply.insertBefore(box, t._proc); // 成功：回复在上，过程收在下面
  } else {
    t._reply.appendChild(box);           // 失败：结论放在过程之后，读完过程正好看到它
  }
  setExpanded(t, open);
  bottom();
}

function humanTime(ms) {
  if (!ms) return '';
  const s = Math.round(ms / 1000);
  return s >= 60 ? L('time.minutes', { minutes: Math.floor(s / 60), seconds: s % 60 }) : L('time.seconds', { seconds: s });
}

function banner(kind, title, detail, action) {
  const b = el(uiCat('banner ', kind));
  const i = el(''); i.innerHTML = kind === 'info' ? ICON.info : ICON.warn;
  b.appendChild(i.firstChild);
  const t = el('b-txt');
  const bb = document.createElement('b'); I18n.bind(bb, "textContent", title);
  t.appendChild(bb);
  if (detail) { const s = document.createElement('span'); I18n.bind(s, "textContent", detail); t.appendChild(s); }
  b.appendChild(t);
  if (action) {
    const btn = document.createElement('button');
    btn.className = 'btn'; I18n.bind(btn, "textContent", action.label);
    btn.onclick = () => { b.remove(); action.run(); };
    b.appendChild(btn);
  } else {
    const btn = document.createElement('button');
    btn.className = 'btn'; I18n.bind(btn, "textContent", L("common.dismiss"));
    btn.onclick = () => b.remove();
    b.appendChild(btn);
  }
  $('#banners').appendChild(b);
  return b;
}

// 安全闸（harness/safety.py）：它要做一个写操作，人不点「允许」它就不做。
// 只允许这一次；超时算拒绝。别的标签页答了，这里收到 confirmed 事件一起撤掉。
let confirmEl = null;
function confirmBanner(text, timeoutS) {
  dismissConfirm();
  const b = el('banner bad');
  const i = el(''); i.innerHTML = ICON.warn; b.appendChild(i.firstChild);
  const t = el('b-txt');
  const bb = document.createElement('b'); I18n.bind(bb, "textContent", L("confirm.title"));
  t.appendChild(bb);
  const s = document.createElement('span');
  I18n.bind(s, "textContent", uiCat(text, timeoutS ? L('confirm.timeout', { seconds: timeoutS }) : ''));
  t.appendChild(s);
  b.appendChild(t);
  const answer = ok => async () => { dismissConfirm(); await api.post('/api/confirm', { ok }); };
  const no = document.createElement('button'); no.className = 'btn'; I18n.bind(no, "textContent", L("confirm.deny")); no.onclick = answer(false);
  const yes = document.createElement('button'); yes.className = 'btn pri'; I18n.bind(yes, "textContent", L("confirm.allow")); yes.onclick = answer(true);
  b.appendChild(no); b.appendChild(yes);
  $('#banners').appendChild(b);
  confirmEl = b;
}
function dismissConfirm() { if (confirmEl) { confirmEl.remove(); confirmEl = null; } }

/* ── 手机画面 ──
   任务在跑的时候画面由 SSE 的 step 事件驱动；空闲的时候自己去抓一帧。
   ⚠ 两者不能同时来 —— 会互相抢镜像窗口，所以 liveTick 只在 !running 时才发请求。 */
// ⚠ steps.jsonl 里的帧名是**不带运行目录的**（frame_179.png），而 /api/frame/ 是
//   相对 runs/ 解析的。不补前缀就是一路 404 —— 点某一步什么都不显示。
//   本次会话的运行 id 从 SSE 的 start 事件拿；历史运行由 /api/runs/<id> 直接补好。
let currentRun = null;
const frameURL = name => !name ? null
  : uiCat('/api/frame/', name.includes('/') ? name : (currentRun ? uiCat(currentRun, '/') + name : name));

function showFrame(name, step) {
  const url = frameURL(name);
  if (!url) return;
  shot.src = url;
  shot.classList.remove('stale');
  if (step) {
    I18n.bind($('.a', hud), "textContent", uiCat('#', step.n));
    I18n.bind($('.b', hud), "textContent", humanize(step) || step.name || '');
    hud.classList.add('on');
  }
  I18n.bind($('#rmeta'), "textContent", String(name).replace(/\.png$/, ''));
}

let liveBusy = false;
async function liveTick(force) {
  // 进来第一眼那栏不该是一块黑的：用户对这个产品的信心来自「我看见它在动我的手机」，
  // 而这份信心得在他还没发出第一条指令之前就建立起来。
  //
  // ⚠ document.hidden 只用来掐掉**轮询**，掐不掉首次加载和切回前台那一次。
  //   有些嵌入式浏览器上下文会永远报 hidden —— 一刀切的话画面就永远是黑的。
  if (running || liveBusy || (document.hidden && !force)) return;
  liveBusy = true;
  try {
    const r = await fetch(uiCat('/api/screen.png?t=', Date.now()));
    if (!r.ok) { noMirror(r.status); return; }
    const b = await r.blob();
    const url = URL.createObjectURL(b);
    const old = shot.dataset.blob;
    shot.src = url;
    shot.classList.remove('stale');
    shot.dataset.blob = url;
    if (old) URL.revokeObjectURL(old);         // 不撤销的话每两秒漏一张图
    hud.classList.remove('on');
    I18n.bind($('#rmeta'), "textContent", L("screen.liveRead"));
  } catch { /* 服务重启中，下一拍再说 */ } finally { liveBusy = false; }
}
function noMirror(code) {
  if (code === 409) return;                    // 正在跑任务，画面由 step 事件给
  shot.removeAttribute('src');
  I18n.bind($('#rmeta'), "textContent", L("screen.disconnected"));
}

/* ══════════════════════════════════════════════════════════════════════
   3. 发送 / 停止
   ══════════════════════════════════════════════════════════════════════ */
function setRunning(on) {
  running = on;
  send.classList.toggle('stop', on);
  send.innerHTML = on ? ICON.stop : ICON.send;                        // 常量图标
  I18n.bind(send, "title", on ? L("action.stop") : L("action.send"));
  pauseBtn.hidden = !on;
  if (!on) setPaused(false);
  I18n.bind($('#rlive'), "textContent", on ? L("state.live") : L("state.idle"));
  $('#rdot').className = uiCat('dot', on ? ' busy' : '');
  $('#raildot').className = uiCat('dot', on ? ' busy' : '');
  if (!on) I18n.bind($('#railtxt'), "textContent", L("state.idle"));
}

// 暂停 → 补一句话 → 继续。任务跑偏时以前唯一的选择是停止、推倒重来。
// 暂停中输入框变成「补一句话」，回车或点 ▶ 就带着这句话继续（留空也行）。
function setPaused(on) {
  paused = on;
  pauseBtn.innerHTML = on ? ICON.play : ICON.pause;
  I18n.bind(pauseBtn, "title", on ? L("action.resume") : L("action.pause"));
  pauseBtn.classList.toggle('on', on);
  I18n.bind(q, "placeholder", on ? L("chat.resumePrompt") : L("chat.prompt"));
  if (on) q.focus();
}

async function ask() {
  if (running) return paused ? resume() : stop();
  const text = q.value.trim();
  if (!text) return;
  q.value = ''; q.style.height = 'auto';
  const r = await api.post('/api/send', { text });
  if (r && r.error) banner('bad', L("error.send"), apiMessage(r, 'error'));
}

async function newChat() {
  const r = await api.post('/api/new-chat');
  if (r && r.ok) { turn = null; tl = null; lastG = null; showEmpty(); }
  else banner('bad', L("error.title"), apiMessage(r || {}, 'error'));
}

async function stop() {
  send.disabled = true;
  try { await api.post('/api/stop'); } finally { send.disabled = false; }
}

async function pause() {
  pauseBtn.disabled = true;
  try { await api.post('/api/pause'); } finally { pauseBtn.disabled = false; }
}

async function resume() {
  const text = q.value.trim();
  q.value = ''; q.style.height = 'auto';
  pauseBtn.disabled = true;
  try { await api.post('/api/resume', { text }); } finally { pauseBtn.disabled = false; }
}

send.onclick = ask;
pauseBtn.onclick = () => (paused ? resume() : pause());
$('#newchat').onclick = newChat;
q.addEventListener('input', () => { q.style.height = 'auto'; q.style.height = uiCat(Math.min(q.scrollHeight, 100), 'px'); });
q.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (!running || paused) ask(); }
});

/* ══════════════════════════════════════════════════════════════════════
   4. 事件流
   ══════════════════════════════════════════════════════════════════════ */
function connect() {
  const es = new EventSource('/api/stream');
  es.onmessage = ev => {
    let d;
    try { d = JSON.parse(ev.data); } catch { return; }
    switch (d.kind) {
      case 'user': startTurn(d.text); setRunning(true); break;
      case 'start': currentRun = d.run; if (turn) turn._run = d.run; setRunning(true); break;
      case 'step':
        addStep(d);
        if (turn) { turn._steps.push(d); updateLive(turn, d); }
        if (d.frame) showFrame(d.frame, d);
        break;
      case 'frame': showFrame(d.frame); break;
      case 'stopping': banner('info', L("state.stopping"), eventMessage(d)); break;
      case 'paused': setPaused(true); banner('info', L("state.paused"), eventMessage(d)); break;
      case 'handover': setPaused(true); banner('info', L("handover.title"), eventMessage(d)); break;
      case 'confirm': confirmBanner(eventMessage(d), d.timeout_s); break;
      case 'confirmed': dismissConfirm(); break;
      case 'resumed': setPaused(false); banner('info', L("action.resume"), d.text ? L('chat.added', { text: d.text }) : ''); break;
      case 'done': addAnswer(d); setRunning(false); refreshState(); refreshRuns(); break;
      case 'memory':
        banner('info', L("memory.new"),
               uiJoin(d.names || [], '、'),
               { label: L("action.undo"), run: () => (d.names || []).forEach(undoMemory) });
        break;
      case 'error': setRunning(false); banner('bad', L("error.title"), eventMessage(d)); break;
    }
  };
  // 连接断了浏览器会自己重连（EventSource 内建）。这里只把状态显性化 ——
  // 「它是不是还活着」是用户最常问的问题。
  es.onerror = () => { $('#sdot').className = 'dot warn'; I18n.bind($('#sst1'), "textContent", L("state.reconnecting")); };
  es.onopen = () => refreshState();
}

async function undoMemory(name) {
  const r = await api.post('/api/memory/undo', { name });
  banner('info', r && r.ok ? L('memory.undone', { name }) : L('memory.notFound', { name }), '');
}

/* ══════════════════════════════════════════════════════════════════════
   5. 状态条
   ══════════════════════════════════════════════════════════════════════ */
async function refreshState() {
  let s;
  try { s = await api.get('/api/state'); } catch { return; }
  setRunning(!!s.running);
  if (s.running) setPaused(!!s.paused);
  if (s.confirm && !confirmEl) confirmBanner(s.confirm_message ? I18n.message(s.confirm_message, s.confirm) : s.confirm, s.confirm_timeout_s);     // 刷新页面时把还没答的问题找回来
  if (!s.confirm) dismissConfirm();
  const m = s.model || {};
  if (!m.ready) {
    $('#sdot').className = 'dot bad';
    I18n.bind($('#sst1'), "textContent", L("state.notConfigured"));
    I18n.bind($('#sst2'), "textContent", L("settings.openHint"));
  } else {
    $('#sdot').className = uiCat('dot', s.running ? ' busy' : '');
    I18n.bind($('#sst1'), "textContent", s.stopping ? L("state.stoppingEllipsis") : (s.paused ? L("state.paused") : (s.running ? L("state.running") : L("state.connected"))));
    I18n.bind($('#sst2'), "textContent", m.spec || '');
  }
}

/* ══════════════════════════════════════════════════════════════════════
   6. 导航 / 外观
   ══════════════════════════════════════════════════════════════════════ */
const LOADERS = { settings: renderSettings, learned: renderLearned };
function nav(name) {
  // ⚠ 回放层盖在主区上。不关掉的话，切了导航、面板也换了，但用户看见的还是回放 ——
  //   看起来就是「点了没反应」。换页 = 离开这次回放。
  closeZoom();
  $$('.sitem').forEach(b => b.setAttribute('aria-current', String(b.dataset.nav === name)));
  $$('.pane').forEach(p => p.classList.toggle('on', p.dataset.pane === name));
  if (LOADERS[name]) LOADERS[name]();
}
$$('[data-nav]').forEach(b => b.onclick = () => nav(b.dataset.nav));

const THEMES = [null, 'light', 'dark'];
let ti = 0;
try { ti = Math.max(0, THEMES.indexOf(localStorage.getItem('theme'))); } catch { ti = 0; }
function applyTheme() {
  if (THEMES[ti]) document.documentElement.dataset.theme = THEMES[ti];
  else delete document.documentElement.dataset.theme;
  try { localStorage.setItem('theme', THEMES[ti] || ''); } catch { /* 隐私模式下写不了，无所谓 */ }
}
applyTheme();
$('#themebtn').onclick = () => { ti = (ti + 1) % 3; applyTheme(); };
$('#detail').onclick = e => e.currentTarget.classList.toggle('on', document.body.classList.toggle('detail'));

/* ══════════════════════════════════════════════════════════════════════
   7. 设置
   ══════════════════════════════════════════════════════════════════════ */
function row(label, ...right) {
  const r = el('row');
  if (label) { const k = el('rk', label); r.appendChild(k); }
  const rr = el('rr grow');
  rr.style.justifyContent = 'flex-end';
  right.forEach(x => rr.appendChild(x));
  r.appendChild(rr);
  return r;
}
function note(icon, ...nodes) {
  const n = el('note');
  const i = el(''); i.innerHTML = icon;
  n.appendChild(i.firstChild);
  const d = el('');
  nodes.forEach(x => d.appendChild((typeof x === 'string' || I18n.isMessage(x)) ? uiNode(x) : x));
  n.appendChild(d);
  return n;
}
const strong = t => { const b = document.createElement('b'); I18n.bind(b, "textContent", t); return b; };
const code = t => { const s = document.createElement('span'); s.className = 'mono'; I18n.bind(s, "textContent", t); return s; };

/* ══════════════════════════════════════════════════════════════════════
   设置：一张连接卡片
   ══════════════════════════════════════════════════════════════════════
   这一页最主要的功能就是「接上一个模型」，而那需要三样：
   **服务商 + 模型 id + 密钥**。所以它们必须在一张卡片里，跟首启弹层一致 ——
   同一件事在两个地方不该长成两个样子。

   三条来自实测的教训：
   ① 换服务商曾经**换不了** —— onchange 里调 renderSettings() 会把已保存的配置
      重新填回下拉框，用户的选择在下一帧就被覆盖。所以现在改字段**只动本地状态**，
      绝不回头拉配置。
   ② 「测试连接」的结果曾经**看不见** —— 它被塞进 #banners，而那个容器在对话页里，
      你在设置页时对话页是 display:none。反馈必须就地显示。
   ③ 拆成「保存」和「测试连接」两个按钮，必然产生「测的是刚填的还是已存的」这个歧义。
      合并成一个主动作：**保存并测试连接**。
*/
const CUSTOM = '__custom__';

async function renderSettings() {
  const body = $('#settingsBody');
  I18n.bind(body, "textContent", L("common.loading"));
  let cfg, mods;
  try {
    [cfg, mods] = await Promise.all([api.get('/api/config'), api.get('/api/models')]);
  } catch (e) {
    uiText(body, uiCat(L('common.loadFailed'), '\n', e.uiMessage || String(e)));
    return;
  }
  I18n.bind(body, "textContent", '');

  const cur = cfg.current || {};        // ⚠ 就地改这个对象（refreshCard 用 Object.assign），别重新绑定
  const provList = mods.providers || [];
  const byName = Object.fromEntries(provList.map(p => [p.name, p]));
  const calibrated = new Set((mods.models || []).filter(m => m.calibrated).map(m => m.spec));
  const dflt = mods.default || '';

  // 本地草稿。⚠ 它是这张卡片的唯一真相 —— 改字段只动它，不回头拉配置（教训①）。
  const draft = {
    provider: cur.provider || dflt.split(':')[0] || (provList[0] || {}).name || '',
    model: cur.model_id || dflt.split(':')[1] || '',
    key: '',
    base: '',
    customName: '',
    adding: false,
  };

  const card = el('group');
  const h = document.createElement('h3'); I18n.bind(h, "textContent", L("model.connection")); card.appendChild(h);
  const list = el('list');
  card.appendChild(list);
  body.appendChild(card);

  /* 顶部状态：用户打开这一页最想知道的就是「现在到底能不能用」。 */
  const head = el('conn');
  const dot = el('dot');
  const headTxt = el('grow');
  const headA = el(''); headA.style.fontSize = 'var(--f-body)'; headA.style.fontWeight = '530';
  const headB = el('rd');
  headTxt.append(headA, headB);
  head.append(dot, headTxt);
  list.appendChild(head);

  function paintHead(dirty) {
    const ready = !!cur.spec && !cfg.current.problem;
    dot.className = uiCat('dot', dirty ? ' warn' : ready ? '' : ' bad');
    I18n.bind(headA, "textContent", dirty ? L("settings.unsaved") : ready ? L("state.connected") : L("model.notReady"));
    I18n.bind(headB, "textContent", dirty ? L("settings.saveHint")
      : ready ? (cur.spec || '') : (apiMessage(cur, 'problem') || L("model.keyMissing")));
  }

  const provSel = document.createElement('select');
  provSel.className = 'inp';
  for (const p of provList) {
    const o = document.createElement('option');
    o.value = p.name;
    I18n.bind(o, "textContent", uiCat(p.name, dflt.startsWith(uiCat(p.name, ':')) ? L("model.recommended") : p.custom ? L("model.custom") : ''));
    provSel.appendChild(o);
  }
  const oc = document.createElement('option');
  oc.value = CUSTOM;
  I18n.bind(oc, "textContent", L("model.addEndpoint"));
  provSel.appendChild(oc);
  provSel.value = draft.provider;

  const nameInp = document.createElement('input');
  nameInp.className = 'inp';
  I18n.bind(nameInp, "placeholder", L("model.namePlaceholder"));
  const baseInp = document.createElement('input');
  baseInp.className = 'inp';
  I18n.bind(baseInp, "placeholder", 'https://…/v1');

  const modelInp = document.createElement('input');
  modelInp.className = 'inp';
  modelInp.value = draft.model;
  I18n.bind(modelInp, "placeholder", L("model.idPlaceholder"));

  const keyInp = document.createElement('input');
  keyInp.className = 'inp'; keyInp.type = 'password'; keyInp.autocomplete = 'off';

  const rowName = row(L("field.name"), nameInp);
  const rowBase = row(L("field.baseUrl"), baseInp);
  list.append(row(L("field.provider"), provSel), rowName, rowBase, row(L("field.model"), modelInp), row(L("field.key"), keyInp));

  const tips = el('row');
  const tipG = el('grow');
  const tipCal = el('rd'), tipKey = el('rd');
  tipG.append(tipCal, tipKey);
  tips.appendChild(tipG);
  list.appendChild(tips);

  const acts = el('row');
  const actL = el('grow');
  const result = el('res');
  result.hidden = true;
  actL.appendChild(result);
  const clearBtn = document.createElement('button');
  clearBtn.className = 'btn dgr'; I18n.bind(clearBtn, "textContent", L("model.clearKey"));
  const saveBtn = document.createElement('button');
  saveBtn.className = 'btn pri'; I18n.bind(saveBtn, "textContent", L("model.saveTest"));
  const rr = el('rr'); rr.append(clearBtn, saveBtn);
  acts.append(actL, rr);
  list.appendChild(acts);

  function sync(dirty) {
    draft.adding = provSel.value === CUSTOM;
    const name = draft.adding ? nameInp.value.trim().toLowerCase() : provSel.value;
    draft.provider = name;
    rowName.hidden = !draft.adding;
    // 自定义端点**必须**有 base_url，这是 resolve 的硬要求；内置的不用填。
    rowBase.hidden = !(draft.adding || (byName[name] || {}).custom);
    if (!draft.adding && byName[name] && baseInp.value === '') baseInp.value = byName[name].base_url || '';

    const p = byName[name] || {};
    const has = (cfg.providers || {})[name];
    I18n.bind(keyInp, "placeholder", has && has.has_key ? L("model.savedKey") : 'sk-…');

    const spec = uiCat(name, ':') + modelInp.value.trim();
    // 内置表只有一个模型标定过坐标。不拦人，但必须说 —— 否则换过去会觉得「它变笨了」。
    I18n.bind(tipCal, "textContent", calibrated.has(spec)
      ? L("model.calibrated")
      : L("model.uncalibrated"));

    /* ⚠ 环境变量优先级比文件高。存了密钥却不生效是最难查的一种「没反应」，
         必须在这里说人话。 */
    const src = cur.api_key_source || '';
    if (src.startsWith('env:')) {
      I18n.bind(tipKey, "textContent", L('model.envOverride', { name: src.slice(4) }));
      tipKey.style.color = 'var(--orange)';
    } else {
      const envs = uiJoin(p.env_vars || [], ' / ');
      const where = CONSOLE[name];
      I18n.bind(tipKey, "textContent", uiJoin([where ? L('model.keyWhere', { console: where }) : '',
                            envs ? L('model.keyEnv', { names: envs }) : ''].filter(Boolean), ' · '));
      tipKey.style.color = '';
    }
    paintHead(dirty);
  }

  function dirty() { sync(true); result.hidden = true; }
  provSel.onchange = dirty;                 // ① 只动本地状态，绝不回头拉配置
  [nameInp, baseInp, modelInp, keyInp].forEach(i => { i.oninput = dirty; });

  function say(kind, text) {                // ② 反馈就地显示，不去别的页面
    result.hidden = false;
    result.className = uiCat('res ', kind);
    I18n.bind(result, "textContent", '');
    const i = el('');
    i.innerHTML = kind === 'ok' ? ICON.ok : kind === 'bad' ? ICON.warn : ICON.info;
    result.append(i.firstChild, uiNode(text));
  }

  saveBtn.onclick = async () => {
    const name = draft.provider;
    const model = modelInp.value.trim();
    if (!name) return say('bad', L("model.enterProvider"));
    if (!model) return say('bad', L("model.enterId"));
    const needBase = draft.adding || (byName[name] || {}).custom;
    if (needBase && !baseInp.value.trim()) return say('bad', L("model.enterBase"));

    saveBtn.disabled = true;
    say('wait', L("model.saving"));
    try {
      const p1 = { provider: name };
      if (keyInp.value) p1.api_key = keyInp.value;
      if (needBase) p1.base_url = baseInp.value.trim();
      if (p1.api_key !== undefined || p1.base_url !== undefined) {
        const r1 = await api.post('/api/config', p1);
        if (r1.error) { saveBtn.disabled = false; return say('bad', apiMessage(r1, 'error')); }
      }
      keyInp.value = '';                    // 存完立刻从 DOM 上抹掉
      const r2 = await api.post('/api/config', { model: uiCat(name, ':') + model });
      if (r2.error) { saveBtn.disabled = false; return say('bad', apiMessage(r2, 'error')); }
      if (!r2.ready) { saveBtn.disabled = false; return say('bad', apiMessage(r2, 'problem') || L("model.savedNotReady")); }

      const t = await api.post('/api/model/test');
      saveBtn.disabled = false;
      say(t.ok ? 'ok' : 'bad', t.ok ? (apiMessage(t, 'detail') || L("model.testOk")) : (apiMessage(t, 'problem') || L("model.testFailed")));
      refreshState();
      // ⚠ 成功之后**不整页重绘** —— 重绘会把刚说的那句「通了」一起抹掉，
      //   用户等了几秒就为了看这一句。只就地更新状态和提示。
      await refreshCard();
    } catch (e) {
      saveBtn.disabled = false;
      say('bad', L('error.request', { error: String(e) }));
    }
  };
  clearBtn.onclick = async () => {
    const r = await api.post('/api/config', { provider: draft.provider, api_key: '' });
    say(r.error ? 'bad' : 'info', apiMessage(r, 'error') || L("model.keyCleared"));
    refreshState();
    await refreshCard();
  };

  /* 只更新状态和提示，不重建 DOM —— 反馈那句话得留在原地。 */
  async function refreshCard() {
    try {
      const fresh = await api.get('/api/config');
      Object.assign(cfg, fresh);
      Object.assign(cur, fresh.current || {});
      const m = await api.get('/api/models');
      m.providers.forEach(p => { byName[p.name] = p; });
    } catch { /* 读不到就维持现状，别把界面搞乱 */ }
    sync(false);
  }

  sync(false);

  /* 连接与权限放最后 —— 第一次打开不该被一串系统授权拦住（docs/22 P2）。 */
  const g3 = el('group');
  const h3 = document.createElement('h3'); I18n.bind(h3, "textContent", L("settings.permissions")); g3.appendChild(h3);
  const list3 = el('list');
  list3.appendChild(el('check', L("settings.checking")));
  g3.appendChild(list3);
  body.appendChild(g3);
  loadDoctor(list3);
}

async function loadDoctor(list) {
  let d;
  try { d = await api.get('/api/doctor'); } catch { return; }
  I18n.bind(list, "textContent", '');
  (d.checks || []).forEach(c => {
    const row = el(uiCat('check ', c.status === 'pass' ? 'pass' : c.status === 'fail' ? 'fail' : ''));
    const cs = el('cs');
    cs.innerHTML = c.status === 'pass' ? ICON.ok : c.status === 'fail' ? ICON.warn : ICON.info;
    const cb = el('cb');
    const b = document.createElement('b'); I18n.bind(b, "textContent", I18n.message(c.messages && c.messages.title, c.title));
    const em = document.createElement('em');
    I18n.bind(em, "textContent", uiJoin([I18n.message(c.messages && c.messages.detail, c.detail), c.status === 'fail' ? I18n.message(c.messages && c.messages.fix, c.fix) : I18n.message(c.messages && c.messages.why, c.why)].filter(Boolean), ' —— '));
    cb.append(b, em);
    row.append(cs, cb);
    if (c.fix_url) {
      // 直接跳到对应那一页系统设置。让用户自己去找「隐私与安全性 → 辅助功能」
      // 是产品化里最容易掉人的一步。
      const ca = el('ca');
      const a = document.createElement('a');
      a.className = 'btn pri'; a.href = c.fix_url; I18n.bind(a, "textContent", L("settings.grant"));
      a.style.textDecoration = 'none';
      ca.appendChild(a);
      row.appendChild(ca);
    }
    list.appendChild(row);
  });
}

/* ══════════════════════════════════════════════════════════════════════
   8. 知识
   ══════════════════════════════════════════════════════════════════════ */
async function renderLearned() {
  const body = $('#learnedBody');
  I18n.bind(body, "textContent", L("common.loading"));
  let d;
  try { d = await api.get('/api/learned'); } catch { I18n.bind(body, "textContent", L("common.loadFailed")); return; }
  I18n.bind(body, "textContent", '');

  const badge = $('#learnedBadge');
  I18n.bind(badge, "textContent", String(d.pending || 0));
  badge.hidden = !d.pending;

  if (!(d.apps || []).length && !(d.memories || []).length) {
    body.appendChild(note(ICON.info,
      L("knowledge.empty")));
    return;
  }

  (d.apps || []).forEach(a => {
    const g = el('group');
    const h = document.createElement('h3'); I18n.bind(h, "textContent", a.display); g.appendChild(h);
    const list = el('list');
    if (!a.procedures.length) list.appendChild(el('row', L("knowledge.noProcedures")));
    a.procedures.forEach(p => {
      const r = el('row');
      const gr = el('grow');
      const t = el(''); t.style.fontSize = 'var(--f-body)'; I18n.bind(t, "textContent", p.name);
      const sub = el('rd', uiJoin([p.description, L('step.count', { count: p.steps }),
                            p.verified ? L('knowledge.verifiedCount', { count: p.verified }) : ''].filter(Boolean), ' · '));
      gr.append(t, sub);
      const rr = el('rr');
      rr.appendChild(p.needs_approval
        ? el('tagpill warn', L("knowledge.approval"))
        : el('tagpill good', p.risk === 'read' ? L("knowledge.read") : L("knowledge.approved")));
      r.append(gr, rr);
      list.appendChild(r);
    });
    g.appendChild(list);
    body.appendChild(g);
  });

  if ((d.memories || []).length) {
    const g = el('group');
    const h = document.createElement('h3'); I18n.bind(h, "textContent", L("memory.title")); g.appendChild(h);
    const list = el('list');
    d.memories.forEach(m => {
      const r = el('row');
      const gr = el('grow');
      const t = el(''); t.style.fontSize = 'var(--f-body)'; I18n.bind(t, "textContent", m.description || m.name);
      const sub = el('rd', uiJoin([m.name, m.created,
                            m.used_success ? L('memory.successCount', { count: m.used_success }) : ''].filter(Boolean), ' · '));
      gr.append(t, sub);
      const rr = el('rr');
      const del = document.createElement('button');
      del.className = 'mini danger'; I18n.bind(del, "textContent", L("action.delete"));
      del.onclick = async () => { await undoMemory(m.name); renderLearned(); };
      rr.appendChild(del);
      r.append(gr, rr);
      list.appendChild(r);
    });
    g.appendChild(list);
    body.append(g, note(ICON.info, L("memory.local"),
                        L("memory.inject"), strong(L("memory.data")), '。'));
  }
}

/* ══════════════════════════════════════════════════════════════════════
   9. 首启：只配模型
   ══════════════════════════════════════════════════════════════════════
   ⚠ 这里**只做一件事**。权限、镜像连接、驱动准入统统不放在这儿 ——
     第一次打开就被一串系统授权拦住，人还没看见这东西能干嘛就先被要走两个权限，
     劝退。那些留在设置页里，用户想看的时候自己去看。

   全部走后端**已有**的端点：/api/models、/api/config、/api/model/test。
*/

// 去哪儿拿这个 key。只写名字不给链接：这是个离线的本机工具，页面里不放外部资源。
const CONSOLE = {
  alibaba: L("console.alibaba"),
  deepseek: L("console.deepseek"),
  openai: 'OpenAI Platform',
  anthropic: 'Anthropic Console',
  moonshot: L("console.moonshot"),
  zhipu: L("console.zhipu"),
  openrouter: 'OpenRouter',
};

function field(labelText, control, tipText) {
  const f = el('field');
  const l = document.createElement('label');
  I18n.bind(l, "textContent", labelText);
  f.append(l, control);
  if (tipText) f.appendChild(el('tip', tipText));
  return f;
}

async function openSetup() {
  const box = $('#setup'), body = $('#setupBody');
  I18n.bind(body, "textContent", '');
  let mods, cfg;
  try {
    [mods, cfg] = await Promise.all([api.get('/api/models'), api.get('/api/config')]);
  } catch (e) {
    uiText(body, uiCat(L('common.loadFailed'), '\n', e.uiMessage || String(e)));
    box.hidden = false;
    $('#setupGo').disabled = true;
    $('#setupLater').onclick = closeSetup;
    return;
  }
  $('#setupGo').disabled = false;

  const dflt = mods.default || '';
  const provSel = document.createElement('select');
  provSel.className = 'inp';
  (mods.providers || []).forEach(p => {
    const o = document.createElement('option');
    o.value = p.name;
    I18n.bind(o, "textContent", uiCat(p.name, dflt.startsWith(uiCat(p.name, ':')) ? L("model.recommended") : ''));
    provSel.appendChild(o);
  });
  provSel.value = (cfg.current && cfg.current.provider) || dflt.split(':')[0] || provSel.value;

  const idInp = document.createElement('input');
  idInp.className = 'inp';
  idInp.value = (cfg.current && cfg.current.model_id) || dflt.split(':')[1] || '';
  I18n.bind(idInp, "placeholder", L("model.idPlaceholder"));

  const keyInp = document.createElement('input');
  keyInp.className = 'inp';
  keyInp.type = 'password';
  keyInp.autocomplete = 'off';
  I18n.bind(keyInp, "placeholder", 'sk-…');

  const idTip = el('tip');
  const keyTip = el('tip');
  const byName = Object.fromEntries((mods.providers || []).map(p => [p.name, p]));
  const calibrated = new Set((mods.models || []).filter(m => m.calibrated).map(m => m.spec));

  function syncTips() {
    const p = byName[provSel.value] || {};
    const envs = uiJoin(p.env_vars || [], ' / ');
    const where = CONSOLE[provSel.value];
    I18n.bind(keyTip, "textContent", uiJoin([where ? L('model.keyWhere', { console: where }) : '',
                          envs ? L('model.keyEnv', { names: envs }) : ''].filter(Boolean), ' · '));
    const spec = uiCat(provSel.value, ':') + idInp.value.trim();
    const has = (cfg.providers || {})[provSel.value];
    I18n.bind(keyInp, "placeholder", has && has.has_key ? L("model.savedKey") : 'sk-…');
    // 内置表是不对称的：只有一个模型在真机上标定过坐标约定。
    // 这不拦人，但得说一声，否则用户换一个之后会觉得「它变笨了」。
    I18n.bind(idTip, "textContent", calibrated.has(spec)
      ? L("model.calibrated")
      : L("model.uncalibrated"));
  }
  provSel.onchange = () => {
    // 换了服务商就得换模型 id —— 留着上一家的（把 qwen3.7-plus 发给 DeepSeek）
    // 会在真正发请求的时候才炸，而且报的是个看不懂的 404。
    idInp.value = dflt.startsWith(uiCat(provSel.value, ':')) ? dflt.split(':')[1] : '';
    syncTips();
  };
  idInp.oninput = syncTips;

  const f1 = field(L("field.provider"), provSel);
  const f2 = el('field');
  const l2 = document.createElement('label'); I18n.bind(l2, "textContent", L("field.model"));
  f2.append(l2, idInp, idTip);
  const f3 = el('field');
  const l3 = document.createElement('label'); I18n.bind(l3, "textContent", L("field.apiKey"));
  f3.append(l3, keyInp, keyTip);
  body.append(f1, f2, f3);
  const result = el('result');
  result.hidden = true;
  body.appendChild(result);
  syncTips();

  const go = $('#setupGo'), later = $('#setupLater');

  function say(kind, text) {
    result.hidden = false;
    result.className = uiCat('result ', kind);
    I18n.bind(result, "textContent", '');
    const i = el('');
    i.innerHTML = kind === 'ok' ? ICON.ok : kind === 'bad' ? ICON.warn : ICON.info;  // 常量图标
    result.append(i.firstChild, uiNode(text));
  }

  go.onclick = async () => {
    const id = idInp.value.trim();
    if (!id) return say('bad', L("model.enterId"));
    const has = (cfg.providers || {})[provSel.value];
    if (!keyInp.value && !(has && has.has_key)) return say('bad', L("model.enterKey"));

    go.disabled = true;
    say('wait', L("model.saving"));
    try {
      const payload = { model: uiCat(provSel.value, ':') + id };
      // 留空 = 不改已保存的那把。别把空串发过去 —— 那是「清除」的意思。
      if (keyInp.value) { payload.provider = provSel.value; payload.api_key = keyInp.value; }
      const saved = await api.post('/api/config', payload);
      keyInp.value = '';                       // 存完立刻从 DOM 上抹掉
      if (saved.error) { go.disabled = false; return say('bad', apiMessage(saved, 'error')); }

      const t = await api.post('/api/model/test');
      go.disabled = false;
      if (!t.ok) return say('bad', apiMessage(t, 'problem') || L("model.testFailed"));
      say('ok', apiMessage(t, 'detail') || L("model.testOk"));
      refreshState();
      setTimeout(closeSetup, 700);             // 让人看见那句「通了」再关
    } catch (e) {
      go.disabled = false;
      say('bad', L('error.request', { error: String(e) }));
    }
  };

  later.onclick = () => {
    closeSetup();
    // 不拦人，但也别让他以为已经配好了。给一条能点回来的提示，仅此而已。
    banner('info', L("model.notSet"), L("model.setupLater"),
           { label: L("settings.open"), run: () => nav('settings') });
  };

  box.hidden = false;
}

function closeSetup() { $('#setup').hidden = true; }

/* ══════════════════════════════════════════════════════════════════════
   10. 折叠手机栏
   ══════════════════════════════════════════════════════════════════════
   ⚠ 折叠走这个小按钮，**不走「点画面」** —— 点整块画面太容易误触，
     而且那个点击要留给「放大看」。折叠之后那条竖条本身可以点开，
     因为那时候它没有别的用途。
*/
function setFold(on) {
  document.body.classList.toggle('fold', on);
  I18n.bind($('#fold'), "title", on ? L("screen.expand") : L("screen.collapse"));
  try { localStorage.setItem('fold', on ? '1' : ''); } catch { /* 隐私模式，无所谓 */ }
}
$('#fold').onclick = () => setFold(!document.body.classList.contains('fold'));
try { if (localStorage.getItem('fold') === '1') setFold(true); } catch { /* 同上 */ }

/* ══════════════════════════════════════════════════════════════════════
   11. 放大 + 回放
   ══════════════════════════════════════════════════════════════════════
   ⚠ 它是**逐帧幻灯片，不是录像**。每一步只留了动作之后的一张截图，两步之间
     手机上的动画和中间态没有。看「走了哪条路、在哪一步开始打转」够用，
     看「某个动效对不对」不行。
*/
const zoom = $('#zoom'), zshot = $('#zshot');
const SPEEDS = [1, 2, 4, 0.5];
let zSteps = [], zCur = -1, zTimer = null, zPlaying = false, zSpeed = 1, zRunId = null;

/* 正在回放的那一次，在侧边栏列表里标出来 —— 否则一列相似的任务名，
   切过去了也看不出到底切没切。 */
function markRun(id) {
  zRunId = id || null;
  $$('#runs .r').forEach(b => b.classList.toggle('on', !!id && b.dataset.rid === id));
}

async function openZoom(opt) {
  opt = opt || {};
  let steps = null, title = '';
  if (opt.runId) {
    // 从后端取：那份带真实耗时，节奏才对，刷新之后也放得出来。
    try {
      const d = await api.get(uiCat('/api/runs/', opt.runId));
      steps = d.steps || [];
      title = d.task || opt.runId;
    } catch { steps = []; }
  }
  if (!steps || !steps.length) {
    // 退路：没有 run id（比如任务还在跑）就放本次会话收到的那份，节奏用固定节拍。
    steps = (turn && turn._steps) ? turn._steps.slice() : [];
    title = L("run.current");
  }
  zSteps = steps;
  markRun(opt.runId);
  I18n.bind($('#ztitle'), "textContent", title);
  renderZSteps();
  zoom.hidden = false;
  zStop();
  zGoto(opt.play ? -1 : Math.max(0, zSteps.length - 1), false);
  if (opt.play && zSteps.length) zPlay();
  else if (!zSteps.length) { zshot.src = shot.src || ''; I18n.bind($('#zkv'), "textContent", ''); }
}

function closeZoom() { zStop(); zoom.hidden = true; markRun(null); }

function renderZSteps() {
  const box = $('#zsteps');
  I18n.bind(box, "textContent", '');
  zSteps.forEach((s, i) => {
    const r = el(uiCat('zs ', cls(s)));
    r.appendChild(el('n', uiCat('#', s.n ?? i + 1)));
    const b = document.createElement('b');
    I18n.bind(b, "textContent", humanize(s) || s.name || '');
    r.appendChild(b);
    r.onclick = () => { zStop(); zGoto(i, true); };
    box.appendChild(r);
  });
}

function zGoto(i, scroll) {
  zCur = i;
  const s = zSteps[i];
  $$('#zsteps .zs').forEach((x, k) => x.classList.toggle('cur', k === i));
  if (scroll) {
    const cur = $('#zsteps .zs.cur');
    if (cur) cur.scrollIntoView({ block: 'nearest' });
  }
  $('#zprog i').style.width = zSteps.length ? uiCat((i + 1) / zSteps.length * 100, '%') : '0';
  if (!s) return;
  const url = frameURL(s.frame);
  if (url) zshot.src = url;
  const kv = $('#zkv');
  I18n.bind(kv, "textContent", '');
  const rows = [
    [L("step.label"), humanize(s) || s.name || ''],
    [L("step.tool"), s.name || ''],
    [L("step.args"), JSON.stringify(s.args || {}).slice(0, 90)],
    [L("step.changed"), s.error ? L("step.rejected") : (s.changed === true ? L("common.yes") : s.changed === false ? L("common.no") : '—')],
    [L("step.execTime"), s.exec_ms != null ? uiCat((s.exec_ms / 1000).toFixed(1), 's') : '—'],
    [L("step.modelTime"), s.latency_ms != null ? uiCat((s.latency_ms / 1000).toFixed(1), 's') : '—'],
    [L("step.frame"), (s.frame || '—').split('/').pop()],
  ];
  if (s.error) rows.push([L("step.error"), s.error]);
  for (const [k, v] of rows) {
    const d = el('kv');
    const a = document.createElement('span'); I18n.bind(a, "textContent", k);
    const b = document.createElement('span'); I18n.bind(b, "textContent", v);
    d.append(a, b);
    kv.appendChild(d);
  }
  if (s.reason) {
    const d = el('kv');
    const a = document.createElement('span'); I18n.bind(a, "textContent", L("step.reason"));
    const b = document.createElement('span'); I18n.bind(b, "textContent", s.reason);
    d.append(a, b);
    kv.appendChild(d);
  }
}

/* 节奏：按真实耗时等比压缩。原速要放三分半，没人看得下去；
   但比例得留着 —— 「哪一步卡了很久」是回放最有价值的信息之一。 */
function waitFor(s) {
  const real = (s && (s.exec_ms || 0) + (s.latency_ms || 0)) || 0;
  return Math.min(2200, Math.max(420, real / 6)) / zSpeed;
}
function zTick() {
  if (zCur + 1 >= zSteps.length) { zStop(); I18n.bind($('#zpp'), "textContent", L("replay.restart")); return; }
  zGoto(zCur + 1, true);
  zTimer = setTimeout(zTick, waitFor(zSteps[zCur]));
}
function zPlay() {
  if (!zSteps.length) return;
  if (zCur + 1 >= zSteps.length) zGoto(-1, false);
  zPlaying = true;
  I18n.bind($('#zpp'), "textContent", L("replay.pause"));
  zTick();
}
function zStop() {
  zPlaying = false;
  clearTimeout(zTimer);
  zTimer = null;
  I18n.bind($('#zpp'), "textContent", L("replay.play"));
}
$('#zpp').onclick = () => (zPlaying ? zStop() : zPlay());
$('#zrst').onclick = () => { zStop(); zGoto(0, true); };
$('#zspd').onclick = e => {
  zSpeed = SPEEDS[(SPEEDS.indexOf(zSpeed) + 1) % SPEEDS.length];
  I18n.bind(e.currentTarget, "textContent", uiCat(zSpeed, '×'));
};
zoom.onclick = e => { if (e.target === zoom) closeZoom(); };

// 点手机：展开态 = 放大；折叠态 = 展开（那时候它没有别的用途）
$('#phone').onclick = () => {
  if (document.body.classList.contains('fold')) return setFold(false);
  openZoom({ runId: (turn && turn._run) || currentRun });
};

/* ══════════════════════════════════════════════════════════════════════
   12. 最近运行
   ══════════════════════════════════════════════════════════════════════ */
async function refreshRuns() {
  let d;
  try { d = await api.get('/api/runs'); } catch { return; }
  const box = $('#runs');
  I18n.bind(box, "textContent", '');
  // ⚠ 不再 slice(0,12)。后端本来就只给 RUNS_LIMIT=40 条，这里再砍到 12 是
  //   在「列表不能滚」的年代加的挡箭牌 —— 挡住了溢出，也挡住了历史。
  //   现在 #runs 自己会滚了（app.css 那三条），砍它没有意义。
  (d.runs || []).forEach(r => {
    const b = document.createElement('button');
    b.className = uiCat('r', r.ok ? '' : ' bad') + (r.id === zRunId ? ' on' : '');
    b.dataset.rid = r.id;
    // title 必须带任务名：侧边栏收成 52px 图标栏时，这一条只剩一个状态点，
    // tooltip 是认出「这是哪次运行」的唯一手段。原来是固定文案，那时候等于没有。
    I18n.bind(b, "title", uiCat(r.task ? r.task.split('\n')[0] : L('run.noTask'), L("run.openHint")));
    b.appendChild(el('rd'));
    const g = el('grow');
    const t = document.createElement('b');
    I18n.bind(t, "textContent", r.task ? r.task.split('\n')[0] : L('run.noTask'));
    const e = document.createElement('em');
    I18n.bind(e, "textContent", uiJoin([when(r.started_at), L('step.count', { count: r.steps }),
                     r.ok ? '' : (r.finished ? L("end.failed") : L("run.unfinished"))].filter(Boolean), ' · '));
    g.append(t, e);
    b.appendChild(g);
    b.onclick = () => openZoom({ runId: r.id, play: true });
    box.appendChild(b);
  });
  // 滚离顶部才给「最近」那行加分隔线。onscroll 是赋值不是 addEventListener，
  // 每次重渲染覆盖同一个处理器，不会越积越多。
  box.onscroll = () => $('.snav-h').classList.toggle('stuck', box.scrollTop > 0);
}
function when(ts) {
  if (!ts) return '';
  const d = new Date(ts * 1000), now = new Date();
  const same = d.toDateString() === now.toDateString();
  return same ? d.toTimeString().slice(0, 5)
              : L('date.monthDay', { month: d.getMonth() + 1, day: d.getDate() });
}

/* ══════════════════════════════════════════════════════════════════════
   10. 起步
   ══════════════════════════════════════════════════════════════════════ */
setRunning(false);
showEmpty();
refreshState();
connect();
setInterval(refreshState, 5000);          // SSE 是主通道，这个只是兜底
refreshRuns();
liveTick(true);                           // 首帧无条件抓：不能让第一眼是一块黑的
setInterval(() => liveTick(), 2000);      // 轮询才受 hidden 约束；跑任务时自动让位给 SSE
// 标签页在后台时不轮询（省电，也不跟别的标签页抢镜像窗口），但切回来要**立刻**补一帧 ——
// 否则用户回到页面第一眼看到的是几分钟前的手机。
document.addEventListener('visibilitychange', () => { if (!document.hidden) liveTick(true); });

// 没配好模型就把那件事**摆在面前**，而不是等他打完字发出去才发现。
// 但也只有这一件 —— 权限、镜像连接一律不在这里拦人。
api.get('/api/state').then(s => {
  if (s && s.model && !s.model.ready) openSetup();
}).catch(() => {});
