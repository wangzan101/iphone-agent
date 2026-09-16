/* Local UI messages only. Never translate OCR, model replies, task text or stored evidence.
 * Bind explicit presentation values, not arbitrary DOM text. Switching language preserves
 * input drafts, event connections, confirmations and replay state without reloading.
 */
'use strict';
const I18n = (() => {
  const catalog = {
    "chat.prompt": ["想让它做什么？","What would you like it to do?"],
    "chat.intro": ["用大白话说就行。它记得上一轮说过什么，也记得以前跑过的任务。","Ask in your own words. It can use earlier conversation and saved experience."],
    "example.version": ["查一下 iOS 版本","Check the iOS version"],
    "example.versionPath": ["设置 → 通用 → 关于本机","Settings → General → About"],
    "example.battery": ["看看电量还剩多少","Check the battery level"],
    "example.batteryPath": ["设置 → 电池","Settings → Battery"],
    "example.screen": ["读一下当前页面的文字","Read the text on the current screen"],
    "example.screenNote": ["只读取当前画面","Read-only screen inspection"],
    "example.storage": ["看看有多少剩余存储","Check available storage"],
    "example.storagePath": ["设置 → 通用 → iPhone 储存空间","Settings → General → iPhone Storage"],
    "state.starting": ["正在开始…","Starting…"],
    "steps.expand": ["展开全部步骤","Show all steps"],
    "steps.latest": ["↓ 回到最新","↓ Jump to latest"],
    "steps.collapse": ["收起步骤","Hide steps"],
    "process.hide": ["收起过程","Hide details"],
    "process.show": ["查看过程","Show details"],
    "chat.continued": ["（继续）","(continued)"],
    "end.stopped": ["你停下了它","Stopped by you"],
    "end.maxSteps": ["步数用完了","Step limit reached"],
    "end.timeout": ["超时了","Timed out"],
    "end.noProgress": ["它绕不出来了","No further progress"],
    "end.interrupted": ["被中断了","Interrupted"],
    "end.failed": ["没做成","Not completed"],
    "replay.label": ["回放","Replay"],
    "common.dismiss": ["知道了","Dismiss"],
    "confirm.title": ["它要做一个写操作，需要你点头","This action needs your approval"],
    "confirm.deny": ["拒绝","Deny"],
    "confirm.allow": ["允许这一次","Allow once"],
    "screen.liveRead": ["实时 · 只读，不会动你的手机","Live view · Capturing only, no phone input"],
    "screen.disconnected": ["镜像没连上 —— 打开「iPhone 镜像」并让手机保持锁定","Mirror disconnected. Open iPhone Mirroring and keep your phone locked."],
    "action.stop": ["停止","Stop"],
    "action.send": ["发送","Send"],
    "state.live": ["实时","Live"],
    "state.idle": ["待机","Idle"],
    "action.resume": ["继续","Resume"],
    "action.pause": ["暂停","Pause"],
    "chat.resumePrompt": ["补一句话再继续（可留空）","Add instructions and resume (optional)"],
    "error.send": ["没发出去","Could not send"],
    "state.stopping": ["正在停止","Stopping"],
    "state.paused": ["已暂停","Paused"],
    "handover.title": ["它需要你来做一步","Your help is needed"],
    "memory.new": ["它记住了新东西","New knowledge saved"],
    "action.undo": ["撤销","Undo"],
    "error.title": ["出错了","Something went wrong"],
    "state.reconnecting": ["连接断了，正在重连…","Disconnected. Reconnecting…"],
    "state.notConfigured": ["还没配好","Setup needed"],
    "settings.openHint": ["点这里去设置","Open Settings"],
    "state.stoppingEllipsis": ["正在停止…","Stopping…"],
    "state.running": ["正在操作","Working"],
    "state.connected": ["已连接","Connected"],
    "common.loading": ["加载中…","Loading…"],
    "common.loadFailed": ["读不出来","Could not load"],
    "model.connection": ["模型连接","Model connection"],
    "settings.unsaved": ["有未保存的修改","Unsaved changes"],
    "model.notReady": ["还不能用","Not ready"],
    "settings.saveHint": ["点「保存并测试连接」生效","Choose Save and test connection to apply"],
    "model.keyMissing": ["还没填密钥","API key not configured"],
    "model.recommended": ["（推荐）"," (default)"],
    "model.custom": ["（自定义）"," (custom)"],
    "model.addEndpoint": ["＋ 添加自定义端点…","+ Add a custom endpoint…"],
    "model.namePlaceholder": ["起个名字，比如 myvllm","Choose a name, e.g. myvllm"],
    "model.idPlaceholder": ["这家服务商的模型名，原样填","Enter this provider’s exact model ID"],
    "field.name": ["名称","Name"],
    "field.baseUrl": ["接口地址","Base URL"],
    "field.provider": ["服务商","Provider"],
    "field.model": ["模型 id","Model ID"],
    "field.key": ["密钥","API key"],
    "model.clearKey": ["清除密钥","Clear key"],
    "model.saveTest": ["保存并测试连接","Save and test connection"],
    "model.savedKey": ["已保存 —— 留空表示不改","Saved — leave blank to keep"],
    "model.calibrated": ["此模型配置已有坐标标定；具体端点和任务仍需验证。","This profile is coordinate-calibrated; endpoint and task compatibility still need verification."],
    "model.uncalibrated": ["未标定坐标，默认仅按元素编号点击；图片和工具兼容性仍需验证。","Not coordinate-calibrated: element-ID taps by default. Image and tool compatibility still need verification."],
    "model.enterProvider": ["先填服务商名字。","Enter a provider name first."],
    "model.enterId": ["先填模型 id。","Enter a model ID first."],
    "model.enterBase": ["自定义端点必须填接口地址。","A custom endpoint requires a base URL."],
    "model.saving": ["正在保存并测试…","Saving and testing…"],
    "model.savedNotReady": ["存下了，但还不能用。","Saved, but the model is not ready."],
    "model.testOk": ["通了","Connection successful"],
    "model.testFailed": ["连不上","Connection failed"],
    "model.keyCleared": ["已清除 —— 会退回用环境变量","Key cleared — environment variables may still supply a key"],
    "settings.permissions": ["连接与权限","Connection and permissions"],
    "settings.checking": ["正在检查…","Checking…"],
    "settings.grant": ["去授权","Open permissions"],
    "knowledge.empty": ["还什么都没学到。跑几个任务之后，它会自己把 App 的用法、验证过的操作步骤和跨任务的经验沉淀在这里。","No knowledge yet. Task runs can produce app knowledge, reusable procedures and cross-task memories."],
    "knowledge.noProcedures": ["还没有验证过的操作步骤","No procedures listed yet"],
    "knowledge.approval": ["写类 · 待你批准","Write · review required"],
    "knowledge.read": ["读类 · 需满足验证条件","Read · eligibility depends on verification"],
    "knowledge.approved": ["已批准","Approved"],
    "memory.title": ["跨任务记忆","Cross-task memory"],
    "action.delete": ["删除","Delete"],
    "memory.local": ["记忆是纯文本文件，就放在你的电脑上，可以直接改也可以直接删。","Memories are local text files that you can edit or delete."],
    "memory.inject": ["注入给模型的时候会明确标成","When sent to the model, they are labeled as "],
    "memory.data": ["「数据，不是指令」","“data, not instructions”"],
    "console.alibaba": ["阿里云百炼控制台","Alibaba Cloud Model Studio"],
    "console.deepseek": ["DeepSeek 开放平台","DeepSeek Platform"],
    "console.moonshot": ["Moonshot 开放平台","Moonshot Platform"],
    "console.zhipu": ["智谱 AI 开放平台","Zhipu AI Platform"],
    "field.apiKey": ["API 密钥","API key"],
    "model.enterKey": ["还没填密钥。","Enter an API key first."],
    "model.notSet": ["还没配模型","No model configured"],
    "model.setupLater": ["没有模型它没法自己判断该点哪。想配的时候来这里。","Configure a model before asking the agent to operate your phone."],
    "settings.open": ["去设置","Open Settings"],
    "screen.expand": ["展开手机画面","Expand phone view"],
    "screen.collapse": ["收起手机画面","Collapse phone view"],
    "run.current": ["本次执行","Current run"],
    "step.label": ["这一步","Step"],
    "step.tool": ["工具","Tool"],
    "step.args": ["参数","Arguments"],
    "step.changed": ["画面变化","Screen changed"],
    "step.rejected": ["动作被拒","Action rejected"],
    "common.yes": ["有","Yes"],
    "common.no": ["无","No"],
    "step.execTime": ["执行耗时","Execution time"],
    "step.modelTime": ["模型思考","Model latency"],
    "step.frame": ["帧","Frame"],
    "step.error": ["被拒原因","Rejection reason"],
    "step.reason": ["它当时想的","Agent’s explanation"],
    "replay.restart": ["↺ 重播","↺ Replay"],
    "replay.pause": ["❚❚ 暂停","❚❚ Pause"],
    "replay.play": ["▶ 播放","▶ Play"],
    "run.noTask": ["（无任务）","(no task)"],
    "run.openHint": ["\n点开回放这次运行","\nOpen run replay"],
    "run.unfinished": ["没跑完","Unfinished"],
    "nav.chat": ["对话","Chat"],
    "nav.knowledge": ["知识","Knowledge"],
    "nav.settings": ["设置","Settings"],
    "nav.recent": ["最近","Recent"],
    "detail.title": ["显示每一步的原始参数","Show raw step arguments"],
    "detail.label": ["细节","Details"],
    "theme.title": ["切换外观：跟随系统 / 浅色 / 深色","Cycle appearance: system / light / dark"],
    "state.connecting": ["连接中…","Connecting…"],
    "chat.hint": ["回车发送 · Shift + 回车换行 · 本机执行，屏幕与任务内容可能发送给所选模型","Enter to send · Shift + Enter for a new line · Runs locally; screen and task content may be sent to your model"],
    "screen.title": ["手机当前画面","Current phone screen"],
    "screen.zoom": ["点开放大","Click to enlarge"],
    "screen.alt": ["手机画面","Phone screen"],
    "knowledge.title": ["它学会了什么","What it has learned"],
    "knowledge.subtitle": ["这里显示从任务中积累的 App 知识、操作步骤和跨任务记忆。","App knowledge, procedures and memories accumulated from tasks."],
    "screen.zoomAlt": ["放大的手机画面","Enlarged phone screen"],
    "zoom.closeHint": ["按 ESC 或点空白处关闭","Press Escape or click outside to close"],
    "setup.title": ["先接上一个模型","Connect a model"],
    "setup.intro": ["需要你自己的模型 API 密钥。屏幕和任务内容可能发送给所选模型；连接测试也会发出一次模型请求。","Use your own model API key. Screen and task content may be sent to that model. Testing the connection also makes a model request."],
    "setup.later": ["稍后再说","Set up later"],
    "setup.go": ["测试并开始","Test and start"],
    "action.tap": ["点击{target}","Tap {target}"],
    "action.tapIcon": ["点击{target}的图标","Tap the icon for {target}"],
    "action.tapRight": ["点击{target}这一行的右端","Tap the right side of the {target} row"],
    "action.tapLeft": ["点击{target}左侧的图标","Tap the icon to the left of {target}"],
    "action.tapId": ["点击 id {id}","Tap element {id}"],
    "action.tapCoord": ["按坐标点击","Tap at coordinates"],
    "action.openApp": ["打开{app}","Open {app}"],
    "dir.up": ["向上","up"],
    "dir.down": ["向下","down"],
    "dir.left": ["向左","left"],
    "dir.right": ["向右","right"],
    "action.page": ["{direction}翻一页","Swipe {direction} one page"],
    "action.scroll": ["{direction}滚动{amount}","Scroll {direction} {amount}"],
    "amount.half": ["半屏","half a screen"],
    "amount.full": ["一屏","one screen"],
    "action.until": ["{direction}滚动，直到出现{text}","Scroll {direction} until {text} appears"],
    "action.collect": ["滚到底，通读整个列表","Read the entire list"],
    "action.type": ["输入{text}","Type {text}"],
    "action.home": ["回到主屏幕","Go to the home screen"],
    "action.switcher": ["打开多任务","Open the app switcher"],
    "action.spotlight": ["打开 Spotlight","Open Spotlight"],
    "action.ime": ["切换输入法","Switch input method"],
    "action.wait": ["等待 {seconds} 秒","Wait {seconds}s"],
    "action.observe": ["重新观察画面","Observe the screen"],
    "action.recall": ["读取记忆{name}","Read memory {name}"],
    "action.start": ["开始（发出任务时的画面）","Start (screen when the task was sent)"],
    "chat.newChat": ["新对话","New chat"],
    "action.recallRuns": ["查看以前的运行记录","Look up past runs"],
    "action.done": ["完成，给出结果","Finish and report the result"],
    "action.failed": ["放弃，说明原因","Stop and explain the failure"],
    "group.pages": ["{direction}翻了 {count} 页","Swiped {direction} · {count} pages"],
    "group.screens": ["{direction}滚动 {count} 屏","Scrolled {direction} · {count} screens"],
    "group.wait": ["等待 {count} 次","Wait actions: {count}"],
    "group.observe": ["重新观察 {count} 次","Observations: {count}"],
    "step.number": ["第 {count} 步","Step {count}"],
    "step.count": ["{count} 步","Steps: {count}"],
    "step.unchangedCount": ["{count} 次没反应","No change: {count}"],
    "step.rejectedCount": ["{count} 次被拒","Rejected: {count}"],
    "time.minutes": ["{minutes} 分 {seconds} 秒","{minutes}m {seconds}s"],
    "time.seconds": ["{seconds} 秒","{seconds}s"],
    "confirm.timeout": ["（{seconds} 秒内不回答就当拒绝）"," (No response within {seconds}s means deny)"],
    "chat.added": ["你补了一句：{text}","You added: {text}"],
    "memory.undone": ["已撤销：{name}","Undone: {name}"],
    "memory.notFound": ["没找到：{name}","Not found: {name}"],
    "model.envOverride": ["⚠ 现在生效的是环境变量 {name}，它的优先级比这里高。要使用这里保存的密钥，先取消那个环境变量。","⚠ {name} overrides the saved key. Unset that environment variable to use the key saved here."],
    "model.keyWhere": ["在 {console} 创建","Create a key in {console}"],
    "model.keyEnv": ["也可以用环境变量 {names}","Alternatively, use {names}"],
    "error.request": ["请求失败：{error}","Request failed: {error}"],
    "knowledge.verifiedCount": ["验证过 {count} 次","Verified runs: {count}"],
    "memory.successCount": ["用成过 {count} 次","Successful uses: {count}"],
    "date.monthDay": ["{month}月{day}日","{month}/{day}"],
    "language.label": ["界面语言","Interface language"],
    "diagnostic.raw": ["原始诊断：{text}","Original diagnostic: {text}"],
    "diagnostic.failed": ["操作失败，请检查连接与权限；原始诊断见下方。","The operation failed. Check connection and permissions; the original diagnostic is shown below."],
    "event.stopping": ["正在停止，当前动作完成后结束。","Stopping after the current action finishes."],
    "event.paused": ["已请求暂停，当前动作可能先完成。可在下方补充说明后继续。","Pause requested. The current action may finish first. Add instructions below to resume."],
    "event.busy": ["上一轮还在运行。一台手机同时只能执行一个任务。","Another task is running. One phone can run only one task at a time."],
    "event.handover": ["需要你操作：{need}。完成后点「继续」，可补充说明。","Your help is needed: {need}. Choose Resume when finished; additional instructions are optional."],
    "event.error": ["运行出错。原始诊断：{detail}","Run failed. Original diagnostic: {detail}"],
    "confirm.tap": ["即将点击「{target}」。Agent 给出的理由：{reason}","About to tap “{target}”. Agent’s explanation: {reason}"],
    "confirm.legacy": ["请检查以下操作说明，再决定是否允许：{detail}","Review the following action description before allowing it: {detail}"],
    "api.invalidBody": ["请求体必须是对象。","The request body must be an object."],
    "api.busy": ["有任务正在运行，停止后再修改设置。","Stop the current task before changing settings."],
    "api.providerMissing": ["请选择服务商。","Choose a provider."],
    "api.runMissing": ["没有这次运行。","Run not found."],
    "api.newChatBusy": ["有任务正在运行，先停下来再开新对话。","Stop the current task before starting a new chat."],
    "config.invalid": ["配置无效。原始诊断：{detail}","Invalid configuration. Original diagnostic: {detail}"],
    "config.permissions": ["配置文件权限过宽。请运行 chmod 600 {path}。","Configuration permissions are too broad. Run chmod 600 {path}."],
    "config.missingKey": ["没有 {provider} 的 API 密钥。请在设置中填写，或使用环境变量 {env}。","No API key for {provider}. Enter it in Settings or use {env}."],
    "config.keyCharacters": ["密钥含非 ASCII 字符。请填写真实 API 密钥。","The key contains non-ASCII characters. Enter a valid API key."],
    "config.providerName": ["服务商名字只能使用小写字母、数字和 _ - .。","Provider names may contain lowercase letters, digits and _ - . only."],
    "config.baseScheme": ["接口地址必须以 http:// 或 https:// 开头。","The base URL must start with http:// or https://."],
    "config.baseCharacters": ["接口地址包含空格或非 ASCII 字符。","The base URL contains whitespace or non-ASCII characters."],
    "config.baseLength": ["接口地址过长。","The base URL is too long."],
    "config.emptyModel": ["请填写模型标识。","Enter a model identifier."],
    "config.modelFormat": ["模型标识格式应为 provider:model。","Use provider:model for the model identifier."],
    "model.testSuccess": ["连接测试成功。该测试不验证图片、工具或真机操作兼容性。","Connection test passed. This does not verify image, tool or phone-task compatibility."],
    "model.auth": ["密钥无效或已过期，请检查后重试。","The API key is invalid or expired. Check it and retry."],
    "model.forbidden": ["此密钥无权调用所选模型。","This key cannot access the selected model."],
    "model.endpoint": ["检查模型 ID 和端点地址是否正确。","Check the model ID and endpoint URL."],
    "model.rateLimit": ["请求受到限流，请稍后重试。","Rate limited. Retry later."],
    "model.timeout": ["连接超时，请检查网络、代理或端点。","Connection timed out. Check your network, proxy and endpoint."],
    "model.tls": ["TLS 连接失败，请检查代理和证书。","TLS connection failed. Check your proxy and certificates."],
    "model.error": ["模型请求失败。原始诊断：{detail}","Model request failed. Original diagnostic: {detail}"],
    "check.perm.accessibility.title": ["辅助功能权限","Accessibility permission"],
    "check.perm.screen_recording.title": ["屏幕录制权限","Screen Recording permission"],
    "check.mirror.window.title": ["iPhone 镜像","iPhone Mirroring"],
    "check.mirror.capture.title": ["抓帧","Screen capture"],
    "check.perceive.ocr.title": ["文字识别","Text recognition"],
    "check.model.title": ["模型","Model"],
    "check.injector.title": ["注入路径","Input path"],
    "check.granted": ["已授权","Granted"],
    "check.denied": ["未授权","Not granted"],
    "check.accessibility.why": ["用于将点击与键盘输入发送到镜像窗口。","Needed to send pointer and keyboard input to the mirror."],
    "check.accessibility.fix": ["在系统设置 → 隐私与安全性 → 辅助功能中授权当前运行程序。","Allow the hosting application in System Settings → Privacy & Security → Accessibility."],
    "check.screen.why": ["用于读取手机画面。","Needed to capture the phone screen."],
    "check.screen.fix": ["在系统设置 → 隐私与安全性 → 屏幕录制中授权当前运行程序。","Allow the hosting application in System Settings → Privacy & Security → Screen Recording."],
    "check.window.detail": ["已找到镜像窗口。{detail}","Mirror window found. {detail}"],
    "check.window.why": ["所有手机操作都发送到这个镜像窗口。","Phone input is sent to this mirror window."],
    "check.capture.why": ["抓帧提供当前屏幕证据。","Captured frames provide current screen evidence."],
    "check.capture.fix": ["确认镜像窗口未最小化，并检查屏幕录制权限。","Keep the mirror visible and check Screen Recording permission."],
    "check.ocr.detail": ["识别到 {count} 个元素。","Recognized elements: {count}."],
    "check.ocr.why": ["OCR 提供屏幕上的文字标签。","OCR provides visible text labels."],
    "check.ocr.fix": ["当前画面可能没有文字。换到含文字的页面后重试。","The screen may contain no text. Try a screen with visible labels."],
    "check.model.why": ["模型负责选择下一步动作。","The model selects the next action."],
    "check.model.fix": ["在设置中选择模型并配置 API 密钥。","Choose a model and configure its API key in Settings."],
    "check.model.detail": ["{spec} · 密钥来源：{source} · 坐标点击：{coord}","{spec} · Key source: {source} · Coordinate taps: {coord}"],
    "check.injector.why": ["常规后台输入尽量不占用 Mac 的光标与焦点。","Normal background input aims to leave the Mac cursor and focus available."],
    "check.injector.detail": ["输入路径：{mode} · SkyLight：{available}","Input path: {mode} · SkyLight: {available}"],
    "check.injector.fix": ["只读检查无法确认输入是否生效，需要监督式真机验证。","Read-only checks cannot verify input delivery. A supervised device check is required."],
    "check.failed": ["此项检查失败。原始诊断：{detail}","This check failed. Original diagnostic: {detail}"],
    "check.failedFix": ["先检查镜像连接与前面的失败项，然后重试。","Check the mirror and earlier failed checks, then retry."]
  };
  const tag = Symbol('ui-message');
  const isMessage = value => !!(value && value[tag]);
  const resolve = value => isMessage(value) ? value.render() : String(value == null ? '' : value);
  const lazy = render => ({ [tag]: true, render, toString: render });
  let locale = 'en';
  try {
    const saved = localStorage.getItem('iphone-agent.locale');
    const preferred = saved || (navigator.languages || [navigator.language])[0] || 'en';
    locale = /^zh(?:-|$)/i.test(preferred) ? 'zh-CN' : 'en';
  } catch {
    locale = /^zh(?:-|$)/i.test(navigator.language || '') ? 'zh-CN' : 'en';
  }
  const msg = (key, params = {}) => lazy(() => {
    const pair = catalog[key];
    if (!pair) return key;
    const template = pair[locale === 'zh-CN' ? 0 : 1];
    // One substitution pass: parameter content is data, never a new template.
    return template.replace(/\{(\w+)\}/g, (match, name) =>
      Object.prototype.hasOwnProperty.call(params, name) ? resolve(params[name]) : match);
  });
  const message = (descriptor, raw = '') => descriptor && catalog[descriptor.code]
    ? msg(descriptor.code, descriptor.params || {}) : raw;
  const join = (parts, separator = ',') => {
    if (!parts.some(isMessage) && !isMessage(separator)) return parts.join(separator);
    return lazy(() => parts.map(resolve).join(resolve(separator)));
  };
  const cat = (...parts) => join(parts, '');
  const bindings = new Map();
  let cleanupPending = false;
  function pruneSoon() {
    if (cleanupPending) return;
    cleanupPending = true;
    queueMicrotask(() => {
      cleanupPending = false;
      for (const node of bindings.keys()) if (!node.isConnected) bindings.delete(node);
    });
  }
  function bind(node, property, value) {
    if (isMessage(value)) {
      if (!bindings.has(node)) bindings.set(node, new Map());
      bindings.get(node).set(property, value);
    } else if (bindings.has(node)) {
      bindings.get(node).delete(property);
      if (!bindings.get(node).size) bindings.delete(node);
    }
    node[property] = resolve(value);
    pruneSoon();
    return value;
  }
  const text = (node, value) => bind(node, 'textContent', value);
  const textNode = value => { const node = document.createTextNode(''); text(node, value); return node; };
  function setLocale(next) {
    locale = next === 'zh-CN' ? 'zh-CN' : 'en';
    try { localStorage.setItem('iphone-agent.locale', locale); } catch { /* private browsing */ }
    document.documentElement.lang = locale;
    for (const [node, properties] of bindings) {
      if (!node.isConnected) { bindings.delete(node); continue; }
      for (const [property, value] of properties) node[property] = resolve(value);
    }
    document.querySelectorAll('[data-language-select]').forEach(select => { select.value = locale; });
  }
  function init() {
    document.querySelectorAll('[data-i18n]').forEach(node => text(node, msg(node.dataset.i18n)));
    for (const property of ['title', 'placeholder', 'alt', 'aria-label']) {
      document.querySelectorAll('[data-i18n-' + property + ']').forEach(node => {
        const value = msg(node.getAttribute('data-i18n-' + property));
        if (property === 'aria-label') {
          // ariaLabel is the reflected property; do not replace the input's current value.
          bind(node, 'ariaLabel', value);
        } else bind(node, property, value);
      });
    }
    document.querySelectorAll('[data-language-select]').forEach(select => {
      select.value = locale;
      select.onchange = () => setLocale(select.value);
    });
    document.documentElement.lang = locale;
  }
  return { msg, message, bind, text, textNode, cat, join, isMessage, setLocale, init,
           get locale() { return locale; }, keys: () => Object.keys(catalog) };
})();
const L = I18n.msg, uiText = I18n.text, uiNode = I18n.textNode, uiCat = I18n.cat, uiJoin = I18n.join;
