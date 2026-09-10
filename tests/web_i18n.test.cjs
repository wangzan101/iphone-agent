/* Run with node --test tests/web_i18n.test.cjs; no npm packages required. */
const assert = require('node:assert/strict');
const test = require('node:test');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const root = path.join(__dirname, '../iphone_agent/web/static');
const source = fs.readFileSync(path.join(root, 'i18n.js'), 'utf8');

function runtime(language='en-US', saved=null, unavailable=false) {
  const values = new Map(saved ? [['iphone-agent.locale', saved]] : []);
  const document = {documentElement:{lang:''},querySelectorAll:()=>[],
    createTextNode:()=>({textContent:'',isConnected:true})};
  const ctx = vm.createContext({document,navigator:{language,languages:[language]},queueMicrotask,
    localStorage:{getItem:k=>{if(unavailable)throw Error();return values.get(k);},
      setItem:(k,v)=>{if(unavailable)throw Error();values.set(k,v);}}});
  vm.runInContext(source+'\nglobalThis.ui = I18n;',ctx);
  return {ui:ctx.ui,document,values};
}
test('saved preference, browser language, English fallback and storage failure',()=>{
  assert.equal(runtime('en-US','zh-CN').ui.locale,'zh-CN');
  assert.equal(runtime('zh-TW').ui.locale,'zh-CN');
  assert.equal(runtime('fr-FR').ui.locale,'en');
  assert.equal(runtime('zh-CN',null,true).ui.locale,'zh-CN');
});
test('bound labels change without replacing user data or input values',()=>{
  const {ui}=runtime();
  const label={isConnected:true}, input={isConnected:true,value:'原始密钥草稿'}, raw={isConnected:true};
  ui.text(label,ui.msg('nav.chat'));
  ui.bind(input,'placeholder',ui.msg('chat.prompt'));
  ui.text(raw,'对话');
  ui.setLocale('zh-CN');
  assert.equal(label.textContent,'对话');
  assert.equal(input.value,'原始密钥草稿');
  ui.setLocale('en');
  assert.equal(label.textContent,'Chat');
  assert.equal(raw.textContent,'对话');
  assert.equal(input.value,'原始密钥草稿');
});
test('replacing a localized value with raw text removes its old binding',()=>{
  const {ui}=runtime(),node={isConnected:true};
  ui.text(node,ui.msg('nav.chat'));
  ui.text(node,'设置');
  ui.setLocale('zh-CN');ui.setLocale('en');
  assert.equal(node.textContent,'设置');
});
test('nested messages and joined counts stay reactive',()=>{
  const {ui}=runtime(), node={isConnected:true};
  ui.text(node,ui.join([ui.msg('action.scroll',{direction:ui.msg('dir.up'),amount:ui.msg('amount.half')}),
    ui.msg('step.count',{count:2})],' · '));
  assert.equal(node.textContent,'Scroll up half a screen · Steps: 2');
  ui.setLocale('zh-CN');
  assert.equal(node.textContent,'向上滚动半屏 · 2 步');
});
test('message parameters are data, not HTML or recursive templates',()=>{
  const {ui}=runtime(),node={isConnected:true};
  ui.text(node,ui.msg('action.tap',{target:'<img onerror=alert(1)> {target} 设置'}));
  assert.equal(node.textContent,'Tap <img onerror=alert(1)> {target} 设置');
  assert.equal(node.innerHTML,undefined);
});
test('two clients localize the same event independently',()=>{
  const a=runtime('en-US').ui,b=runtime('zh-CN').ui;
  const event={code:'confirm.tap',params:{target:'发送',reason:'原始理由'}};
  assert.match(String(a.message(event)),/About to tap/);
  assert.match(String(b.message(event)),/即将点击/);
  assert.equal(event.params.target,'发送');
});
test('unknown descriptors preserve the original diagnostic',()=>{
  const {ui}=runtime();
  assert.equal(ui.message({code:'future.error'},'原始诊断'),'原始诊断');
});
test('every catalog entry has both languages and matching interpolation parameters',()=>{
  const {ui}=runtime();
  for(const key of ui.keys()){
    ui.setLocale('en');const en=String(ui.msg(key));
    ui.setLocale('zh-CN');const zh=String(ui.msg(key));
    assert.notEqual(en,key,key);assert.notEqual(zh,key,key);
    assert.deepEqual((en.match(/\{\w+\}/g)||[]).sort(),(zh.match(/\{\w+\}/g)||[]).sort(),key);
  }
});
test('static and literal message references resolve',()=>{
  const {ui}=runtime(),keys=new Set(ui.keys());
  const html=fs.readFileSync(path.join(root,'index.html'),'utf8');
  const app=fs.readFileSync(path.join(root,'app.js'),'utf8');
  for(const match of app.matchAll(/\bL\(['"]([^'"]+)['"]/g))assert.ok(keys.has(match[1]),match[1]);
  for(const match of html.matchAll(/data-i18n(?:-[\w-]+)?="([^"]+)"/g))assert.ok(keys.has(match[1]),match[1]);
  assert.match(html,/src="i18n.js"/);
  assert.doesNotMatch(app,/打开设置里的飞行模式/);
});
