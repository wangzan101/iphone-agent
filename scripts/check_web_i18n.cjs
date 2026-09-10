const fs = require('fs');
const assert = require('node:assert/strict');
const path = require('node:path');
// Optional developer check; playwright is not a runtime dependency of the product.
const {chromium} = require('playwright');
let browser;
(async () => {
  browser = await chromium.launch({headless:true,channel:process.env.IPHONE_BROWSER_CHANNEL || 'chrome'});
  const page = await browser.newPage({locale:'en-US',viewport:{width:1280,height:900}});
  const errors=[], writes=[];
  page.on('pageerror', e=>errors.push(e.message));
  const root=process.argv[2] || path.join(__dirname, '../iphone_agent/web/static'), out=process.argv[3];
  let badConfig = false;
  const config={current:{spec:null,problem:'没有密钥',problem_message:{code:'config.missingKey',params:{provider:'alibaba',env:'DASHSCOPE_API_KEY'}}},providers:{}};
  const models={default:'alibaba:qwen3.7-plus',models:[{spec:'alibaba:qwen3.7-plus',calibrated:true}],providers:[{name:'alibaba',env_vars:['DASHSCOPE_API_KEY']},{name:'openai',env_vars:['OPENAI_API_KEY']}]};
  await page.route('**/*',async route=>{
    const url=new URL(route.request().url()), path=url.pathname;
    if(route.request().method()==='POST') writes.push(path);
    const mocks={
      '/api/state':{running:false,paused:false,model:{ready:false}},
      '/api/models':models,'/api/config':config,
      '/api/runs':{runs:[{id:'20000101-000000-abcd',task:'用户原始任务：设置',steps:3,started_at:1,finished:true,ok:true}]},
      '/api/learned':{apps:[],memories:[]},
      '/api/doctor':{checks:[{id:'perm.accessibility',status:'fail',title:'辅助功能权限',detail:'未授权',fix:'请授权',fix_url:'x-apple.systempreferences:test',messages:{title:{code:'check.perm.accessibility.title'},detail:{code:'check.denied'},fix:{code:'check.accessibility.fix'}}}]}
    };
    if(path==='/api/stream')return route.fulfill({status:200,contentType:'text/event-stream',body:': mock\n\n'});
    if(path.startsWith('/api/screen'))return route.fulfill({status:404,body:'no mirror'});
    if(path === '/api/config' && badConfig) return route.fulfill({status:400,json:{error:'配置权限太宽',error_message:{code:'config.permissions',params:{path:'.iphone/config.toml'}}}});
    if(mocks[path]) return route.fulfill({json:mocks[path]});
    const name=path==='/'?'index.html':path.slice(1);
    if(!['index.html','app.js','app.css','i18n.js'].includes(name))return route.fulfill({status:404,body:'not found'});
    return route.fulfill({contentType:name.endsWith('.js')?'text/javascript':name.endsWith('.css')?'text/css':'text/html',body:fs.readFileSync(root+'/'+name)});
  });
  await page.goto('http://127.0.0.1:18765/');
  await page.locator('#setup').waitFor({state:'visible'});
  assert.equal(await page.locator('html').getAttribute('lang'),'en');
  assert.equal(await page.locator('#setup h2').innerText(),'Connect a model');
  await page.locator('#setupBody input[type=password]').fill('local-draft-not-a-real-key');
  await page.locator('#setup [data-language-select]').selectOption('zh-CN');
  assert.equal(await page.locator('#setup h2').innerText(),'先接上一个模型');
  assert.equal(await page.locator('#setupBody input[type=password]').inputValue(),'local-draft-not-a-real-key');
  await page.locator('#setup [data-language-select]').selectOption('en');
  await page.locator('#setupLater').click();
  await page.locator('[data-nav=settings]').first().click();
  await page.locator('#settingsBody input[type=password]').fill('settings-draft');
  await page.locator('#sidebar [data-language-select]').selectOption('zh-CN');
  assert.equal(await page.locator('#settingsBody input[type=password]').inputValue(),'settings-draft');
  await page.locator('#sidebar [data-language-select]').selectOption('en');
  assert.match(await page.locator('#settingsBody').innerText(),/Accessibility permission/);
  await page.locator('[data-nav=chat]').first().click();
  await page.evaluate(()=>{
    startTurn('不要翻译：设置');
    const step={n:1,name:'open_app',args:{name:'设置'},reason:'原始理由',changed:true};
    addStep(step);turn._steps.push(step);updateLive(turn,step);
    confirmBanner(L('confirm.tap',{target:'发送',reason:'原始理由'}),60);
  });
  assert.match(await page.locator('#log').innerText(),/Open 「设置」/);
  await page.locator('#sidebar [data-language-select]').selectOption('zh-CN');
  assert.match(await page.locator('#log').innerText(),/打开「设置」/);
  assert.match(await page.locator('#log').innerText(),/不要翻译：设置/);
  await page.locator('#sidebar [data-language-select]').selectOption('en');
  assert.match(await page.locator('#banners').innerText(),/Allow once/);
  assert.match(await page.locator('#banners').innerText(),/原始理由/);
  assert.equal(writes.length,0,'Language switches must not write config, send tasks or answer confirmations');
  if(out) await page.screenshot({path:out+'/web-en.png',fullPage:true,animations:'disabled'});
  await page.locator('#sidebar [data-language-select]').selectOption('zh-CN');
  if(out) await page.screenshot({path:out+'/web-zh.png',fullPage:true,animations:'disabled'});
  await page.reload();
  await page.locator('#setup').waitFor({state:'visible'});
  assert.equal(await page.locator('html').getAttribute('lang'),'zh-CN');
  await page.locator('#setup [data-language-select]').selectOption('en');
  await page.setViewportSize({width:760,height:900});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth <= innerWidth),true);
  badConfig=true;
  await page.reload();
  await page.locator('#setup').waitFor({state:'visible'});
  assert.match(await page.locator('#setupBody').innerText(),/Configuration permissions are too broad/);
  assert.equal(await page.locator('#setupGo').isDisabled(),true);
  assert.deepEqual(errors,[]);
  console.log('PASS: first-run English; live bilingual UI; setup/settings drafts and source text preserved; no mutation requests.');
})().catch(e=>{console.error(e);process.exitCode=1;}).finally(async()=>{if(browser)await browser.close();});
