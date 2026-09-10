# Languages / 语言支持

## What is bilingual

The local web interface supports **English** and **Simplified Chinese**. Choose the language in the sidebar or the first-run model setup dialog. A saved browser preference takes precedence; otherwise Chinese browser locales use Simplified Chinese and other locales use English.

Switching languages does not reload the page, submit a task, answer a confirmation, or save model settings. Unsaved input values, active confirmation controls and the existing conversation remain in place. Language is a browser preference, not a global setting on the phone-control session.

本机 Web 已提供中英文切换，入口在侧栏和首次配置弹窗。切换只改变界面显示，不会操作手机，也不会清空正在填写的模型配置。

## Three separate language layers

| Layer | Behavior |
|---|---|
| Web interface | Localized labels, guidance, status, common errors and diagnostic explanations. |
| Agent and user content | Original task text, model responses, explanations and skill prose are preserved. Ask explicitly for a preferred answer language when needed. |
| Phone interface | OCR labels and device content stay in the phone's own language. Changing the Web language does not change iOS or App settings. |

Old run records and uncommon diagnostics may still contain their original language. Known backend messages carry stable codes and parameters so each browser can render them independently. Unrecognized diagnostics are not machine-translated or silently rewritten.

原始记录、模型输出和部分底层诊断可能仍是中文。界面双语不等于历史内容被自动翻译，也不等于手机系统语言随之改变。

## Device-language compatibility

OCR is configured for Simplified Chinese and English. Home-screen and Spotlight heuristics include Chinese and English markers, and skill risk keyword checks cover both languages.

These are narrow compatibility improvements, not a guarantee that every English-language screen or task works. Heuristic recognition and keyword safety checks still have blind spots. Supervised real-device acceptance across languages remains a release gate; synthetic regression checks are not device acceptance.

Phone UI labels, app versions, layouts and model behavior all affect results. Report them with any reproducible failure.

## Tests and contribution

`tests/test_i18n.py` covers language-related Python behavior and invokes dependency-free JavaScript tests when Node.js is available. Run those directly with:

```sh
node --test tests/web_i18n.test.cjs
```

An optional browser smoke test uses Playwright and a separately launched Chrome instance. All page resources and API requests are intercepted with synthetic fixtures: it does not start `iphone serve`, connect to a phone, or call a model.

With Playwright available to Node and Google Chrome installed:

```sh
node scripts/check_web_i18n.cjs
```

For a separate developer dependency directory:

```sh
task_browser_deps=$(mktemp -d)
npm install --prefix "$task_browser_deps" playwright
NODE_PATH="$task_browser_deps/node_modules" node scripts/check_web_i18n.cjs
```

Playwright is a development check, not a product dependency or a frontend build step. The smoke test checks first-run English, language persistence, switching with unsaved configuration, raw text preservation, confirmation labels, narrow-window overflow and configuration-error guidance.

When adding a UI message, supply both languages with matching named placeholders in `iphone_agent/web/static/i18n.js`. Bind it at an explicit presentation boundary. Never translate arbitrary DOM text, OCR, action arguments, model output or stored evidence.
