// Usage: node tests/test_nn_view_clipboard.mjs GENERATED.nn.html
// Real offline Chrome page; exercise all clipboard paths without touching the
// user's system clipboard. Node 22+ and Chrome, or CHROME=/path/to/chrome.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { setTimeout as sleep } from 'node:timers/promises';

assert.ok(process.argv[2], 'Pass a generated NN microscope HTML report');
assert.ok(globalThis.WebSocket, 'Use Node 22+, or node --experimental-websocket on Node 20');
const profile = mkdtempSync(join(tmpdir(), 'continugo-clipboard-chrome-'));
const chrome = spawn(process.env.CHROME || 'google-chrome', [
  '--headless', '--no-sandbox', '--disable-gpu', '--disable-background-networking',
  '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank',
], { stdio: 'ignore' });
let launchError;
chrome.on('error', error => { launchError = error; });
let ws;
let call;
try {
  let port;
  for (let i = 0; i < 600; i++) {
    if (launchError) throw launchError;
    try { port = readFileSync(join(profile, 'DevToolsActivePort'), 'utf8').split('\n')[0]; } catch {}
    if (port) break;
    await sleep(50);
  }
  assert.ok(port, 'Chrome did not start');
  const targets = await fetch(`http://localhost:${port}/json/list`).then(r => r.json());
  ws = new WebSocket(targets.find(t => t.type === 'page').webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
  let id = 0;
  const pending = new Map();
  const errors = [];
  ws.onmessage = ({ data }) => {
    const message = JSON.parse(data);
    if (message.id) {
      const { resolve, reject } = pending.get(message.id);
      pending.delete(message.id);
      message.error ? reject(Error(JSON.stringify(message.error))) : resolve(message.result);
    } else if (message.method === 'Runtime.exceptionThrown') {
      errors.push(message.params.exceptionDetails);
    }
  };
  call = (method, params = {}) => new Promise((resolve, reject) => {
    pending.set(++id, { resolve, reject });
    ws.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async expression => {
    const result = await call('Runtime.evaluate', {
      expression, awaitPromise: true, returnByValue: true,
    });
    if (result.exceptionDetails) throw Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  await call('Runtime.enable');
  await call('Page.enable');
  await call('Page.navigate', { url: pathToFileURL(resolve(process.argv[2])).href });
  let ready;
  for (let i = 0; i < 100; i++) {
    ready = await evaluate("typeof copySGF==='function' && !!document.getElementById('sgf').onclick");
    if (ready) break;
    await sleep(50);
  }
  assert.ok(ready, 'Copy button did not initialize');
  const result = await evaluate(`(async () => {
    const check=(condition,message)=>{if(!condition)throw Error(message)};
    const source=JSON.parse(document.getElementById('data').textContent).sgf;
    const expected=new TextDecoder().decode(Uint8Array.from(atob(source.split(',')[1]),c=>c.charCodeAt(0)));
    const button=document.getElementById('sgf');
    check(!document.getElementById('copystatus'),'Persistent status text must not be rendered');
    check(button.tagName==='BUTTON'&&!button.hasAttribute('download'),'Download was not replaced');
    let captured=null,legacyCalls=0;
    check(window.isSecureContext,'Offline file is not a secure context in this browser');
    Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async text=>{captured=text}}});
    document.execCommand=()=>{legacyCalls++;throw Error('Unexpected legacy fallback')};
    await copySGF();
    check(captured===expected,'Clipboard API did not receive the complete original SGF');
    check(legacyCalls===0,'Modern copy unnecessarily invoked fallback');
    check(button.textContent==='Copied ✓'&&!button.disabled,'Missing transient success feedback');
    await new Promise(resolve=>setTimeout(resolve,1300));
    check(button.textContent==='Copy SGF','Copy feedback did not reset');
    navigator.clipboard.writeText=async()=>{throw Error('Permission denied')};
    document.execCommand=command=>{
      legacyCalls++;check(command==='copy','Unexpected legacy command');
      check(document.activeElement.value===expected,'Fallback modified SGF');
      check(document.activeElement.selectionStart===0&&document.activeElement.selectionEnd===expected.length,'Fallback selection incomplete');
      return true;
    };
    await copySGF();
    check(legacyCalls===1&&button.textContent==='Copied ✓','Denied-API fallback failed');
    Object.defineProperty(navigator,'clipboard',{configurable:true,value:undefined});
    document.execCommand=()=>false;
    await copySGF();
    const dialog=document.getElementById('sgfcopy'),field=document.getElementById('sgftext');
    check(dialog.open&&field.value===expected,'Manual fallback did not expose the original SGF');
    check(field.selectionStart===0&&field.selectionEnd===expected.length,'Manual fallback selection incomplete');
    check(button.textContent==='Copy SGF','Failed copy falsely claimed success');
    const position=document.getElementById('position').textContent;
    field.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));
    check(document.getElementById('position').textContent===position,'Copy dialog navigated the game');
    document.getElementById('copyclose').click();
    check(!dialog.open&&!button.disabled,'Manual dialog failed to close');
    document.execCommand=()=>{throw Error('Copy unsupported')};
    await copySGF();
    check(dialog.open,'Throwing legacy copy did not reach manual fallback');
    dialog.close();
    check(document.querySelectorAll('textarea').length===1,'Temporary clipboard fields leaked');
    return {characters:expected.length,modern:true,legacy:true,manual:true};
  })()`);
  assert.deepEqual(errors, [], 'Browser runtime errors');
  console.log('NN microscope clipboard paths passed:', result);
} finally {
  if (call) { try { await call('Browser.close'); } catch {} }
  if (ws) ws.close();
  if (chrome.exitCode === null) chrome.kill();
  await sleep(150);
  rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
}
