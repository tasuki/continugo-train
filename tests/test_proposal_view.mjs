// Usage: node tests/test_proposal_view.mjs GENERATED/index.html [NEW_SCREENSHOT_DIR]
// Real shared viewer + overlay tests via Chrome CDP, no automation dependency.
// Node 22+; on Node 20 add --experimental-websocket.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { setTimeout as sleep } from 'node:timers/promises';

assert.ok(process.argv[2], 'Pass a generated proposal-view index.html');
assert.ok(globalThis.WebSocket, 'Use Node 22+, or node --experimental-websocket on Node 20');
const html = resolve(process.argv[2]);
const screenshots = process.argv[3] && resolve(process.argv[3]);
if (screenshots) mkdirSync(screenshots, { recursive: true });
const profile = mkdtempSync(join(tmpdir(), 'continugo-proposals-chrome-'));
const chrome = spawn(process.env.CHROME || 'google-chrome', [
  '--headless', '--no-sandbox', '--disable-gpu', '--disable-background-networking',
  '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank',
], { stdio: 'ignore' });
let launchError;
chrome.on('error', error => { launchError = error; });
let ws;
let closeBrowser;
try {
  let port;
  for (let i = 0; i < 100; i++) {
    if (launchError) throw launchError;
    try { port = readFileSync(join(profile, 'DevToolsActivePort'), 'utf8').split('\n')[0]; } catch {}
    if (port) break;
    await sleep(50);
  }
  assert.ok(port, 'Chrome did not start; set CHROME to its executable path');
  const targets = await fetch(`http://localhost:${port}/json/list`).then(r => r.json());
  ws = new WebSocket(targets.find(target => target.type === 'page').webSocketDebuggerUrl);
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
  const call = (method, params = {}) => new Promise((resolve, reject) => {
    pending.set(++id, { resolve, reject });
    ws.send(JSON.stringify({ id, method, params }));
  });
  closeBrowser = () => call('Browser.close');
  const evaluate = async expression => {
    const result = await call('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (result.exceptionDetails) throw Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  await call('Runtime.enable');
  await call('Page.enable');
  await call('Emulation.setDeviceMetricsOverride', { width: 1400, height: 1800, deviceScaleFactor: 1, mobile: false });
  await call('Page.navigate', { url: pathToFileURL(html).href });
  let ready = false;
  for (let i = 0; i < 100; i++) {
    ready = await evaluate("!!document.querySelector('#position > div')?.shadowRoot?.querySelector('#stones')");
    if (ready) break;
    await sleep(50);
  }
  assert.ok(ready, 'Shared viewer did not render');
  const result = await evaluate(`(async () => {
    const check = (condition, message) => { if (!condition) throw Error(message); };
    const frame = () => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const data = JSON.parse(document.getElementById('proposal-data').textContent);
    const host = document.querySelector('#position > div');
    const root = host.shadowRoot;
    const overlay = document.getElementById('candidate-overlay');
    const marks = document.getElementById('candidate-marks');
    const source = document.getElementById('source');
    const opacity = document.getElementById('opacity');
    const nativeStones = () => [...root.querySelectorAll('#stones circle')].map(c => [
      Number(c.getAttribute('cx')), Number(c.getAttribute('cy')),
      c.classList.contains('black') ? 'black' : 'white',
    ].join(',')).sort().join(';');
    const expected = data.position.stones.map(s => [s.x,s.y,s.color].join(',')).sort().join(';');
    check(nativeStones() === expected, 'Shared replay disagrees with authoritative Rust position');
    check(document.getElementById('error').hidden, 'Viewer error displayed');
    check(getComputedStyle(overlay).pointerEvents === 'none', 'Overlay blocks native hover');
    check(root.querySelectorAll('#policy-alternatives circle').length === 0, 'Recorded MCTS policy mixed with proposals');
    const count = () => marks.querySelectorAll('circle').length;
    check(count() === data.candidates.length, 'All proposals shown initially');
    check(marks.getAttribute('opacity') === '1', 'Dots must be opaque');
    check([...marks.children].every(c => Number(c.getAttribute('r')) === 2.15), 'Dot radii');
    const chooseMode = async mode => {
      const input = document.querySelector('input[name="mode"][value="'+mode+'"]');
      input.checked = true; input.dispatchEvent(new Event('change')); await frame();
    };
    await chooseMode('stones');
    check(marks.getAttribute('opacity') === '0.2' && !opacity.disabled, 'Default stone opacity');
    check([...marks.children].every(c => Number(c.getAttribute('r')) === data.position.radius), 'Candidate stone radii');
    check([...marks.children].every(c => c.getAttribute('fill') === (data.position.player === 'black' ? '#222' : '#fff')), 'Mover-coloured candidate stones');
    check([...marks.children].every(c => !c.hasAttribute('opacity')), 'Opacity must composite the layer, not accumulate per stone');
    for (const option of source.options) {
      source.value = option.value; source.dispatchEvent(new Event('change')); await frame();
      const mask = Number(option.value);
      const expectedIds = data.candidates.filter(c => (c.sources & mask) === mask).map(c => c.action).sort((a,b) => a-b);
      const actualIds = [...marks.children].map(c => Number(c.dataset.action)).sort((a,b) => a-b);
      check(JSON.stringify(actualIds) === JSON.stringify(expectedIds), 'Source filter '+option.textContent);
    }
    source.value = '0'; source.dispatchEvent(new Event('change'));
    opacity.value = '0.35'; opacity.dispatchEvent(new Event('input')); await frame();
    check(marks.getAttribute('opacity') === '0.35', 'Opacity control');
    check(document.getElementById('opacity-value').value === '0.35', 'Opacity output');
    await chooseMode('dots');
    check(marks.getAttribute('opacity') === '1' && opacity.disabled, 'Switching back to dots');
    const showCells = document.getElementById('show-cells');
    showCells.click(); await frame();
    check(getComputedStyle(document.getElementById('selected-cells')).display === 'none', 'Hide selected cells');
    showCells.click(); await frame();
    check(document.querySelectorAll('#selected-cells rect').length === data.cells.length, 'Selected cell count');
    const stack = document.querySelector('.board-stack');
    for (const size of [420, 800]) {
      stack.style.width = size+'px'; stack.style.transform = 'translate(17px,23px)'; await frame();
      const a = root.getElementById('board').getBoundingClientRect();
      const b = overlay.getBoundingClientRect();
      check(['left','top','width','height'].every(k => Math.abs(a[k]-b[k])<0.5), 'Overlay alignment after resize/offset');
      check(Math.abs(a.width-size)<0.5 && Math.abs(a.height-size)<0.5, 'Responsive square');
    }
    stack.style.width = ''; stack.style.transform = ''; await frame();
    if (data.candidates.length) {
      const candidate = data.candidates[0];
      const r = root.getElementById('board').getBoundingClientRect();
      root.getElementById('board').dispatchEvent(new PointerEvent('pointermove', {
        bubbles: true, composed: true, pointerType: 'mouse',
        clientX: r.left + (candidate.action%1000)/1000*r.width,
        clientY: r.top + Math.floor(candidate.action/1000)/1000*r.height,
      }));
      await frame();
      check(root.querySelector('#ghost-stone circle')?.getAttribute('cx') === String(candidate.action%1000), 'Native placement hover survives overlay');
      host.dispatchEvent(new PointerEvent('pointerleave')); await frame();
    }
    check(nativeStones() === expected, 'Display toggles must not change position');
    return { move: data.move, player: data.position.player, legal: data.candidates.length };
  })()`);
  if (screenshots) {
    const screenshot = async (name, mode, mask) => {
      await evaluate(`(async () => {
        const frame=()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
        const source=document.getElementById('source'); source.value='${mask}'; source.dispatchEvent(new Event('change'));
        const input=document.querySelector('input[name="mode"][value="${mode}"]'); input.checked=true; input.dispatchEvent(new Event('change'));
        const opacity=document.getElementById('opacity'); opacity.value='0.2'; opacity.dispatchEvent(new Event('input'));
        await frame();
      })()`);
      const { data } = await call('Page.captureScreenshot', { format: 'png' });
      writeFileSync(join(screenshots, name+'.png'), Buffer.from(data, 'base64'), { flag: 'wx' });
    };
    await screenshot('dots', 'dots', 0);
    await screenshot('stones', 'stones', 0);
    await screenshot('connected-hug-stones', 'stones', 5);
  }
  assert.deepEqual(errors, [], 'No browser runtime errors');
  console.log('Proposal viewer browser tests passed:', result);
} finally {
  // Let Chrome shut down its children and flush the profile before removal.
  if (closeBrowser && ws?.readyState === 1) {
    await Promise.race([closeBrowser().catch(() => {}), sleep(1000)]);
  }
  ws?.close();
  if (chrome.exitCode === null && !launchError) {
    const exited = new Promise(resolve => chrome.once('exit', resolve));
    await Promise.race([exited, sleep(1000)]);
    if (chrome.exitCode === null) { chrome.kill(); await exited; }
  }
  rmSync(profile, { recursive: true, force: true, maxRetries: 20, retryDelay: 100 });
}
