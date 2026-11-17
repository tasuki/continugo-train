// Usage: node tests/test_nn_view_clipboard_unit.mjs GENERATED.nn.html
// Dependency-free logic tests; DOM and clipboard are isolated test doubles.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const html = readFileSync(process.argv[2], 'utf8');
const script = html.slice(html.lastIndexOf('<script>') + 8, html.lastIndexOf('</script>'));
new vm.Script(script); // Check the full production script's JavaScript syntax.
const functions = script.slice(script.indexOf('function legacyCopySGF()'), script.indexOf('function gallery('));
assert.ok(functions.includes('async function copySGF()'));
// Large, Unicode, annotation-bearing SGF: ensure no URL encoding, truncation,
// reserialization, or annotation stripping in any copy path.
const expected = '(;FF[9071]GM[9071]SZ[9]C[漢字 · café 👁️ ' + 'annotation '.repeat(20000) + '];B[dSdS]C[{"policy":[["dSdS",7],["",1]]}])\n';
const temporary = new Set();
let document;
const field = () => ({
  value: '', style: {}, readOnly: false, selectionStart: null, selectionEnd: null,
  focus() { document.activeElement = this; },
  select() { this.selectionStart = 0; this.selectionEnd = this.value.length; },
  setSelectionRange(a, b) { this.selectionStart = a; this.selectionEnd = b; },
  remove() { temporary.delete(this); },
});
const nodes = {
  sgf: { textContent: 'Copy SGF', disabled: false, focus() { document.activeElement = this; } },
  sgftext: field(),
  sgfcopy: { open: false, showModal() { this.open = true; }, close() { this.open = false; } },
};
document = {
  activeElement: nodes.sgf,
  createElement(tag) { assert.equal(tag, 'textarea'); return field(); },
  body: { append(node) { temporary.add(node); } },
  execCommand() { throw Error('Unexpected legacy copy'); },
};
let received;
const navigator = { clipboard: { async writeText(text) { received = text; } } };
const timers = new Map();
let timerId = 0;
const context = vm.createContext({
  copyFeedbackTimer: null,
  setTimeout(callback, delay) { timers.set(++timerId, {callback, delay}); return timerId; },
  clearTimeout(id) { timers.delete(id); },
  sgfText: expected, document, navigator, window: { isSecureContext: true },
  $: id => nodes[id],
});
vm.runInContext(functions, context);
await vm.runInContext('copySGF()', context);
assert.equal(received, expected);
assert.equal(nodes.sgf.textContent, 'Copied ✓');
assert.equal(nodes.sgf.disabled, false);
assert.equal(timers.size, 1);
const [firstTimerId, firstTimer] = [...timers][0];
assert.equal(firstTimer.delay, 1200);
firstTimer.callback();
timers.delete(firstTimerId);
assert.equal(nodes.sgf.textContent, 'Copy SGF');

navigator.clipboard.writeText = async () => { throw Error('Permission denied'); };
document.execCommand = command => {
  assert.equal(command, 'copy');
  assert.equal(document.activeElement.value, expected);
  assert.equal(document.activeElement.selectionStart, 0);
  assert.equal(document.activeElement.selectionEnd, expected.length);
  return true;
};
await vm.runInContext('copySGF()', context);
assert.equal(nodes.sgf.textContent, 'Copied ✓');
assert.equal(timers.size, 1);
const [previousTimerId] = timers.keys();
await vm.runInContext('copySGF()', context);
assert.equal(timers.size, 1);
assert.equal(timers.has(previousTimerId), false, 'Repeated copy must cancel the previous reset timer');
assert.equal(temporary.size, 0);
assert.equal(document.activeElement, nodes.sgf);

navigator.clipboard = undefined;
for (const fallback of [() => false, () => { throw Error('Unsupported'); }]) {
  document.execCommand = fallback;
  await vm.runInContext('copySGF()', context);
  assert.equal(nodes.sgfcopy.open, true);
  assert.equal(nodes.sgftext.value, expected);
  assert.equal(nodes.sgftext.selectionStart, 0);
  assert.equal(nodes.sgftext.selectionEnd, expected.length);
  assert.equal(nodes.sgf.textContent, 'Copy SGF');
  assert.equal(timers.size, 0);
  assert.equal(nodes.sgf.disabled, false);
  assert.equal(temporary.size, 0);
  nodes.sgfcopy.close();
}
console.log('Clipboard logic passed: API success, denied API + legacy success, absent API, false/throwing legacy + selected manual fallback; complete Unicode annotations preserved; transient button feedback resets after 1200 ms.');
