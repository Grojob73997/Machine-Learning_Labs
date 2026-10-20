// Run with Node.js after src/build_ui.py to compare browser and Python inference.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {forward} = require('./network.js');
const context = {window: {}};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, 'data.js'), 'utf8'), context);
const data = context.window.LAB_DATA;
let checked = 0, maxError = 0;
for (const [key, model] of Object.entries(data.models)) {
  for (let i = 0; i < data.samples.length; i++) {
    const probabilities = forward(data.samples[i].pixels, model.parameters).probabilities;
    assert.equal(probabilities.length, 10);
    assert.ok(Math.abs(probabilities.reduce((a,b)=>a+b,0)-1) < 1e-12);
    probabilities.forEach((value, j) => {
      const error = Math.abs(value - model.referenceProbabilities[i][j]);
      maxError = Math.max(maxError, error);
      assert.ok(error < 1e-12, `${key}, example ${i}, class ${j}: error ${error}`);
    });
    checked++;
  }
}
assert.equal(checked, 60);
assert.equal(data.history.length, 300);
assert.equal(data.gradients.filter(row=>row.passed).length, 20);
console.log(`PASS: ${checked} browser/Python predictions; maximum error ${maxError.toExponential(3)}.`);
console.log('PASS: 300 logged epochs and 20 passing gradient checks.');
