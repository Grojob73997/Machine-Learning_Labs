// Forward inference only. The weights are trained and exported by Python.
(function (root) {
  function dense(input, weights, bias) {
    return bias.map((b, j) => input.reduce((sum, value, i) => sum + value * weights[i][j], b));
  }
  function forward(input, p) {
    const a1 = dense(input, p.W1, p.b1).map(x => Math.max(0, x));
    const a2 = dense(a1, p.W2, p.b2).map(x => Math.max(0, x));
    const logits = dense(a2, p.W3, p.b3);
    const max = Math.max(...logits);
    const exponentials = logits.map(x => Math.exp(x - max));
    const total = exponentials.reduce((a, b) => a + b, 0);
    return {a1, a2, logits, probabilities: exponentials.map(x => x / total)};
  }
  const api = {dense, forward};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.LabMath = api;
})(typeof window !== 'undefined' ? window : globalThis);
