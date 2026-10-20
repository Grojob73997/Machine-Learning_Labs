(() => {
  'use strict';
  const D = window.LAB_DATA;
  const content = document.querySelector('#content');
  if (!D) {
    content.innerHTML = '<div class="error"><h2>Experiment data is missing</h2><p>Run src/build_ui.py in PyCharm, then refresh this page.</p></div>';
    return;
  }
  const tasks = [
    ['Dataset', 'Prepare the digits', 'Start with 8 × 8 images, scale the pixels, and create a reproducible train/test split.'],
    ['Architecture', 'Inside the network', 'Explore the two-hidden-layer design, from 64 input pixels to ten output classes.'],
    ['Forward pass', 'Follow a prediction', 'See a held-out digit pass through the actual trained weights, ReLU layers, and softmax.'],
    ['Loss function', 'Measure the error', 'Cross-entropy connects the correct digit to the probability assigned by the network.'],
    ['Backpropagation', 'Check the gradients', 'Compare analytical gradients against independent numerical slopes before training.'],
    ['Training', 'Learn from the data', 'Compare six experiments: two hidden widths, three learning rates, and 50 epochs each.'],
    ['Timing', 'Where the time goes', 'Measure the forward and backward passes separately on a fixed batch of 32 examples.']
  ];
  let task = 2, width = 64, rate = 0.1, sampleIndex = 0;
  const colors = ['#7e72be', '#c78139', '#087e74'];
  const esc = value => String(value).replace(/[&<>"']/g, x => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));
  const pct = value => (value * 100).toFixed(2) + '%';
  const num = (value, digits = 4) => Number(value).toFixed(digits);
  const metric = (label, value, note, highlight = false) => `<div class="metric ${highlight ? 'highlight' : ''}"><div class="metric-label">${label}</div><div class="metric-value">${value}</div><div class="metric-note">${note}</div></div>`;
  const card = (title, subtitle, body, tag = '') => `<section class="card"><div class="card-head"><div><h2>${title}</h2><p>${subtitle}</p></div>${tag ? `<span class="tag">${tag}</span>` : ''}</div>${body}</section>`;
  const detail = (label, value) => `<div class="detail-row"><span>${label}</span><strong>${value}</strong></div>`;
  const link = (path, title) => `<a class="button secondary" href="${path}">${title} ↗</a>`;
  const source = n => `<div class="links">${link(`../src/task-${n}.py`, `Task ${n} · Python source`)}</div>`;
  const selected = () => D.accuracy.find(r => r.width === width && r.learning_rate === rate);
  const parameters = () => D.models[`${width}-${rate}`].parameters;
  const infer = () => window.LabMath.forward(D.samples[sampleIndex].pixels, parameters());
  const table = (headers, rows) => `<div class="table-wrap"><table><thead><tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table></div>`;
  const nav = document.querySelector('#navigation');
  nav.innerHTML = tasks.map((t, i) => `<a class="nav-link" href="#task-${i + 1}" data-task="${i + 1}"><span class="number">0${i + 1}</span>${t[0]}</a>`).join('');

  function architecture() {
    const x = [85, 300, 515, 730], ys = [112, 148, 184, 258];
    const labels = ['INPUT', 'HIDDEN 01', 'HIDDEN 02', 'OUTPUT'];
    const sub = ['64 features', `${width} neurons · ReLU`, `${width} neurons · ReLU`, '10 classes · softmax'];
    const layerColors = ['#417895', '#087e74', '#087e74', '#b87837'];
    let lines = '', nodes = '', captions = '';
    for (let layer = 0; layer < 3; layer++) for (const a of ys) for (const b of ys) {
      lines += `<line x1="${x[layer]}" y1="${a}" x2="${x[layer+1]}" y2="${b}" stroke="#d6e4df" stroke-width="1"/>`;
    }
    x.forEach((cx, i) => {
      const indices = i === 3 ? [0, 1, 2, 9] : [1, 2, 3, i === 0 ? 64 : width];
      captions += `<text x="${cx}" y="34" text-anchor="middle" font-size="10" fill="#657e84" letter-spacing="1.4">${labels[i]}</text><text x="${cx}" y="59" text-anchor="middle" font-size="11" fill="#203d45">${sub[i]}</text>`;
      ys.forEach((cy, j) => {
        nodes += `<circle cx="${cx}" cy="${cy}" r="13" fill="white" stroke="${layerColors[i]}" stroke-width="1.8"/><text x="${cx}" y="${cy+4}" text-anchor="middle" fill="${layerColors[i]}" font-size="10">${indices[j]}</text>`;
      });
      nodes += `<rect x="${cx-7}" y="209" width="14" height="26" fill="white"/><text x="${cx}" y="230" text-anchor="middle" font-size="24" fill="${layerColors[i]}">⋮</text>`;
      captions += `<text x="${cx}" y="308" text-anchor="middle" font-size="11" fill="#607b81">(B, ${i === 0 ? 64 : i === 3 ? 10 : width})</text>`;
    });
    return `<div class="diagram-scroll"><svg class="diagram" viewBox="0 0 815 335" role="img" aria-label="Network with 64 inputs, two hidden layers of ${width} neurons and 10 softmax outputs">${lines}${nodes}${captions}</svg></div><div class="diagram-footer"><span><i class="legend-dot" style="background:#417895"></i>Input pixels</span><span><i class="legend-dot" style="background:#087e74"></i>Learned features</span><span><i class="legend-dot" style="background:#b87837"></i>Class probabilities</span><span>Dense connections · dots omit neurons · B = batch size</span></div>`;
  }

  function samples() {
    return `<div class="digit-samples" aria-label="Choose a test digit">${D.samples.map((s, i) => `<button class="sample-button ${i === sampleIndex ? 'active' : ''}" data-sample="${i}" aria-label="Inspect test digit ${s.label}" aria-pressed="${i === sampleIndex}"><canvas width="8" height="8" data-digit="${i}"></canvas><span>${s.label}</span></button>`).join('')}</div>`;
  }
  function paintDigits() {
    document.querySelectorAll('canvas[data-digit]').forEach(canvas => {
      const pixels = D.samples[Number(canvas.dataset.digit)].pixels;
      const ctx = canvas.getContext('2d');
      pixels.forEach((v, i) => {
        const r = Math.round(17 + v * 181), g = Math.round(43 + v * 194), b = Math.round(53 + v * 162);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(i % 8, Math.floor(i / 8), 1, 1);
      });
    });
  }
  function probabilityBars(probs) {
    const winner = probs.indexOf(Math.max(...probs));
    return probs.map((p, i) => `<div class="probability ${winner === i ? 'winner' : ''}"><span>${i}</span><div class="prob-track"><div class="prob-fill" style="width:${p * 100}%"></div></div><span>${pct(p)}</span></div>`).join('');
  }

  function dataset() {
    const countMax = Math.max(...D.trainCounts.map((v, i) => v + D.testCounts[i]));
    return `<div class="metrics">${metric('Images', '1,797', 'Ten digit classes')}${metric('Input features', '64', '8 × 8 grayscale pixels')}${metric('Training', D.trainSize.toLocaleString(), '70% · stratified split')}${metric('Test', D.testSize, '30% · held out', true)}</div><div class="grid-two">${card('A closer look at the digits', 'One example of each class from the held-out test set.', `${samples()}<div class="inference-grid"><div><canvas class="digit-large" width="8" height="8" data-digit="${sampleIndex}"></canvas><p class="digit-caption">True label: <b>${D.samples[sampleIndex].label}</b><br>Test row: ${D.samples[sampleIndex].testIndex}</p></div><div>${detail('Original pixel range', '0 to 16')}${detail('Scaling', 'X / 16.0')}${detail('Scaled pixel range', '0 to 1')}${detail('Target encoding', 'One-hot · 10 columns')}${detail('Random seed', '42')}</div></div>${source(1)}`, 'LOAD_DIGITS')}${card('Class balance', 'Stratification preserves the class proportions.', `<div class="counts">${D.trainCounts.map((v, i) => `<div class="count-row"><b>${i}</b><div class="count-track"><span style="width:${v / countMax * 100}%"></span><span style="width:${D.testCounts[i] / countMax * 100}%"></span></div><span>${v} / ${D.testCounts[i]}</span></div>`).join('')}</div><div class="chart-legend"><span><i class="legend-dot" style="background:#087e74"></i>Train</span><span><i class="legend-dot" style="background:#97cbb9"></i>Test</span></div><div class="note">Training labels: (1257, 10)<br>Test labels: (540, 10)</div>`)}</div>`;
  }

  function design() {
    const count = 64 * width + width + width * width + width + width * 10 + 10;
    return `<div class="metrics">${metric('Input → output', '64 → 10', 'Pixels to digit classes')}${metric('Hidden layers', '2', `${width} neurons in each layer`)}${metric('Parameters', count.toLocaleString(), 'Weights and biases')}${metric('Test accuracy', pct(selected().test_accuracy), `Trained model · learning rate ${rate}`, true)}</div>${card('Network architecture', 'Change the hidden width above to explore the two required designs.', architecture(), `64 → ${width} → ${width} → 10`)}<div class="grid-two grid-equal">${card('Parameter dimensions', 'Rows are examples; each layer computes A @ W + b.', table(['Layer', 'Weight shape', 'Bias shape'], [`<tr><td>Input → hidden 1</td><td class="mono">(64, ${width})</td><td class="mono">(${width},)</td></tr>`,`<tr><td>Hidden 1 → hidden 2</td><td class="mono">(${width}, ${width})</td><td class="mono">(${width},)</td></tr>`,`<tr><td>Hidden 2 → output</td><td class="mono">(${width}, 10)</td><td class="mono">(10,)</td></tr>`]))}${card('Initialization & activations', 'The drawing matches the Python implementation.', `${detail('Weights', 'Gaussian · mean 0 · std 0.01')}${detail('Biases', 'All zeros')}${detail('Hidden activation', 'ReLU: max(0, z)')}${detail('Output activation', 'Stable softmax')}<div class="links">${link('../results/task-2-network-diagram.pdf', 'Print both designs')}${link('../src/task-2.py', 'Task 2 · Python source')}</div>`)}</div>`;
  }

  function forwardPage() {
    const result = infer(), p = result.probabilities;
    const winner = p.indexOf(Math.max(...p)), sample = D.samples[sampleIndex];
    return `<div class="metrics">${metric('Batch output', '(32, 10)', 'Shape assertion passed')}${metric('Probability sum', p.reduce((a,b)=>a+b,0).toFixed(8), 'Each row sums to one')}${metric('ReLU 1 active', result.a1.filter(v=>v>0).length + ' / ' + width, 'For this test example')}${metric('Predicted digit', winner, `True label: ${sample.label}`, true)}</div><div class="grid-two">${card('Try a held-out digit', 'Inference runs in your browser using exported Python-trained weights.', `${samples()}<div class="inference-grid"><div><canvas class="digit-large" width="8" height="8" data-digit="${sampleIndex}"></canvas><p class="digit-caption">True digit ${sample.label} · test row ${sample.testIndex}</p><div class="prediction"><strong>${winner}</strong><p>Model prediction<br><span class="pill ${winner !== sample.label ? 'wrong' : ''}">${winner === sample.label ? 'Correct' : 'Incorrect'}</span></p></div></div><div>${probabilityBars(p)}</div></div>`, `W = ${width} · LR = ${rate}`)}${card('The forward path', 'All matrix operations are implemented from scratch.', `<div class="formula">Z1 = X @ W1 + b1<br>A1 = ReLU(Z1)<br>Z2 = A1 @ W2 + b2<br>A2 = ReLU(Z2)<br>Z3 = A2 @ W3 + b3<br>P = softmax(Z3)</div><p class="body-copy">Subtract the largest logit before exponentiation. This keeps softmax numerically stable without changing its probabilities.</p><div class="note">The Python task checks batch shape (32, 10) and row sums within 10⁻⁶. Browser predictions are independently checked against Python outputs for all 60 model/example combinations.</div>${source(3)}`)}</div>`;
  }

  function lossPage() {
    const result = infer(), truth = D.samples[sampleIndex].label;
    const logits = result.logits, max = Math.max(...logits);
    const loss = Math.log(logits.reduce((s,x)=>s+Math.exp(x-max),0)) - (logits[truth]-max);
    return `<div class="metrics">${metric('Initial loss', num(D.initialLosses[width], 6), `Untrained · width ${width}`)}${metric('Expected baseline', num(Math.log(10),6), 'ln(10) · ten equal probabilities')}${metric('Final train loss', num(selected().train_loss), 'Mean over training examples')}${metric('Final test loss', num(selected().test_loss), 'Mean over held-out examples', true)}</div><div class="grid-two grid-equal">${card('Cross-entropy loss', 'The network is penalized when it gives low probability to the correct class.', '<div class="formula">L = −(1/B) Σᵢ Σⱼ Yᵢⱼ log(Pᵢⱼ)</div><p class="body-copy">With one-hot labels, each example contributes only the negative log-probability of its true class. The implementation computes this from shifted logits using log-sum-exp.</p><div class="note">At initialization, all ten classes have probability near 0.1. The loss should therefore be approximately −ln(0.1) = 2.302585.</div>' + source(4))}${card('Inspect one example', 'Select a digit and see its contribution to the loss.', `${samples()}${detail('True class', truth)}${detail('Probability of true class', pct(result.probabilities[truth]))}${detail('Example loss: −ln(Ptrue)', num(loss, 6))}<p class="body-copy">This is a single-example loss from the trained model. The metrics above report the mean loss across a dataset.</p>`)}</div>`;
  }

  function gradients() {
    const rows = D.gradients.filter(r=>r.width === width), passed = rows.filter(r=>r.passed).length;
    return `<div class="metrics">${metric('Checks passed', `${passed} / 10`, `Width ${width}`, true)}${metric('Required', '≥ 9 / 10', 'Manual passing threshold')}${metric('Perturbation', '10⁻⁵', 'Central difference · ± epsilon')}${metric('Max relative error', Math.max(...rows.map(r=>r.relative_error)).toExponential(2), 'Required < 10⁻⁶')}</div>${card('Analytical vs. numerical gradients', 'Ten randomly selected weights, seed 42. The same coordinates are checked without replacing failures.', table(['Parameter', 'Index', 'Analytical', 'Numerical', 'Relative error', 'Result'], rows.map(r=>`<tr><td>${esc(r.parameter)}</td><td class="mono">${esc(r.index)}</td><td class="mono">${r.analytical.toExponential(5)}</td><td class="mono">${r.numerical.toExponential(5)}</td><td class="mono">${r.relative_error.toExponential(3)}</td><td class="pass">${r.passed ? 'PASS' : 'FAIL'}</td></tr>`)), 'VERIFIED BEFORE TRAINING')}<div class="grid-two grid-equal">${card('Backward propagation', 'The chain rule sends the output error through each layer.', '<div class="formula">D3 = (P − Y) / B<br>D2 = (D3 @ W3ᵀ) ⊙ (Z2 > 0)<br>D1 = (D2 @ W2ᵀ) ⊙ (Z1 > 0)<br>dW = Apreviousᵀ @ D<br>db = sum(D, axis=0)</div>' + source(5))}${card('Independent numerical check', 'Central differences estimate a slope from two nearby losses.', '<div class="formula">gnum = [L(w + ε) − L(w − ε)] / (2ε)</div><p class="body-copy">Relative error = |ganalytic − gnum| / max(10⁻¹², |ganalytic| + |gnum|).</p><div class="note">The reference forward pass uses Python Decimal with 50-digit precision to reduce cancellation for tiny gradients. Model training itself uses NumPy float64.</div><p class="body-copy">The manual also asks for a gradient derivation by hand. The implemented formulas here support that explanation.</p>' + link('../results/today_gradient_check.csv','Download gradient checks'))}</div>`;
  }

  function lossChart() {
    const rates = [0.001,0.01,0.1], series = rates.map(r=>D.history.filter(h=>h.width===width && h.learning_rate===r));
    const all = series.flat(), maximum = Math.ceil(Math.max(...all.map(r=>r.train_loss)) * 2) / 2;
    const left=48, top=22, plotWidth=665, plotHeight=225;
    const xp = epoch=>left+(epoch-1)/49*plotWidth, yp=loss=>top+plotHeight-loss/maximum*plotHeight;
    let svg='';
    for(let i=0;i<=5;i++) {const loss=i*maximum/5; svg+=`<line x1="${left}" y1="${yp(loss)}" x2="${left+plotWidth}" y2="${yp(loss)}" stroke="#e5ece8"/><text x="${left-12}" y="${yp(loss)+4}" text-anchor="end" font-size="10" fill="#82948c">${loss.toFixed(1)}</text>`;}
    [1,10,20,30,40,50].forEach(e=>{svg+=`<text x="${xp(e)}" y="${top+plotHeight+23}" text-anchor="middle" font-size="10" fill="#82948c">${e}</text>`;});
    series.forEach((rows,i)=>{svg+=`<polyline points="${rows.map(r=>`${xp(r.epoch)},${yp(r.train_loss)}`).join(' ')}" fill="none" stroke="${colors[i]}" stroke-width="${rate===rates[i]?3:2}" ${i===1?'stroke-dasharray="6 4"':''}><title>Learning rate ${rates[i]}, final training loss ${rows.at(-1).train_loss.toFixed(6)}</title></polyline>`;});
    return `<svg class="chart" viewBox="0 0 745 300" role="img" aria-label="Training loss over 50 epochs, hidden width ${width}"><text x="48" y="11" font-size="10" fill="#607b70">Cross-entropy</text>${svg}<text x="380" y="294" text-anchor="middle" font-size="10" fill="#82948c">Epoch</text></svg><div class="chart-legend">${rates.map((r,i)=>`<span><i class="legend-dot" style="background:${colors[i]}"></i>LR ${r}</span>`).join('')}</div>`;
  }

  function training() {
    return `<div class="metrics">${metric('Test accuracy', pct(selected().test_accuracy), `Width ${width} · learning rate ${rate}`, true)}${metric('Training loss', num(selected().train_loss), 'After epoch 50')}${metric('Experiments', '6', '2 widths × 3 learning rates')}${metric('Epochs logged', '300', '50 per experiment')}</div>${card('Training loss', 'Actual per-epoch cross-entropy. The selected learning rate is drawn with a thicker line.',lossChart(), `HIDDEN WIDTH ${width}`)}<div class="grid-two">${card('Compare every experiment', 'Each run starts from the same initialization for its width.', table(['Width','Learning rate','Train loss','Test loss','Test accuracy'],D.accuracy.map(r=>`<tr class="${r.width===width&&r.learning_rate===rate?'selected':''}"><td>${r.width}</td><td>${r.learning_rate}</td><td>${num(r.train_loss)}</td><td>${num(r.test_loss)}</td><td>${pct(r.test_accuracy)}</td></tr>`))) }${card('What these results show', 'The prescribed small initialization makes the learning rate influential.', '<p class="body-copy">At learning rates 0.001 and 0.01, the loss remains near ln(10) after 50 epochs. A rate of 0.1 produces clearer learning; the wider network reaches the highest test accuracy in this experiment.</p><div class="note">No accuracy target is imposed by the manual. The interface reports every run, including the configurations that learn slowly.</div><div class="links">'+link('../results/loss_curves.png','Loss curves PNG')+link('../results/today_accuracy.csv','Accuracy CSV')+'</div>'+source(6))}</div>`;
  }

  function timing() {
    const row=D.timing.find(r=>r.width===width&&r.learning_rate===rate);
    const max=Math.max(...D.timing.map(r=>Math.max(r.forward_median_us,r.backward_median_us)));
    const dominant=row.backward_median_us >= row.forward_median_us?'Backward':'Forward';
    return `<div class="metrics">${metric('Forward median', num(row.forward_median_us,2)+' μs', 'Fixed batch of 32')}${metric('Backward median',num(row.backward_median_us,2)+' μs','Cached forward activations',true)}${metric('Back / forward',num(row.backward_median_us/row.forward_median_us,2)+'×','Ratio of measured medians')}${metric('Repetitions','100','One warm-up before timing')}</div><div class="grid-two">${card('Forward vs. backward', 'Median execution time in microseconds; shorter bars are faster.',D.timing.map(r=>`<div class="timing-item"><div class="timing-label"><strong>W ${r.width} · LR ${r.learning_rate}</strong><span>${num(r.forward_median_us,2)} / ${num(r.backward_median_us,2)} μs</span></div><div class="timing-pair"><div class="timing-bar" style="width:${r.forward_median_us/max*100}%;background:#91c5b2"></div><div class="timing-bar" style="width:${r.backward_median_us/max*100}%;background:#087e74"></div></div></div>`).join('')+'<div class="chart-legend"><span><i class="legend-dot" style="background:#91c5b2"></i>Forward</span><span><i class="legend-dot" style="background:#087e74"></i>Backward</span></div>')} ${card('Explain the difference','Connect the measurements to the matrix operations.',`<p class="body-copy"><b>${dominant} propagation takes longer for the selected run.</b> The backward pass computes weight gradients and propagates errors through transposed weight matrices, adding matrix products and ReLU masks.</p><p class="body-copy">A wider hidden layer increases matrix sizes. W2 grows from 16 × 16 to 64 × 64, so hidden-to-hidden matrix work increases substantially.</p><div class="note">Timings depend on hardware and system load. These are medians of 100 Python calls using perf_counter_ns, one BLAS thread, and one warm-up for each operation. Backward timing excludes the forward pass.</div>${source(7)}<div class="links">${link('../results/today_timing.csv','Timing CSV')}</div>`)}</div>`;
  }

  function render() {
    document.querySelector('#eyebrow').textContent=`TASK 0${task} / 07`;
    document.querySelector('#page-title').textContent=tasks[task-1][1];
    document.querySelector('#page-description').textContent=tasks[task-1][2];
    document.title=`Task ${task} · ${tasks[task-1][0]} — Digits Lab`;
    document.querySelectorAll('.nav-link').forEach(a=>{
      const active=Number(a.dataset.task)===task;
      a.classList.toggle('active',active);
      if(active) a.setAttribute('aria-current','page'); else a.removeAttribute('aria-current');
    });
    content.innerHTML=[dataset,design,forwardPage,lossPage,gradients,training,timing][task-1]();
    paintDigits();
  }
  content.addEventListener('click',event=>{
    const button=event.target.closest('[data-sample]');
    if(button){sampleIndex=Number(button.dataset.sample);render();content.querySelector(`[data-sample="${sampleIndex}"]`)?.focus({preventScroll:true});}
  });
  document.querySelector('#width').addEventListener('change',event=>{width=Number(event.target.value);render();});
  document.querySelector('#rate').addEventListener('change',event=>{rate=Number(event.target.value);render();});
  const dialog=document.querySelector('#help-dialog');
  document.querySelector('#help-button').addEventListener('click',()=>dialog.showModal());
  function navigate(){const match=location.hash.match(/^#task-([1-7])$/);task=match?Number(match[1]):2;render();}
  window.addEventListener('hashchange',navigate);
  document.querySelector('#updated').textContent='Measured '+new Date(D.generatedAt).toLocaleString();
  navigate();
})();
