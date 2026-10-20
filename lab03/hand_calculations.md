# Question 3: working to reproduce by hand

Write your name, student number 73997, date, and "Lab 3 - Question 3" on paper.
Use the following as a worked guide: show the substitutions and intermediate
arithmetic yourself, then check your answers. A printed copy of this guide is
not handwritten evidence. Photograph or scan your completed sheets and save
them in results/handwritten/ before final submission.

## Sheet 1: definitions and MLP

Write: "These are forward-pass counts for ONE image. One MAC is one
multiplication plus one addition, counted as TWO FLOPs. Biases are included in
parameter counts but excluded from the representative matrix FLOPs. Activation,
pooling, softmax, normalization, residual and mean-pooling operations are omitted."

For a linear layer with I inputs and O outputs:

- Parameters = I x O + O (weights plus one bias per output).
- MACs = I x O.
- FLOPs = 2 x I x O.

The 28 x 28 image has 784 pixels.

First layer, 784 -> 256:

1. Weights: 784 x 256 = 200,704.
2. Biases: 256.
3. Parameters: 200,704 + 256 = 200,960.
4. FLOPs: 2 x 784 x 256 = 2 x 200,704 = **401,408**.

Remaining layers:

| Layer | Parameters | FLOPs |
|---|---:|---:|
| 256 -> 128 | 256 x 128 + 128 = 32,896 | 2 x 256 x 128 = 65,536 |
| 128 -> 10 | 128 x 10 + 10 = 1,290 | 2 x 128 x 10 = 2,560 |

Total parameters = 200,960 + 32,896 + 1,290 = **235,146**.
Total FLOPs = 401,408 + 65,536 + 2,560 = **469,504**.
Flatten and ReLU add no trainable parameters.

Book: section 4.2, especially Listing 4.2, printed p. 156 / PDF page 208.

## Sheet 2: CNN

Draw the shapes:

1 x 28 x 28 -> Conv -> 16 x 28 x 28 -> Pool -> 16 x 14 x 14
-> Conv -> 32 x 14 x 14 -> Pool -> 32 x 7 x 7 -> Flatten 1568 -> 10.

For stride s, padding p and dilation 1:

H_out = floor((H_in + 2p - K) / s) + 1.

Here K=3, p=1, s=1, so the first height is
floor((28+2-3)/1)+1 = 28. Width is also 28.

Convolution parameters = K x K x C_in x C_out + C_out.
Convolution FLOPs = 2 x K x K x C_in x C_out x H_out x W_out.

First convolution:

1. Products per output = 3 x 3 x 1 = 9.
2. Output values = 16 x 28 x 28 = 12,544.
3. MACs = 9 x 12,544 = 112,896.
4. FLOPs = 2 x 112,896 = **225,792**.
5. Parameters = 3 x 3 x 1 x 16 + 16 = **160**.

Second convolution (after the first pool):

1. Products per output = 3 x 3 x 16 = 144.
2. Output values = 32 x 14 x 14 = 6,272.
3. MACs = 144 x 6,272 = 903,168.
4. FLOPs = 2 x 903,168 = **1,806,336**.
5. Parameters = 3 x 3 x 16 x 32 + 32 = **4,640**.

Final linear:

- Inputs = 32 x 7 x 7 = 1,568.
- Parameters = 1,568 x 10 + 10 = **15,690**.
- FLOPs = 2 x 1,568 x 10 = **31,360**.

Totals: parameters = 160 + 4,640 + 15,690 = **20,490**;
FLOPs = 225,792 + 1,806,336 + 31,360 = **2,063,488**.

Explain: weights are reused over many image positions. Few parameters do not
necessarily mean few FLOPs. Max-pooling has no learned parameters, but performs
comparisons; those comparisons are outside this counting convention.

Book: section 4.3, especially Listing 4.4, printed p. 161 / PDF page 213.

## Sheet 3: attention FLOPs

Draw a 4 x 4 grid of patches on the image. Each patch is 7 x 7 pixels.

- Sequence length L = (28/7) x (28/7) = 16 patches.
- Patch input size = 7 x 7 = 49.
- Embedding dimension d = 64.
- Heads h = 4, so features per head d_head = 64/4 = 16.
- Feedforward dimension f = 128.

Patch projection:

2 x L x 49 x d = 2 x 16 x 49 x 64 = **100,352 FLOPs**.

ONE encoder layer:

| Operation | Substitution | FLOPs |
|---|---|---:|
| Q, K and V projections | 2 x 16 x 64 x 3 x 64 | 393,216 |
| Scores QK^T, all four heads | 4 x 2 x 16 x 16 x 16 | 32,768 |
| Attention weights times V | 4 x 2 x 16 x 16 x 16 | 32,768 |
| Output projection | 2 x 16 x 64 x 64 | 131,072 |
| Feedforward 64 -> 128 | 2 x 16 x 64 x 128 | 262,144 |
| Feedforward 128 -> 64 | 2 x 16 x 128 x 64 | 262,144 |

For clarity: each head's Q and K has shape (16,16). QK^T is
(16,16) @ (16,16), giving (16,16) scores. Four heads together cost
4 x 2 x L x L x d_head = 2 x L x L x d.
Do NOT multiply by four a second time when d=64 is already used.

Attention sublayer (QKV + scores + weighted V + output):
393,216 + 32,768 + 32,768 + 131,072 = **589,824 FLOPs**.

Full encoder: 589,824 + 262,144 + 262,144 = **1,114,112 FLOPs**.
Two encoders: 2 x 1,114,112 = **2,228,224 FLOPs**.
Classifier after mean pooling: 2 x 64 x 10 = **1,280 FLOPs**.

Whole attention model:
100,352 + 2,228,224 + 1,280 = **2,329,856 FLOPs**.

Write: "Mean pooling turns (16,64) into (64). This is why the final
classifier is applied once per image, not once per token."

Book: section 4.5.2, printed pp. 169-173 / PDF pages 221-225.

## Sheet 4: attention parameters and profiler comparison

Patch projection: 49 x 64 + 64 = **3,200** parameters.

ONE encoder:

| Component | Parameter calculation | Count |
|---|---|---:|
| QKV | 3 x (64 x 64 + 64) | 12,480 |
| Attention output projection | 64 x 64 + 64 | 4,160 |
| Feedforward first layer | 64 x 128 + 128 | 8,320 |
| Feedforward second layer | 128 x 64 + 64 | 8,256 |
| Two LayerNorms (scale and shift each) | 2 x (64 + 64) | 256 |

Encoder total = 12,480 + 4,160 + 8,320 + 8,256 + 256 = **33,472**.
Classifier = 64 x 10 + 10 = **650**.
Whole model = 3,200 + 2 x 33,472 + 650 = **70,794**.

Parameter counts do not multiply by the number of tokens: the same learned
projection is reused for each token.

Copy this comparison AFTER doing the arithmetic above:

| Model | Hand FLOPs | Native THOP output | 2 x native | Aligned THOP MACs | 2 x aligned |
|---|---:|---:|---:|---:|---:|
| MLP | 469,504 | 234,752 | 469,504 | 234,752 | 469,504 |
| CNN | 2,063,488 | 1,031,744 | 2,063,488 | 1,031,744 | 2,063,488 |
| Attention | 2,329,856 | 591,488 | 1,182,976 | 1,164,928 | 2,329,856 |

Native attention THOP includes some normalization operations but misses
MultiheadAttention. Each encoder is missing 294,912 MACs, so two are missing
589,824 MACs. Four LayerNorm modules contribute 16,384 native operation units.
Thus 1,164,928 - 589,824 + 16,384 = 591,488.

Write an explanation in your own words: "The default profiler has incomplete
attention coverage, and its normalization rule uses a different convention.
I kept its original output and then added an explicit attention rule and
excluded normalization for a like-for-like comparison. Matching a custom
formula is a consistency check, not independent hardware validation."

Native attention parameter count: 37,514 (incomplete).
torchinfo and direct enumeration: 70,794 (complete).

For batch 128, multiply the representative FLOPs by 128:
MLP **60,096,512**; CNN **264,126,464**; attention **298,221,568**.
Parameter counts stay unchanged.

## Sheet 5: measured latency, conclusion, and submission check

Read the final timing rows from report.md:

| Model | Batch 1 median, 100 runs | Batch 128 median, 20 runs |
|---|---:|---:|
| MLP | 0.05560 ms | 0.6660 ms |
| CNN | 0.19435 ms | 5.0201 ms |
| Attention | 0.81765 ms | 5.2359 ms |

These are actual recorded measurements from the supplied run, not universal
constants. If you rerun the experiment, use the newly generated report values.
Write: "Each timing uses one warm-up excluded from the measurement, eval mode,
no autograd, CPU float32, four computation threads, and the median."

For 100 sorted times, median = (50th + 51st)/2.
For 20 sorted times, median = (10th + 11th)/2.
Example amortized batch-128 cost: MLP 0.6660 / 128 = 0.005203125 ms/image.
This is throughput-related average service time, not single-request latency.

State: "The exact prescribed architectures do not satisfy the 100k-150k target.
I followed the explicit layers and disclosed the contradiction."

Then explain your choice: CNN gives the best observed accuracy (89.17%) and
smallest parameter count; MLP gives the lowest observed latency and 87.79%
accuracy. Attention achieves 83.73% here. The best choice depends on the
accuracy and resource constraint.

Final check: include your name, formulas, substituted values, units, workings,
three representative layer calculations, profiler comparison, and explanation.
Keep the handwritten pages with the lab; scan them into results/handwritten/.
