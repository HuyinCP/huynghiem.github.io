---
title: Building the Self-Attention Mechanism from Scratch
description: A step-by-step NumPy implementation of scaled dot-product self-attention for a short input sentence.
lead: In this project, I build a basic self-attention mechanism to calculate how each token attends to the other tokens in a sentence.
date: Spring 2025
output: blog/self-attention-from-scratch.html
back_url: ../projects.html
back_label: Projects
---

Self-attention lets each token in a sequence gather information from the other tokens. In this project, I work through the scaled dot-product operation and implement a single attention head with NumPy. The small example below makes the matrix operations visible; its token vectors are hand-written for demonstration rather than learned from data.

<nav class="article-toc" aria-label="Article table of contents">
  <p><strong>Contents</strong></p>
  <ol>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#mathematics">The Attention Calculation</a></li>
    <li><a href="#implementation">NumPy Implementation</a></li>
    <li><a href="#results">Reading the Results</a></li>
    <li><a href="#references">Reference</a></li>
  </ol>
</nav>

<h2 id="overview">1. Overview</h2>

Suppose a sentence contains $n$ tokens. We represent their vectors as rows of a matrix $X$. Three learned projection matrices normally transform those vectors into **queries**, **keys**, and **values**:

<div class="equation-block" role="math" aria-label="Queries, keys, and values are linear projections of input tokens">
  \[
    Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V.
  \]
</div>

A query asks which tokens are relevant to the current token. Keys are compared with that query, while values provide the information that will be combined. Since all three are derived from the same input sequence, this is **self-attention**.

<figure class="article-figure">
  <img src="../assets/images/self_attention_machanism/avt.png" alt="Scaled dot-product attention diagram showing query-key multiplication, scaling, optional masking, softmax, and multiplication by values">
  <figcaption>Figure 1. The operations in one scaled dot-product attention head.</figcaption>
</figure>

<h2 id="mathematics">2. The Attention Calculation</h2>

First, compare every query with every key using a dot product. Divide the scores by the square root of the query/key dimension $d_k$:

<div class="equation-block" role="math" aria-label="Attention scores equal query-key dot products divided by the square root of the key dimension">
  \[
    S = \frac{QK^\mathsf{T}}{\sqrt{d_k}}.
  \]
</div>

Each row of $S$ contains the scores for one token attending to all tokens, including itself. The scaling helps keep the dot products from becoming too large as $d_k$ grows. Apply softmax to each row to obtain nonnegative **attention weights** that sum to one:

<div class="equation-block" role="math" aria-label="Row-wise softmax converts attention scores into normalized attention weights">
  \[
    A_{ij} = \frac{e^{S_{ij}}}{\sum_{m=1}^{n} e^{S_{im}}}.
  \]
</div>

Finally, use those weights to combine the value vectors:

<div class="equation-block" role="math" aria-label="The attention output is the weighted sum of value vectors">
  \[
    O = AV = \operatorname{softmax}\!\left(\frac{QK^\mathsf{T}}{\sqrt{d_k}}\right)V.
  \]
</div>

The output $O$ has one context vector per input token. If a token must not see later tokens, as in an autoregressive decoder, set the corresponding future-token scores to $-\infty$ **before** applying softmax. This optional step is called a causal mask.

<h2 id="implementation">3. NumPy Implementation</h2>

The function below computes one attention head. Subtracting the row maximum before exponentiation makes softmax numerically more stable.

```python
import numpy as np


def softmax(logits):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)


def self_attention(x, w_q, w_k, w_v, causal=False):
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    scores = (q @ k.T) / np.sqrt(q.shape[-1])
    if causal:
        future_tokens = np.triu(np.ones(scores.shape, dtype=bool), k=1)
        scores = np.where(future_tokens, -np.inf, scores)

    weights = softmax(scores)
    context = weights @ v
    return scores, weights, context


# Hand-written vectors for a three-token toy sentence.
tokens = ["the", "cat", "sat"]
x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
identity = np.eye(2)

scores, weights, context = self_attention(
    x, identity, identity, identity
)

print(tokens)
print("Attention weights:\n", np.round(weights, 3))
print("Context vectors:\n", np.round(context, 3))
```

Using identity projection matrices means $Q=K=V=X$ in this toy example. A trained model would learn separate projection matrices instead.

<h2 id="results">4. Reading the Results</h2>

Running the code prints:

```text
['the', 'cat', 'sat']
Attention weights:
 [[0.401 0.198 0.401]
  [0.198 0.401 0.401]
  [0.248 0.248 0.503]]
Context vectors:
 [[0.802 0.599]
  [0.599 0.802]
  [0.752 0.752]]
```

The first row shows that the token “the” assigns about $40.1\%$ of its attention to itself, $19.8\%$ to “cat”, and $40.1\%$ to “sat”. Multiplying this row by $V$ gives its context vector, approximately $[0.802, 0.599]$.

These numbers demonstrate the calculation, not linguistic understanding: the vectors and projection matrices were chosen by hand. A full Transformer learns its projections during training and usually runs several attention heads in parallel.

<h2 id="references">Reference</h2>

- Vaswani, A. et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). See Section 3.2.1 for scaled dot-product attention.
