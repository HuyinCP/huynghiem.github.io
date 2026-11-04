---
title: Building Long Short-Term Memory (LSTM) from Scratch
description: A PyTorch implementation of an LSTM language model compared with a recurrent neural network for sentence generation.
lead: In this project, I build an LSTM-based model with PyTorch and compare its sentence-generation performance with the RNN model from an earlier project.
date: Spring 2025
output: blog/lstm-from-scratch.html
back_url: ../projects.html
back_label: Projects
---

Long short-term memory networks are designed to preserve useful information across longer sequences. In this project, I build an LSTM language model with PyTorch, then compare its generated sentences with the results from my recurrent neural network implementation.

<nav class="article-toc" aria-label="Article table of contents">
  <p><strong>Contents</strong></p>
  <ol>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#lstm">The LSTM Cell</a></li>
    <li><a href="#implementation">PyTorch Implementation</a></li>
    <li><a href="#comparison">Comparison with the RNN</a></li>
  </ol>
</nav>

<h2 id="overview">1. Overview</h2>

An LSTM extends the recurrent network idea with a cell state and gates. These components help the model decide what information to keep, update, and expose while processing a sequence.

<figure class="article-figure">
  <img src="../assets/images/lstm/avt.png" alt="LSTM project overview">
  <figcaption>Figure 1. Overview of the LSTM language-model project.</figcaption>
</figure>

<h2 id="lstm">2. The LSTM Cell</h2>

The forget, input, and output gates control how the hidden state and cell state change at each time step. Their sigmoid activations produce values between zero and one, which lets the cell retain or discard information gradually.

<div class="equation-block" role="math" aria-label="LSTM gate equations">
  \[
    f_t = \sigma(W_f [h_{t-1}, x_t] + b_f), \qquad
    i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)
  \]
  \[
    c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t, \qquad
    h_t = o_t \odot \tanh(c_t).
  \]
</div>

<h2 id="implementation">3. PyTorch Implementation</h2>

The model uses an embedding layer, an LSTM layer, and a linear output layer. During training, the network learns to predict the next token from the preceding context. During generation, the predicted token is fed back into the model to extend the sentence.

```python
import torch
from torch import nn


class LSTMTextModel(nn.Module):
    def __init__(self, vocabulary_size, embedding_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, tokens, state=None):
        embedded = self.embedding(tokens)
        sequence, state = self.lstm(embedded, state)
        return self.output(sequence), state
```

<h2 id="comparison">4. Comparison with the RNN</h2>

The comparison focuses on how the two models preserve context while generating text. The LSTM's cell state gives it a dedicated path for carrying information over longer spans, while the simpler RNN can be more sensitive to vanishing gradients.

The final results and examples can be added here as training experiments are recorded.
