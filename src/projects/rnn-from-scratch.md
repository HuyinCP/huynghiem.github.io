---
layout: project
title: Building a Recurrent Neural Network (RNN) from Scratch
description: Building a Recurrent Neural Network from scratch with NumPy.
lead: In this project, I will build an RNN using only NumPy and use the model to generate text.
date: 16 OCTOBER 2026
output: blog/rnn-from-scratch.html
back_url: ../projects.html
back_label: Projects
---

In this project, we’re going to build a simple recurrent neural network from scratch, using NumPy.

We’ll add a fun twist to this project by training the RNN on a dataset containing the written works by Shakespeare, and we’ll see how well it can generate new writing based on the learned patterns!

Given the foundational knowledge on neural networks established in previous projects, we’ll move relatively quickly into the theory for RNNs.

This project is also closely linked to the next one on Long Short-Term Memory (LSTM) networks, where we will improve on the RNN architecture.

Here’s the structure for this project post:

<nav class="article-toc" aria-label="Article table of contents">
  <p><strong>Contents</strong></p>
  <ol>
    <li><a href="#overview">Recurrent Neural Network Overview</a></li>
    <li>
      <a href="#mathematics">Describing the RNN Mathematically</a>
      <ol>
        <li><a href="#forward">Forward Propagation</a></li>
        <li>
          <a href="#backward">Backpropagation Through Time</a>
          <ol>
            <li><a href="#bptt-derivation">Deriving Components in Backpropagation Through Time</a></li>
          </ol>
        </li>
        <li><a href="#gradients">Vanishing and Exploding Gradients</a></li>
      </ol>
    </li>
    <li><a href="#implementation">NumPy Implementation and Results</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</nav>

<h2 id="overview">1. Recurrent Neural Network Overview</h2>

Recurrent Neural Networks are a type of neural network designed for processing sequential data, such as text, audio, time-series data, and more. As the name suggests, the key feature that makes RNNs unique is the presence of recursive computational loops that span adjacent time steps.

This recurrence enables RNN models to effectively maintain a persistent internal state, or “memory,” of prior inputs, which can inform the processing of data points later in the sequence.

You might recall that a basic neural network usually has at least an input layer, a hidden layer, and an output layer. Here’s how the basic architecture of an RNN differs:

<figure class="article-figure">
  <img src="../assets/images/rnn/1.png" alt="Comparison between a basic neural network and an RNN with a recurrent hidden state">
  <figcaption>Figure 1. A basic neural network compared with a recurrent neural network.</figcaption>
</figure>

We can see that the RNN introduces a recursive relationship through $h^{\langle t\rangle}$ and also adds a time dimension $t$ to its layers.

In the diagram above, the RNN consists of a single computational unit with a self-connected hidden state. However, we can “unfold” this unit to see how information cycles across time steps:

<figure class="article-figure">
  <img src="../assets/images/rnn/2.png" alt="An RNN unfolded across three time steps with input x, hidden state h, and output y">
  <figcaption>Figure 2. An RNN unfolded across multiple time steps.</figcaption>
</figure>

As data passes through the RNN, we can see how the input sequence is processed across all time steps and how the model is influenced by information from previous time steps.

<figure class="article-figure">
  <img src="../assets/images/rnn/3.png" alt="Information flowing through an RNN across previous, current, and next time steps">
  <figcaption>Figure 3. Information from earlier inputs influences the hidden states at later time steps.</figcaption>
</figure>

An RNN receives the input $x^{\langle t\rangle}$ and the previous hidden state $h^{\langle t-1\rangle}$, then produces a new hidden state $h^{\langle t\rangle}$.

Hopefully, it is also clear how each hidden unit $h^{\langle \cdots \rangle}$ receives two inputs. For example, $h^{\langle t\rangle}$ receives the two inputs colored in red: $h^{\langle t-1\rangle}$ and $x^{\langle t\rangle}$. Correspondingly, $h^{\langle t+1\rangle}$ receives the two inputs colored in purple: $h^{\langle t\rangle}$ and $x^{\langle t+1\rangle}$.

You may also be wondering what a multi-layer RNN looks like. It can be difficult to imagine all the inputs and outputs moving through the network, so the following diagram shows both its compact and unfolded forms:

<figure class="article-figure">
  <img src="../assets/images/rnn/4.png" alt="A two-layer RNN shown in compact and unfolded forms across three time steps">
  <figcaption>Figure 4. A multi-layer RNN unfolded across multiple time steps.</figcaption>
</figure>

There is also an important point to be made about **parameter sharing** in RNNs. Let’s say we have a classic feedforward neural network that is trained to predict the next character in English sentences, or text sequences. The feedforward network would require separate weights for each input-output mapping at every character position. Consider two similar but differently worded sentences:

1. “Today, I went to the dentist”
2. “I went to the dentist today”

The feedforward neural network would learn the contextual relationships independently and therefore relearn from scratch even if it had seen similar semantic patterns before. It cannot develop a unified grammar model.

On the other hand, an RNN shares weights across time steps. If it learns a pattern or rule in one part of a sentence, it can reuse that knowledge later even when the sequence order changes slightly. More technically, the RNN embeds language patterns as reusable transition functions in the shared weights, allowing those patterns to be retained and applied regardless of input sequence length or sentence order.

For sequential data, such as characters in a sentence, RNNs can therefore use fewer trainable parameters while achieving better performance than a classic feedforward neural network.

We are now ready to look under the hood and describe the RNN operations mathematically.

<h2 id="mathematics">2. Describing the RNN Mathematically</h2>

In this section, we’ll understand the operations happening within the RNN mathematically.

Let’s begin by understanding which weight matrices are passed around in a single-hidden-layer RNN:

<figure class="article-figure">
  <img src="../assets/images/rnn/5.png" alt="Weight matrices connecting the input, hidden state, recurrent state, and output of a single-layer RNN">
  <figcaption>Figure 5. Weight matrices in a single-hidden-layer RNN.</figcaption>
</figure>

We can see that $h^{\langle t\rangle}$ receives two weight matrices: $W_{hx}$ from $x$, and $W_{hh}$ from its previous hidden state.

The hidden state $h^{\langle t\rangle}$ then outputs a weight matrix $W_{yh}$, which is passed into $y^{\langle t\rangle}$.

Let’s unfold this structure to see it more clearly:

<figure class="article-figure">
  <img src="../assets/images/rnn/6.png" alt="RNN weight matrices shown in compact and unfolded forms across three time steps">
  <figcaption>Figure 6. Weight matrices shared across the unfolded RNN.</figcaption>
</figure>

We now have a bigger picture of how the weight matrices are passed around the RNN. However, neural networks usually have other components, such as bias terms and activation functions that introduce non-linearity. In the next section, we’ll look at how these components and weight matrices are combined to form the input to each node.

<h3 id="forward">2.1. Forward Propagation</h3>

The net input to the hidden state at time $t$ and the resulting activation are:

{% include equations/rnn/forward-hidden.html %}

The output and probability distribution for the next character are:

{% include equations/rnn/forward-output.html %}

<p class="draft-note">After each equation, explain its meaning with a small example instead of only listing symbols.</p>

<h3 id="backward">2.2. Backpropagation Through Time</h3>

Using cross-entropy, the total loss over a sequence of length $T$ can be written as:

{% include equations/rnn/sequence-loss.html %}

<h4 id="bptt-derivation">2.2.1. Deriving Components in Backpropagation Through Time</h4>

An important gradient at the output layer is:

{% include equations/rnn/output-gradient.html %}

<p class="draft-note">Explain how the gradient moves backward from the final time step to the first and why this method is called Backpropagation Through Time.</p>

<h3 id="gradients">2.3. Vanishing and Exploding Gradients</h3>

Dependencies across many time steps produce a product of multiple derivatives:

{% include equations/rnn/recurrent-dependency.html %}

<p class="draft-note">Explain intuitively what happens when the multiplied terms are repeatedly smaller than 1 or larger than 1.</p>

<h2 id="implementation">3. NumPy Implementation and Results</h2>

<p class="draft-note">Describe the dataset, hyperparameters, and how you validate the model before adding your implementation.</p>

```python
import numpy as np

# Example structure for one forward step
def rnn_step(x_t, h_prev, W_hx, W_hh, b_h):
    z_h = W_hx @ x_t + W_hh @ h_prev + b_h
    h_t = np.tanh(z_h)
    return h_t
```

### 3.1. Training Results

<p class="draft-note">Add the loss curve, generated text, and your own observations here.</p>

<h2 id="references">References</h2>

- [Coding Owen's RNN project article](https://codingowen.github.io/projects/recurrent_nn_from_scratch/)
- Add the books, lectures, repositories, or documentation you actually used.
