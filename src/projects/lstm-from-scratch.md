---
title: Building Long Short-Term Memory (LSTM) from Scratch
description: A PyTorch implementation of an LSTM language model compared with a recurrent neural network for sentence generation.
lead: In this project, I build an LSTM-based model with PyTorch and compare its sentence-generation performance with the RNN model from an earlier project.
date: Published 11 January 2025
output: blog/lstm-from-scratch.html
back_url: ../projects.html
back_label: Projects
---

In this project, I build an LSTM-based model using PyTorch and some mathematics, then compare its performance against the RNN model from the previous project when generating sentences.

We will build a simple Long Short-Term Memory (LSTM) recurrent model with PyTorch and apply it to the same task as the earlier RNN model. The goal is to compare the sentences produced by both models and understand where the LSTM architecture helps.

Given the foundational knowledge of RNNs established in the previous project, we can move relatively quickly into the theory behind LSTMs.

<nav class="article-toc" aria-label="Article table of contents">
  <p><strong>Contents</strong></p>
  <ol>
    <li><a href="#lstm-overview">LSTM Overview</a></li>
    <li><a href="#forward-propagation">Forward Propagation</a></li>
    <li><a href="#backward-propagation">Backward Propagation</a></li>
    <li><a href="#gradients">How LSTMs Solve Vanishing and Exploding Gradients</a></li>
    <li><a href="#implementation">Code Implementation and Results</a></li>
  </ol>
</nav>

<h2 id="lstm-overview">1. LSTM Overview</h2>

<p>In this section, we’ll cover the architecture of the LSTM cell and examine how it improves upon the basic RNN model from the previous project. To do that, we’ll briefly review RNNs and adjust our visual perspective of them.</p>

<p>Recall that RNNs are a special type of neural network consisting of recursive computational loops that span adjacent time steps. We can visualize them unfolding like this:</p>

<figure class="article-figure">
  <img src="../assets/images/lstm/1.png" alt="An RNN unfolded across adjacent time steps">
  <figcaption>Figure 1. An RNN unfolded across adjacent time steps.</figcaption>
</figure>

<p>Also recall that our RNN has biases and activation functions. We described the net input to node $h$ as $z_h^{\langle t\rangle}$, and the activation at node $h$ as $h^{\langle t\rangle}$, like so:</p>

<h3>Net Input to Node $h$</h3>

<div class="equation-block" role="math" aria-label="RNN net input to node h">
  \[
    z_h^{\langle t\rangle} = W_{hx}x^{\langle t\rangle} + W_{hh}h^{\langle t-1\rangle} + b_h
  \]
</div>

<h3>Activation at Node $h$</h3>

<div class="equation-block" role="math" aria-label="RNN activation at node h">
  \[
    h^{\langle t\rangle} = \sigma_h\!\left(z_h^{\langle t\rangle}\right)
  \]
</div>

<p>Here, the activation function $\sigma_h$ is usually the tanh function.</p>

<p>Now, to prepare for learning about LSTMs, we’ll change our visual understanding of RNNs just a little. We’ll need to see RNNs as forming a chain of repeating modules, with a very simple structure, such as a single tanh layer:</p>

<figure class="article-figure">
  <img src="../assets/images/lstm/2.png" alt="An RNN repeating module shown as an internal x-ray view">
  <figcaption>Figure 2. An internal view of an RNN repeating module.</figcaption>
</figure>

<p>We can see that our node $h^{\langle t\rangle}$ is now an “x-ray” image of itself, where we can see how its inputs map to its activation function and to the outputs. We’ll use this visual perspective when moving forward to learn about LSTMs. We’ll also call the central node $h$ the “repeating module”.</p>

<p>LSTMs also have this chain-like structure, but the repeating module has a more complicated internal structure. Instead of the single tanh neural network layer used earlier, the LSTM repeating module has four components:</p>

<h2 id="forward-propagation">2. Forward Propagation</h2>

<h3 id="backward-propagation">2.1 Backward Propagation</h3>

<h3 id="gradients">2.2 How LSTMs Solve Vanishing and Exploding Gradients</h3>

<h2 id="implementation">3. Code Implementation and Results</h2>
