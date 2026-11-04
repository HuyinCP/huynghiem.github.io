---
title: Building a Vision Transformer (ViT) from Scratch
description: A Vision Transformer implementation for image classification.
lead: In this project, I build a Vision Transformer for image classification.
date: Published 08 March 2025
output: blog/vision-transformer-from-scratch.html
back_url: ../projects.html
back_label: Projects
---

In this project, I build a Vision Transformer for image classification.

<p class="article-published"><strong>Published</strong><br>08 March 2025</p>

In this project, we’re going to implement a Vision Transformer (ViT) for image classification on the CIFAR-10 image dataset.

Given that we’ve laid the theoretical foundation for the attention mechanism before, this will be more of a code-along post, with some theory guiding us along the way.

I trained the ViT on a 10-class image-classification task for only 50 epochs and achieved 64% test accuracy. That is about six times better than random guessing, but still nowhere near a state-of-the-art CNN. ViTs usually take hundreds to thousands of training epochs and a large amount of data to perform at their best, so this project is mainly a learning exercise and a simple, unoptimized demonstration.

Really quickly, though, here’s a cool visualization of how our ViT learns to compute attention over the images:

<figure class="article-figure">
  <img src="../assets/images/vit/1.png" alt="Visualization of Vision Transformer attention over image patches">
  <figcaption>Figure 1. A visualization of attention computed over image patches.</figcaption>
</figure>

<p>We’ll see how this is implemented in the following sections.</p>

<nav class="article-toc" aria-label="Article table of contents">
  <p><strong>Contents</strong></p>
  <ol>
    <li><a href="#self-attention">Quick recap on how Self-Attention works</a></li>
    <li><a href="#vit-overview">Overview of the Vision Transformer architecture</a></li>
    <li><a href="#implementation">Code-along each portion of the ViT</a>
      <ol>
        <li><a href="#embeddings">Transform input images into embeddings</a></li>
        <li><a href="#encoder">Pass through the Transformer Encoder</a></li>
        <li><a href="#classification">Do image classification with the CLS token</a></li>
      </ol>
    </li>
    <li><a href="#results">Results</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</nav>

<h2 id="self-attention">1. Quick Recap on How Self-Attention Works</h2>

<h2 id="vit-overview">2. Overview of the Vision Transformer Architecture</h2>

<h2 id="implementation">3. Code-along Each Portion of the ViT</h2>

<h3 id="embeddings">3.1 Transform Input Images into Embeddings</h3>

<h3 id="encoder">3.2 Pass through the Transformer Encoder</h3>

<h3 id="classification">3.3 Do Image Classification with the CLS Token</h3>

<h2 id="results">4. Results</h2>

<h2 id="references">5. References</h2>
