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

<p>In the Self-Attention mechanism, we represent words as vectors in an embedding space. By considering the context of other words, a machine-learning model can learn better word embeddings and improve its performance on NLP tasks.</p>

<p>To consider the context of other words, we compare the embedding vectors and calculate their similarity scores. In the embedding space, similar tokens share similar locations, so token vectors that are close together will have a higher similarity score.</p>

<p>As we pass word embeddings through self-attention in Transformer encoder blocks, the position of each embedding vector is updated to better reflect the meaning of its token in context, based on the other relevant words in the same sentence.</p>

<p>Recall this diagram from the previous Self-Attention project:</p>

<figure class="article-figure">
  <img src="../assets/images/vit/2.png" alt="Self-attention mechanism showing token vectors and attention scores">
  <figcaption>Figure 2. Self-attention updates token representations using context.</figcaption>
</figure>

<p>The initial embedding for the token “Apple” is the same for each possible interpretation of “Apple”, whether it refers to a fruit or a technology brand. By passing the tokens through attention encoder blocks, the “Apple” token is pushed towards its context-based meaning. The encoder with attention layers can therefore add contextual meaning to embeddings.</p>

<p>Usually, Transformer-based architectures contain several encoder blocks. Each encoder block consists of normalization layers, multi-head attention layers, and a multilayer perceptron (MLP) component.</p>

<p>Each encoder block encodes more information into the embeddings by taking context into account, producing a deeper semantic representation of each token. At the end of this process, we get optimized and improved embeddings.</p>

<p>To transform these information-rich embeddings into useful predictions, we need a final set of layers, usually called the “head”. Different heads are used for different tasks, such as classification, question answering, and Named Entity Recognition (NER).</p>

<figure class="article-figure">
  <img src="../assets/images/vit/3.png" alt="Transformer encoder blocks and task-specific prediction head">
  <figcaption>Figure 3. Transformer encoder blocks followed by a task-specific head.</figcaption>
</figure>

<p>The Vision Transformer works similarly, but rather than taking word tokens as input, the ViT takes image patches. Aside from that, the overall Transformer architecture remains very similar for the ViT.</p>

<p>Next, we’ll overview the ViT architecture and make sense of its core components.</p>

<h2 id="vit-overview">2. Overview of the Vision Transformer Architecture</h2>

<p>This is the overall Vision Transformer architecture:</p>

<figure class="article-figure">
  <img src="../assets/images/vit/4.png" alt="Overall Vision Transformer architecture from image patches to classification">
  <figcaption>Figure 4. The overall Vision Transformer architecture.</figcaption>
</figure>

<p>The Vision Transformer (ViT) is inspired by Transformer architectures used in NLP, particularly BERT, an encoder-only Transformer model.</p>

<p>The key idea behind ViT is to represent an image as a sequence of smaller, non-overlapping image patches. These patches are treated as input tokens, similar to words in NLP. Their embeddings are then processed by a Transformer, enabling the network to capture spatial relationships and patterns across the entire image.</p>

<p>Looking at the overall diagram, we can break the ViT into the following series of components and steps:</p>

<ol>
  <li>Transform input images into embeddings</li>
  <li>Pass the embeddings through the Transformer Encoder</li>
  <li>Perform image classification with the CLS token</li>
</ol>

<p>Next, we will cover the relevant theory for each portion while implementing the model in code as we go along.</p>

<h2 id="implementation">3. Code-along Each Portion of the ViT</h2>

<p>The changes introduced by the ViT are mostly limited to the first few processing steps, so most of the accompanying theoretical explanation will focus on Section 3.1 below.</p>

<h3 id="embeddings">3.1 Transform Input Images into Embeddings</h3>

<p>We need to take an input image and first process it into a set of patch embeddings.</p>

<p>Some might ask: why not go more granular and feed the pixel values of the image directly into the model? Self-attention requires comparing every input token with every other token in order to learn context. If we process a small $32 \times 32$ image at the pixel level, we would have $32^2 = 1{,}024$ individual pixel tokens. Since self-attention has quadratic complexity, each token attends to every other token, resulting in $1{,}024^2 = 1{,}048{,}576$ attention computations per layer. That is only for a single attention layer in a multi-layer Transformer, making pixel-level processing computationally impractical.</p>

<p>Instead, we partition the image into patches and treat each patch as a token. These patches are embedded into a lower-dimensional representation, significantly reducing the number of tokens while still capturing meaningful spatial information before further processing.</p>

<p>Here is a high-level overview of the steps we need to take:</p>

<figure class="article-figure">
  <img src="../assets/images/vit/5.png" alt="Vision Transformer image patch embedding pipeline">
  <figcaption>Figure 5. From image patches to position-aware token embeddings.</figcaption>
</figure>

<ol>
  <li>Split the image into non-overlapping image patches.</li>
  <li>Embed patches into lower-dimensional embeddings through a linear projection.</li>
  <li>Prepend a trainable class embedding to the set of patch embeddings.</li>
  <li>Sum the patch embeddings with learned positional embeddings.</li>
</ol>

<p>After these steps, the patch embeddings are processed like token embeddings in a typical Transformer. Let’s cover each sub-step in more detail.</p>

<h4 id="split-patches">1. Split the Image into Non-overlapping Image Patches</h4>

<p>Splitting the image into non-overlapping image patches is a simple process, analogous to how sentences are split into tokens for NLP tasks.</p>

<p>Let’s assume a square image patch and define a variable, <code>patch_size</code>, which describes the size of our image patch, like so:</p>

<figure class="article-figure">
  <img src="../assets/images/vit/6.png" alt="Analogy between splitting a sentence into word tokens and splitting an image into non-overlapping square patches of size patch_size">
  <figcaption>Figure 6. Splitting an image into non-overlapping patches of size <code>patch_size</code>, analogous to tokenizing a sentence in NLP.</figcaption>
</figure>

<h4 id="patch-projection">2. Embed Patches into Lower-Dimensional Embeddings via Linear Projection</h4>

<p>Next, we need to embed image patches into their embedded vector form. Let’s understand this mathematically.</p>

<p>Let’s say we’re working with $32 \times 32$ input images in RGB form. Then our input image is a $(3, 32, 32)$ tensor, where we have $3$ channels and a $32 \times 32$ shape.</p>

<p>If we define <code>patch_size</code> to be $4$, then each image patch is a $(3, 4, 4)$ 3D tensor.</p>

<p>The Transformer expects each token to be a flat feature vector of a fixed dimension. Let’s call the dimensionality of our desired embedding vector <code>hidden_size</code>.</p>

<p>Since each image patch is in 3D tensor form, we need a linear projection to flatten each patch while preserving its structure and map it into a fixed-dimensional embedding vector. This operation is represented mathematically as:</p>

<div class="equation-block" role="math" aria-label="Linear projection mapping a C by P by P patch into a D-dimensional vector">
  \[
    \mathbb{R}^{C \times P \times P} \rightarrow \mathbb{R}^{D}
  \]
</div>

<p>where $C \times P \times P$ is the original shape of each image patch, and $D$ is <code>hidden_size</code>. This linear projection learns to extract features from each image patch, just like an embedding layer in NLP.</p>

<p>Let’s implement the two steps above in code. To do so, we’ll define a <code>PatchEmbeddings</code> class to handle splitting an input image and embedding it as vectors, based on a user-defined dictionary of variables called <code>config</code>:</p>

```python
class PatchEmbeddings(nn.Module):
    """
    Converts an input image into patches, then projects patches into embedded vectors.
    """

    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection = nn.Conv2d(
            self.num_channels,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x
```

<p>Notice that we use the <code>nn.Conv2d()</code> convolution layer to do two things at once:</p>

<ol>
  <li>Split the image into non-overlapping patches by setting both <code>kernel_size</code> and <code>stride</code> to <code>patch_size</code>.</li>
  <li>Perform a linear projection to the number of output channels defined by <code>hidden_size</code>.</li>
</ol>

<p>Let’s say our input has shape <code>(B, 3, 32, 32)</code>, where we have $B$ RGB ($3$-channel) images of size $32 \times 32$, with <code>patch_size</code> $= 4$ and <code>hidden_size</code> $= 48$.</p>

<p>Passing <code>(B, 3, 32, 32)</code> into <code>nn.Conv2d()</code> outputs <code>(B, 48, 8, 8)</code>, giving $8 \times 8 = 64$ patches from splitting each $32 \times 32$ image into patches of size $4$.</p>

<p>Then, passing the output <code>(B, 48, 8, 8)</code> into <code>x.flatten(2).transpose(1, 2)</code> performs two operations:</p>

<ul>
  <li><code>flatten(2)</code> collapses the last two dimensions ($8 \times 8$) into one, resulting in shape <code>(B, 48, 64)</code>.</li>
  <li><code>transpose(1, 2)</code> swaps dimensions $1$ and $2$, yielding <code>(B, 64, 48)</code>. This matches the <code>(batch_size, num_patches, embedding_dim)</code> shape expected by the Transformer.</li>
</ul>

<p>Hence, each image has been transformed into a sequence of $64$ embedding vectors.</p>

<h4 id="cls-token">3. Prepend a Trainable Class Embedding to the Set of Patch Embeddings</h4>

<p>One feature popularized in Transformer models like BERT is the use of a <code>[CLS]</code> (“classification”) token—a special token prepended to every input sequence.</p>

<p>In BERT, the <code>[CLS]</code> token does not represent any word from the input; instead, it serves as a global representation of the entire sequence. After passing through multiple self-attention layers, its final embedding is used for classification.</p>

<p>Because Transformers do not have a built-in pooling mechanism for aggregating information across tokens, ViT introduces a <code>[CLS]</code> token to:</p>

<ul>
  <li>Serve as a global representation of the entire image.</li>
  <li>Interact with all other patch embeddings via self-attention.</li>
  <li>Be used for final image classification.</li>
</ul>

<figure class="article-figure">
  <img src="../assets/images/vit/7.png" alt="Prepending a learnable [CLS] class embedding to the projected image patch embeddings before the Transformer encoder">
  <figcaption>Figure 7. Prepending a learnable <code>[CLS]</code> token embedding to the sequence of patch embeddings.</figcaption>
</figure>

<p>The <code>[CLS]</code> token’s embedding is learned and optimized during training.</p>

<h4 id="positional-embeddings">4. Sum Our Patch Embeddings with Learned Positional Embeddings</h4>

<p>Transformers treat all input tokens independently at first. The Self-Attention mechanism does not inherently recognize the order or spatial position of input tokens.</p>

<p>What are the implications of this? If we shuffle the input patches of two copies of the same image, the Vision Transformer would treat them as the same input since self-attention does not inherently track spatial positions. The ViT would then output the same result, leaving the model with a poor ability to understand image structure.</p>

<p>To preserve spatial information in ViTs, we add position embeddings to the patch embeddings before feeding them into the Transformer:</p>

<div class="equation-block" role="math" aria-label="Patch Embeddings plus Position Embeddings becomes Input to Transformer">
  \[
    \text{Patch Embeddings} + \text{Position Embeddings} \rightarrow \text{Input to Transformer}
  \]
</div>

<p>There are multiple ways to introduce position embeddings in ViTs, but the standard approach in the original ViT paper is to use <strong>learnable position embeddings</strong> with the same dimensionality as the patch embeddings.</p>

<p>Each patch index gets its own learnable position embedding, and during training, these embeddings are learned alongside the rest of the model.</p>

<p>One big question I had when learning about this topic was: <em>why do we need to use learnable position embeddings specifically?</em></p>

<p>Earlier, we discussed why the model needs a spatial understanding of image patches. In the simplest case, if positional encodings were not learnable—for example, if they were hardcoded using the raw $(x, y)$ coordinates of each image patch—the model might struggle to generalize across different spatial arrangements. With rigid hardcoded positions, the network might not interpret “cat at top of image” and “cat at bottom of image” similarly, introducing rigid spatial biases about image patches.</p>

<p>Instead, learnable position embeddings allow the model to flexibly adapt its spatial representation based on the data, helping it approximate a form of translational invariance.</p>

<p>Furthermore, in Appendix D.4 of the original ViT paper, the authors ran an ablation study comparing different positional embeddings and found that having any type of learnable position embedding (whether 1D, 2D, or relative) performed significantly better than having no positional embedding at all.</p>

<figure class="article-figure">
  <img src="../assets/images/vit/8.png" alt="Adding learnable position embeddings (0 through 9) to the [CLS] token and patch embeddings before passing into the Transformer Encoder">
  <figcaption>Figure 8. Summing learnable position embeddings with the <code>[CLS]</code> token and projected patch embeddings.</figcaption>
</figure>

<p>Here’s one last tidbit that might be interesting: notice that the additional <code>[CLS]</code> token we prepend for classification also receives a learnable positional embedding. Why?</p>

<p>Because the Transformer architecture does not inherently distinguish between token roles, without its own positional embedding the <code>[CLS]</code> token would be treated like any other image patch, and the model wouldn’t easily recognize that it serves a global, higher-level purpose.</p>

<p>Having a unique learnable position embedding helps the model recognize that this token plays a special role. Through self-attention, the <code>[CLS]</code> token then aggregates information from all patches to summarize the image effectively.</p>

<p>Let’s implement the two steps above in code by defining an <code>Embeddings</code> class that combines our patch embeddings with the <code>[CLS]</code> token and positional embeddings:</p>

```python
class Embeddings(nn.Module):
    """
    Combines original patch embeddings with [CLS] token and position embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"])
        )
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x
```

<ul>
  <li>Notice that we initialize <code>self.cls_token</code> as a single learnable token with shape <code>(1, 1, hidden_size)</code>, representing <code>(batch dimension placeholder, one token, hidden_size)</code>.</li>
  <li>Next, we initialize <code>self.position_embeddings</code> as a learnable tensor of shape <code>(1, num_patches + 1, hidden_size)</code>. The leading <code>1</code> broadcasts a single shared set of positional embeddings across all images in the batch, and <code>num_patches + 1</code> accounts for the prepended <code>[CLS]</code> token.</li>
  <li>We expand the single <code>[CLS]</code> token to match <code>batch_size</code> using <code>self.cls_token.expand(batch_size, -1, -1)</code>, where <code>-1</code> tells PyTorch to keep the original size along that dimension.</li>
  <li>We concatenate the <code>[CLS]</code> token to the patch embeddings along <code>dim=1</code> (the sequence length dimension).</li>
  <li>Finally, after adding <code>self.position_embeddings</code>, we apply <code>self.dropout(x)</code> for regularization.</li>
</ul>

<p>After this step is complete, the input image has been converted into a sequence of patch embeddings with positional information, ready to be fed into the Transformer Encoder.</p>

<h3 id="encoder">3.2 Pass through the Transformer Encoder</h3>

<p>In this section, we’ll build the Transformer Encoder to process our combined image patch embeddings. As shown below, the Transformer Encoder is made up of several smaller components:</p>

<figure class="article-figure">
  <img src="../assets/images/vit/9.png" alt="Transformer Encoder internal structure showing Layer Normalization, Multi-Head Attention, skip connections, and MLP">
  <figcaption>Figure 9. Components inside a Transformer Encoder block: Layer Normalization, Multi-Head Attention, residual skip connections, and MLP.</figcaption>
</figure>

<p>We’ll focus on building the core component first: the Multi-Head Attention module.</p>

<h4 id="multi-head-attention">Multi-Head Attention Module</h4>

<p>Multi-head attention is used to compute all pairwise self-attention interactions between image patches in the input image. The Multi-Head Attention module consists of smaller attention heads, each of which computes self-attention independently over different learned feature subspaces.</p>

<p>Let’s begin by implementing a single self-attention head first. Recall from our earlier Self-Attention project that an attention head takes a sequence of embeddings and computes the $K$, $Q$, and $V$ matrices:</p>

```python
class AttentionHead(nn.Module):
    """
    Single attention head.
    """

    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Recall that Attention(Q, K, V) = softmax(Q K^T / sqrt(head_size)) V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)
```

<ul>
  <li>Note that <code>attention_head_size</code> defines the reduced dimensionality per attention head.</li>
  <li>We define the $K$, $Q$, and $V$ transformations as fully connected <code>nn.Linear</code> layers that map the input embeddings from <code>hidden_size</code> to <code>attention_head_size</code>.</li>
</ul>

<p>In the Multi-Head Attention module, the outputs from all individual attention heads are concatenated and linearly projected to obtain the final output of the module.</p>

<p>Next, let’s implement the Multi-Head Attention module:</p>

```python
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.kqv_bias = config["kqv_bias"]
        self.heads = nn.ModuleList([])

        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.kqv_bias,
            )
            self.heads.append(head)

        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        attention_outputs = [head(x) for head in self.heads]
        attention_output = torch.cat(
            [attn_output for attn_output, _ in attention_outputs], dim=-1
        )
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)

        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack(
                [attn_probs for _, attn_probs in attention_outputs], dim=1
            )
            return (attention_output, attention_probs)
```

<ul>
  <li>Notice the seemingly redundant calculation for <code>attention_head_size</code> and <code>all_head_size</code>. Although <code>all_head_size</code> usually equals <code>hidden_size</code>, writing it this way keeps the implementation flexible if <code>hidden_size</code> is not evenly divisible or differs in custom configurations.</li>
  <li>Even when <code>all_head_size == hidden_size</code>, the <code>nn.Linear()</code> output projection is crucial for re-weighting and mixing the contributions from different attention heads.</li>
</ul>

<h4 id="completing-encoder">Completing the Transformer Encoder</h4>

<p>To complete one full Transformer layer, we also need to implement the MLP layer, two normalization layers, and residual skip connections.</p>

<p>For the MLP layer, we’ll build a two-layer feed-forward network with a GELU activation in between:</p>

```python
class MLP(nn.Module):
    """
    The Multi-Layer Perceptron module.
    """

    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = NewGELUActivation()
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x
```

<p>Now that we have implemented the Multi-Head Attention module and the MLP layer, we can combine them into a full Transformer layer with skip connections and layer normalization.</p>

<p>We’ll call this class <code>Block</code>, since multiple Transformer layers (or blocks) are stacked sequentially to form the complete Transformer Encoder:</p>

```python
class Block(nn.Module):
    """
    A single, full Transformer layer.
    """

    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, output_attentions=False):
        attention_output, attention_probs = self.attention(
            self.layernorm_1(x), output_attentions=output_attentions
        )
        x = x + attention_output  # skip connection
        mlp_output = self.mlp(self.layernorm_2(x))
        x = x + mlp_output  # skip connection

        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)
```

<p>Finally, we’ll build the full Transformer Encoder, which stacks multiple Transformer blocks sequentially:</p>

```python
class Encoder(nn.Module):
    """
    Transformer Encoder module.
    """

    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([])

        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        all_attentions = []

        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)

        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)
```

<h3 id="classification">3.3 Do Image Classification with the CLS Token</h3>

<p>Now that the full Transformer Encoder is implemented, we obtain updated, context-rich embeddings for the image patches and the <code>[CLS]</code> token.</p>

<figure class="article-figure">
  <img src="../assets/images/vit/10.png" alt="Using the [CLS] token output from the Transformer Encoder with an MLP classification head to predict image classes">
  <figcaption>Figure 10. Feeding the final <code>[CLS]</code> token representation into the classification head to predict class logits.</figcaption>
</figure>

<p>To turn these embeddings into useful predictions, we pass the <code>[CLS]</code> token embedding into a classification layer—a fully connected layer that takes the <code>[CLS]</code> embedding of size <code>hidden_size</code> and outputs logits for each image class.</p>

<p>Here’s the code implementation for the full ViT classifier model:</p>

```python
class ViTClassifier(nn.Module):
    """
    The full ViT model for image classification.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        self.embedding = Embeddings(config)
        self.encoder = Encoder(config)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        embedding_output = self.embedding(x)
        encoder_output, all_attentions = self.encoder(
            embedding_output, output_attentions=output_attentions
        )
        logits = self.classifier(encoder_output[:, 0])

        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=self.config["initializer_range"]
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)
```

<p>Note that <code>_init_weights()</code> ensures all parameters start with reasonable values to avoid unstable training. Using PyTorch’s <code>self.apply()</code>, it inspects each module type and applies the appropriate parameter initialization across every layer of the model.</p>

<h2 id="results">4. Results</h2>

<p>In this section, we’ll review the results from training our image-classifying ViT model. The full implementation was written in a Jupyter notebook.</p>

<p>First, here’s a quick preview of the $32 \times 32$ images contained in the CIFAR-10 dataset:</p>

<figure class="article-figure">
  <img src="../assets/images/vit/11.png" alt="Sample 32 by 32 CIFAR-10 images across classes including bird, horse, car, truck, plane, cat, ship, and dog">
  <figcaption>Figure 11. Sample $32 \times 32$ images from the CIFAR-10 dataset.</figcaption>
</figure>

<p>I trained the model for $50$ epochs, attaining a final test accuracy of $64\%$.</p>

<p>Here are two plots showing how training loss, validation loss, and validation accuracy changed over the epochs:</p>

<figure class="article-figure">
  <img src="../assets/images/vit/12.png" alt="Training loss, validation loss, and validation accuracy curves over 50 epochs">
  <figcaption>Figure 12. Training loss, validation loss, and validation accuracy across 50 epochs.</figcaption>
</figure>

<p>Perhaps most interesting of all, here’s a visualization of how the model computes attention over the images. We can see that the model generally learned to focus on the key subject of each image while ignoring the background:</p>

<figure class="article-figure">
  <img src="../assets/images/vit/13.png" alt="Attention heatmap visualizations over CIFAR-10 test images showing ground-truth and predicted labels">
  <figcaption>Figure 13. Attention maps over CIFAR-10 test images with ground-truth and predicted classes.</figcaption>
</figure>

<h2 id="references">5. References</h2>
