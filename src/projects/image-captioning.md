---
title: Image Captioning with VGG16 and Transformer
description: A step-by-step lecture walkthrough of building an end-to-end Image Captioning system using Selenium dataset scraping, pretrained VGG16 feature extraction, a custom Transformer encoder-decoder in TensorFlow/Keras, and Perplexity & BLEU evaluation.
lead: Building an end-to-end Image Captioning pipeline from scratch: collecting an astronomy dataset with Selenium, extracting visual features with pretrained VGG16, implementing a Transformer encoder-decoder in TensorFlow/Keras, and evaluating captions with Perplexity and BLEU.
date: 2026
difficulty: 4
output: blog/image-captioning.html
back_url: ../projects.html
back_label: Projects
---

<p>In this project, we walk through the complete pipeline for building an <strong>Image Captioning</strong> system that automatically generates descriptive natural-language captions from input images. Following a structured lecture progression, we cover problem framing, automated dataset collection with Selenium, image and text preprocessing, combining a pretrained <strong>VGG16</strong> CNN backbone with a custom <strong>Transformer</strong> encoder-decoder in TensorFlow/Keras, training with masked loss and warmup scheduling, autoregressive inference, and evaluation via <strong>Perplexity</strong> and <strong>BLEU</strong>.</p>

<nav class="article-toc" aria-label="Article table of contents">
  <p><strong>Contents</strong></p>
  <ol>
    <li><a href="#introduction">Introduction and Project Pipeline</a></li>
    <li><a href="#dataset">Dataset Structure and Web Scraping with Selenium</a></li>
    <li><a href="#method">Method: VGG16 + Transformer Architecture and Training</a>
      <ol>
        <li><a href="#model-design">Model Design Idea: Bridging Vision and Transformers</a></li>
        <li><a href="#preprocessing">Data Preprocessing and Augmentation Pipeline</a></li>
        <li><a href="#transformer-implementation">Implementing VGG16 and Transformer Modules</a></li>
        <li><a href="#scheduler-loss">Learning Rate Scheduler, Masked Loss, and Masked Accuracy</a></li>
        <li><a href="#training-inference">Training, Learning Curves, and Autoregressive Inference</a></li>
      </ol>
    </li>
    <li><a href="#evaluation">Evaluation Metrics: Perplexity and BLEU Score</a></li>
    <li><a href="#summary">Summary and References</a></li>
  </ol>
</nav>

<h2 id="introduction">1. Introduction and Project Pipeline</h2>

<p><strong>Text Generation</strong> is a broad NLP family that automatically produces new text conditioned on an input signal. Depending on the input modality, this family spans tasks such as Automatic Speech Recognition (audio to text), Machine Translation (source language to target language), Abstractive Text Summarization (long article to concise summary), and <strong>Image Captioning</strong> (image to descriptive sentence).</p>

<figure class="article-figure">
  <img src="../assets/images/img_captioning/1.png" alt="Overview of text generation tasks including Image Captioning, Automatic Speech Recognition, Machine Translation, and Abstractive Text Summarization">
  <figcaption>Figure 1. Image Captioning within the wider landscape of sequence-to-sequence and multimodal text generation tasks.</figcaption>
</figure>

<p>Formally, <strong>Image Captioning</strong> is the multimodal task of taking an input image and generating a fluent, semantically accurate sentence describing what appears in the scene:</p>

<figure class="article-figure">
  <img src="../assets/images/img_captioning/avt.png" alt="High-level Image Captioning diagram feeding an aurora photo into an Image Captioning Model to produce a text caption">
  <figcaption>Figure 2. High-level Image Captioning workflow mapping an input image to a natural language description.</figcaption>
</figure>

<p>Standard benchmarks such as Flickr8k/30k and MS-COCO pair everyday scenes (sports, pets, people outdoors) with human-written descriptions:</p>

<figure class="article-figure">
  <img src="../assets/images/img_captioning/2.png" alt="Classic Image Captioning examples showing everyday scenes paired with descriptive captions">
  <figcaption>Figure 3. Classic Image Captioning examples pairing scenes with concise natural language descriptions.</figcaption>
</figure>

<p>In this project, we apply Image Captioning to a specialized domain: <strong>space and astronomy imagery</strong>. Given a space photograph (such as galaxies, nebulae, planets, or spacecraft), our model generates a textual caption describing the astrophysical event. The end-to-end engineering workflow breaks down into four stages:</p>

<ol>
  <li><strong>Collecting the dataset:</strong> Scrape paired astronomy images and captions from science articles.</li>
  <li><strong>Preparing training data:</strong> Normalize text, build a vocabulary, resize and augment images, and align decoder inputs/targets.</li>
  <li><strong>Training and evaluating the model:</strong> Combine VGG16 with a Transformer encoder-decoder and measure performance with Perplexity and BLEU.</li>
  <li><strong>Writing the inference program:</strong> Generate captions autoregressively token by token for unseen test images.</li>
</ol>

<figure class="article-figure">
  <img src="../assets/images/img_captioning/3.png" alt="Space Image Captioning problem description and four-step project pipeline from dataset collection to inference">
  <figcaption>Figure 4. Problem formulation and the four-stage Image Captioning project pipeline.</figcaption>
</figure>

<h2 id="dataset">2. Dataset Structure and Web Scraping with Selenium</h2>

<h3 id="dataset-structure">2.1 Dataset Structure</h3>

<p>Our collected dataset lives inside a root directory <code>space_image_captioning_dataset/</code> containing two components: a folder <code>space_images/</code> holding sequentially named JPEG files (<code>IMG_000001.jpg</code>, <code>IMG_000002.jpg</code>, ...) and a UTF-8 text file <code>space_captions.txt</code> where each line stores a tab-separated pair <code>image_path\tcaption</code>.</p>

<figure class="article-figure">
  <img src="../assets/images/img_captioning/4.png" alt="Dataset folder hierarchy showing space_images directory with numbered JPG files and tab-separated space_captions.txt">
  <figcaption>Figure 5. Directory structure of the Space Image Captioning dataset and the tab-separated caption index file.</figcaption>
</figure>

```python
import os

root_dir = "./space_image_captioning_dataset"
images_dir = os.path.join(root_dir, "space_images")
captions_path = os.path.join(root_dir, "space_captions.txt")

os.makedirs(root_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
caption_file = open(captions_path, "w", encoding="utf8")
```

<h3 id="web-scraping">2.2 Web Scraping Pipeline with Selenium</h3>

<p>To assemble a domain-specific dataset from web articles, we use <strong>Selenium</strong> to drive a browser, traverse paginated article listings, inspect the DOM via XPath, and download each valid image together with its caption.</p>

<figure class="article-figure">
  <img src="../assets/images/img_captioning/5.png" alt="Web scraping workflow using Selenium to crawl URLs, parse HTML pages, extract image links, and save data">
  <figcaption>Figure 6. Automated web scraping pipeline using Selenium to collect paired images and captions.</figcaption>
</figure>

<p>Real-world web pages are messy, so a reliable scraper must handle several edge cases cleanly:</p>

<ul>
  <li><strong>Pagination and XPath discovery:</strong> Loop across listing pages (<code>/page1</code> to <code>/page100</code>), collect article links matching <code>//a[@class="title"]</code>, and navigate into each article to locate <code>div[@class="img-box"]</code> containers.</li>
  <li><strong>Missing tags and timeouts:</strong> Wrap element lookups inside <code>WebDriverWait</code> and <code>try...except</code> blocks so missing images or broken tags never crash the crawler.</li>
  <li><strong>Caption fallback:</strong> Some figure boxes omit the <code>&lt;em&gt;</code> caption element and store the description in the image’s <code>alt</code> attribute instead. Falling back to <code>img_tag.get_attribute("alt")</code> recovers these pairs.</li>
  <li><strong>Filtering and RGB conversion:</strong> Skip animated <code>.gif</code> files and empty captions, fetch raw image bytes with <code>requests.get()</code> + <code>BytesIO</code>, and convert palette-mode images (<code>img.mode == "P"</code>) to standard <code>"RGB"</code> before saving as JPEG.</li>
</ul>

```python
from io import BytesIO
import requests
from PIL import Image
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Inside each article page after locating img_box_tags:
for img_box_tag in img_box_tags:
    wait = WebDriverWait(img_box_tag, 5)
    try:
        img_tag = wait.until(EC.visibility_of_element_located((By.TAG_NAME, "img")))
    except Exception:
        continue

    try:
        caption_tag = wait.until(EC.visibility_of_element_located((By.TAG_NAME, "em")))
        img_caption = caption_tag.text.strip()
    except Exception:
        img_caption = (img_tag.get_attribute("alt") or "").strip()

    img_url = img_tag.get_attribute("src") or ""
    if not img_url or img_url.lower().endswith(".gif") or not img_caption:
        continue

    try:
        img_resp = requests.get(img_url, timeout=10)
        img = Image.open(BytesIO(img_resp.content))
    except Exception:
        continue

    if img.mode != "RGB":
        img = img.convert("RGB")

    img_name = f"IMG_{img_idx:06d}.jpg"
    img_save_path = os.path.join(images_dir, img_name)
    img.save(img_save_path)
    caption_file.write(f"{img_save_path}\t{img_caption}\n")
    img_idx += 1
```

<h2 id="method">3. Method: VGG16 + Transformer Architecture and Training</h2>

<h3 id="model-design">3.1 Model Design Idea: Bridging Vision and Transformers</h3>

<p>The original Transformer architecture (Vaswani et al., 2017) expects a <strong>sequence of dense embedding vectors</strong> as input to its Encoder. We cannot feed a raw 3D RGB image tensor of shape $(H, W, 3)$ directly into a standard Transformer Encoder.</p>

<p>How do we convert an image into a meaningful sequence of vectors? One naive idea would be to flatten each of the $3$ color channels into a long vector, creating a sequence of just $3$ channel vectors. However, raw pixel values carry almost no high-level semantic meaning and destroy 2D spatial locality.</p>

<p>Instead, we use a pretrained <strong>VGG16</strong> convolutional neural network (trained on ImageNet, with its final classification head removed) as a visual feature extractor. Passing an $(H, W, 3)$ image through VGG16 produces a compact spatial feature map of shape $(H', W', 512)$, which we reshape into a sequence of $H' \times W'$ spatial feature tokens, each of dimension $512$. That sequence enters the Transformer Encoder, while the embedded caption tokens enter the Transformer Decoder.</p>

<figure class="article-figure">
  <img src="../assets/images/img_captioning/6.png" alt="Adapting the original Transformer architecture into a VGG16 plus Transformer encoder-decoder for Image Captioning">
  <figcaption>Figure 7. Adapting the original Transformer into a multimodal VGG16 + Transformer Image Captioning architecture.</figcaption>
</figure>

<figure class="article-figure">
  <img src="../assets/images/img_captioning/7.png" alt="Removing the classification head of pretrained ImageNet VGG16 to extract visual features for the Transformer Encoder">
  <figcaption>Figure 8. Using pretrained VGG16 (without the top classification head) as the visual feature extractor.</figcaption>
</figure>

<figure class="article-figure">
  <img src="../assets/images/img_captioning/8.png" alt="Comparison between naive 3-channel RGB flattening and extracting a sequence of (h*w, 512) spatial feature vectors via VGG16">
  <figcaption>Figure 9. Why VGG16 feature maps $(H' \times W', 512)$ provide a far richer visual token sequence than flattened raw RGB channels.</figcaption>
</figure>

<p>During training, every sample supplies a triplet built from the image and its ground-truth caption (teacher forcing):</p>

<ul>
  <li><strong>Encoder Input:</strong> The preprocessed input image passed through VGG16.</li>
  <li><strong>Decoder Input:</strong> The tokenized caption shifted right (<code>cap[:-1]</code>, starting with <code>&lt;start&gt;</code>).</li>
  <li><strong>Decoder Output (Target):</strong> The next-token target sequence (<code>cap[1:]</code>, ending with <code>&lt;end&gt;</code>).</li>
</ul>

<figure class="article-figure">
  <img src="../assets/images/img_captioning/9.png" alt="Mapping training inputs X (image for Encoder and shifted caption for Decoder) and target output y (next tokens from Softmax)">
  <figcaption>Figure 10. Teacher-forcing setup: image features feed the Encoder, shifted caption tokens feed the Decoder, and the model predicts the next token at every position.</figcaption>
</figure>

<h3 id="preprocessing">3.2 Data Preprocessing and Augmentation Pipeline</h3>

<h4>Text Normalization and Corpus Filtering</h4>

<p>Every raw caption undergoes five normalization steps: lowercasing, removing punctuation, collapsing repeated whitespace, trimming leading/trailing spaces, and wrapping the sentence with special <code>&lt;start&gt;</code> and <code>&lt;end&gt;</code> tokens. Captions shorter than $5$ words or longer than <code>MAX_SEQ_LEN - 2</code> are filtered out so the training distribution stays consistent.</p>

```python
import re


def text_normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = "<start> " + text + " <end>"
    return text


with open(captions_filepath, "r", encoding="utf8") as f:
    lines = f.readlines()

dataset_dict = {}
skipped_images_lst = []
corpus = []

for line in lines:
    line = line.rstrip("\n")
    img_path, caption = line.split("\t")
    tokens = caption.split(" ")

    if len(tokens) < 5 or len(tokens) > (MAX_SEQ_LEN - 2):
        skipped_images_lst.append(img_path)
        continue

    normalized_caption = text_normalize(caption)
    corpus.append(normalized_caption)
    dataset_dict[img_path] = normalized_caption
```

<figure class="article-figure">
  <img src="../assets/images/img_captioning/10.png" alt="Sample astronomy images paired with normalized captions wrapped in start and end tokens">
  <figcaption>Figure 11. Examples of astronomy images paired with normalized <code>&lt;start&gt; ... &lt;end&gt;</code> captions.</figcaption>
</figure>

<h4>Image Preprocessing and Data Augmentation</h4>

<p>Each image is read from disk, decoded as a 3-channel JPEG, resized to <code>IMG_SIZE</code>, and cast to <code>tf.float32</code>. For the training split, lightweight on-the-fly augmentation (horizontal flip, small random rotation, and contrast jitter) helps prevent overfitting on a modestly sized custom dataset.</p>

<figure class="article-figure article-figure-small">
  <img src="../assets/images/img_captioning/11.png" alt="Four-step image preprocessing pipeline: Read Image, Decode JPEG, Resize, and cast from uint8 to float32">
  <figcaption>Figure 12. Image preprocessing steps prior to feeding tensors into VGG16.</figcaption>
</figure>

```python
import tensorflow as tf


def preprocess_img(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


img_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomContrast(0.3),
])
```

<figure class="article-figure article-figure-small">
  <img src="../assets/images/img_captioning/12.png" alt="Original image compared against randomly flipped, rotated, and contrast-adjusted augmented versions">
  <figcaption>Figure 13. Visual effect of random flip, rotation, and contrast augmentation on training images.</figcaption>
</figure>

<h4>Text Vectorization, Padding, and Dataset Splits</h4>

<p>Next, we adapt a <code>tf.keras.layers.TextVectorization</code> layer on the normalized corpus to map words to integer IDs and pad every sequence with <code>0</code> up to <code>MAX_SEQ_LEN</code>. Notice we pass <code>standardize=None</code> so our custom <code>&lt;start&gt;</code> and <code>&lt;end&gt;</code> tokens are preserved verbatim.</p>

<figure class="article-figure">
  <img src="../assets/images/img_captioning/13.png" alt="Diagram showing input tokens padded to MAX_SEQ_LEN = 7">
  <figcaption>Figure 14. Padding tokenized sequences to a fixed length <code>MAX_SEQ_LEN</code> for batched training.</figcaption>
</figure>

```python
import numpy as np

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_SEQ_LEN,
    standardize=None,
)
vectorizer.adapt(corpus)

# Randomly shuffle and split 70% train / 20% validation / 10% test
X = np.array(list(dataset_dict.keys()))
y = np.array(list(dataset_dict.values()))

n_samples = len(X)
train_idx = int(n_samples * 0.7)
val_idx = train_idx + int(n_samples * 0.2)

idxs = np.random.permutation(n_samples)
X, y = X[idxs], y[idxs]

X_train, y_train = X[:train_idx], y[:train_idx]
X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
X_test, y_test = X[val_idx:], y[val_idx:]
```

<p>Finally, we assemble asynchronous <code>tf.data.Dataset</code> pipelines that apply augmentation only to the training split and slice each vectorized caption into <code>decoder_input = cap[:-1]</code> and <code>decoder_output = cap[1:]</code>:</p>

```python
AUTOTUNE = tf.data.AUTOTUNE


def process_input_data(img_path, caption, augment=False):
    img = preprocess_img(img_path)
    if augment:
        img = img_augmentation(img)
    cap = vectorizer(caption)
    decoder_input = cap[:-1]
    decoder_output = cap[1:]
    return (img, decoder_input), decoder_output


train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
    .map(lambda x, y: process_input_data(x, y, augment=True), num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
    .map(process_input_data, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
    .map(process_input_data, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
)
```

<h3 id="transformer-implementation">3.3 Implementing VGG16 and Transformer Modules</h3>

<p>Now we build every block of our VGG16 + Transformer architecture cleanly from the ground up:</p>

<figure class="article-figure">
  <img src="../assets/images/img_captioning/14.png" alt="Full Image Captioning architecture connecting Input Image through VGG16 to Transformer Encoder, and Input Caption Embedding plus Positional Encoding through Transformer Decoder to Linear and Softmax">
  <figcaption>Figure 15. Complete Image Captioning architecture implemented in TensorFlow/Keras.</figcaption>
</figure>

<h4>1. Sinusoidal Positional Encoding and Positional Embedding</h4>

<p>Because Transformers process tokens in parallel without recurrence, we inject word order using sinusoidal positional encodings added to the scaled token embeddings ($\sqrt{d_{\text{model}}}$):</p>

```python
def positional_encoding(length, depth):
    depth = depth // 2
    positions = np.arange(length)[:, np.newaxis]       # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth   # (1, depth)

    angle_rates = 1 / (10000 ** depths)                # (1, depth)
    angle_rads = positions * angle_rates               # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)], axis=-1
    )
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # Scale embeddings relative to positional encoding magnitude
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x
```

<h4>2. Attention Layers and Feed-Forward Network</h4>

<p>All three attention variants (Global Self-Attention in the Encoder, Causal Self-Attention in the Decoder to prevent peeking at future words, and Cross-Attention bridging caption queries with visual encoder keys/values) inherit from a shared <code>BaseAttention</code> block with residual addition and Layer Normalization:</p>

```python
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True,
        )
        # Cache attention weights for visualization
        self.last_attn_scores = attn_scores
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True,
        )
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation="relu"),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate),
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x
```

<h4>3. Pretrained VGG16 Visual Backbone</h4>

<p>We load ImageNet-pretrained <code>VGG16</code> with <code>include_top=False</code>, freeze its weights, and reshape its convolutional output feature grid into a flat sequence of 512-dimensional spatial vectors:</p>

<figure class="article-figure article-figure-small">
  <img src="../assets/images/img_captioning/15.png" alt="VGG16 convolutional blocks downsampling spatial dimensions while increasing depth from 64 to 512">
  <figcaption>Figure 16. VGG16 convolutional hierarchy producing 512-channel spatial feature maps.</figcaption>
</figure>

```python
def get_cnn_model():
    base_model = tf.keras.applications.vgg16.VGG16(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False
    base_model_out = base_model.output
    base_model_out = tf.keras.layers.Reshape(
        (-1, base_model_out.shape[-1])
    )(base_model_out)

    cnn_model = tf.keras.models.Model(base_model.input, base_model_out)
    return cnn_model
```

<h4>4. Transformer Encoder, Decoder, and Full ImageCaptioningModel</h4>

<p>The <code>Encoder</code> normalizes and projects VGG16’s 512-channel features down to <code>d_model</code> before passing them through $N$ encoder layers. The <code>Decoder</code> embeds the partial caption, applies masked causal self-attention, attends to the visual encoder context via cross-attention, and feeds the final hidden states into a <code>Dense(vocab_size)</code> projection:</p>

```python
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )
        self.ffn = FeedForward(d_model, dff, dropout_rate=dropout_rate)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.enc_layers = [
            EncoderLayer(
                d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.ffn = tf.keras.layers.Dense(d_model, activation="relu")

    def call(self, x):
        x = self.layernorm(x)
        x = self.ffn(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )
        self.cross_attention = CrossAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )
        self.ffn = FeedForward(d_model, dff, dropout_rate=dropout_rate)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)
        self.last_attn_scores = self.cross_attention.last_attn_scores
        x = self.ffn(x)
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(
                d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]
        self.last_attn_scores = None

    def call(self, x, context):
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        return x


class ImageCaptioningModel(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.cnn_model = get_cnn_model()
        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=vocab_size,
            dropout_rate=dropout_rate,
        )
        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=vocab_size,
            dropout_rate=dropout_rate,
        )
        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs):
        img, x = inputs
        context = self.cnn_model(img)
        context = self.encoder(context)
        x = self.decoder(x, context)
        logits = self.final_layer(x)

        try:
            del logits._keras_mask
        except AttributeError:
            pass
        return logits
```

<p>Instantiating the model with <code>N_LAYERS = 4</code>, <code>D_MODEL = 128</code>, <code>D_FF = 512</code>, <code>N_HEADS = 8</code>, and <code>DROPOUT_RATE = 0.2</code> yields roughly <strong>22.8 million parameters</strong> (14.7M in VGG16, 2.7M in the Transformer Encoder, 5.1M in the Decoder, and 0.33M in the output vocabulary projection).</p>

<h3 id="scheduler-loss">3.4 Learning Rate Scheduler, Masked Loss, and Masked Accuracy</h3>

<p>Following the original Transformer paper, we update the learning rate at every training step (batch) using linear warmup followed by inverse square-root decay:</p>

<div class="equation-block" role="math" aria-label="Transformer warmup learning rate schedule equation">
  \[
    \text{lrate} = d_{\text{model}}^{-0.5} \cdot \min\!\left(\text{step\_num}^{-0.5},\; \text{step\_num} \cdot \text{warmup\_steps}^{-1.5}\right)
  \]
</div>

```python
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
```

<p>Because captions are padded with token index <code>0</code> up to <code>MAX_SEQ_LEN</code>, computing standard cross-entropy across every position would let padding tokens dominate the gradients. We therefore mask out all <code>label == 0</code> positions in both <code>masked_loss</code> and <code>masked_accuracy</code>:</p>

```python
def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    loss = loss_object(label, pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = (label == pred) & (label != 0)
    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(label != 0, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)
```

<h3 id="training-inference">3.5 Training, Learning Curves, and Autoregressive Inference</h3>

<p>We train the model for 50 epochs using <code>ic_model.fit(train_ds, epochs=50, validation_data=val_ds)</code> and plot training and validation loss and accuracy:</p>

<figure class="article-figure">
  <img src="../assets/images/img_captioning/16.png" alt="Training loss, training accuracy, validation loss, and validation accuracy plotted across 50 epochs">
  <figcaption>Figure 17. Training and validation loss/accuracy curves across 50 epochs (validation loss bottoms out near epoch 30, highlighting the benefit of early stopping).</figcaption>
</figure>

<p>As seen in Figure 17, training loss drops steadily from ~7.9 to ~1.3 while training accuracy climbs above 70%. Validation loss reaches its minimum around epoch 30 before gently ticking upward as the model begins to overfit the small custom dataset, while validation token accuracy plateaus around 31%.</p>

<p>At inference time, we generate a caption autoregressively: starting with <code>output_str = "&lt;start&gt;"</code>, we repeatedly feed the image and current partial caption into the model, take the highest-probability next token via greedy <code>np.argmax</code>, and stop as soon as <code>&lt;end&gt;</code> or <code>MAX_SEQ_LEN</code> is reached:</p>

```python
def generate_caption(img_path, ic_model, vectorizer):
    img = preprocess_img(img_path)
    display_img = img.numpy().clip(0, 255).astype(np.uint8)
    img_batch = np.expand_dims(display_img, axis=0)

    output_str = "<start>"
    for idx in range(MAX_SEQ_LEN):
        tokenized_caption = vectorizer([output_str])[:, :-1]
        pred = ic_model.predict((img_batch, tokenized_caption), verbose=0)
        sampled_token_idx = np.argmax(pred[0, idx, :])
        sampled_token = index_lookup[sampled_token_idx]
        if sampled_token == "<end>":
            break
        output_str += f" {sampled_token}"

    generated_caption = output_str.replace("<start>", "").replace("<end>", "").strip()
    return display_img, generated_caption
```

<figure class="article-figure article-figure-small">
  <img src="../assets/images/img_captioning/17.png" alt="Sample inference result showing an asteroid impacting Earth with a generated caption describing a meteor hitting Earth">
  <figcaption>Figure 18. Example inference prediction generated by the trained model on an unseen space image.</figcaption>
</figure>

<h2 id="evaluation">4. Evaluation Metrics: Perplexity and BLEU Score</h2>

<p>Token-level accuracy alone does not fully measure sentence quality in captioning. Two standard language generation metrics complement each other well:</p>

<h3 id="perplexity">4.1 Perplexity</h3>

<p><strong>Perplexity</strong> measures how uncertain (surprised) the language model is when predicting a ground-truth token sequence $W = (w_1, w_2, \dots, w_N)$. Lower perplexity indicates higher confidence and better calibration:</p>

<div class="equation-block" role="math" aria-label="Perplexity equation">
  \[
    PP(W)
    = 2^{-\frac{1}{N}\log_2 P(w_1, w_2, \dots, w_N)}
    = P(w_1, w_2, \dots, w_N)^{-\frac{1}{N}}
    = \sqrt[N]{\frac{1}{P(w_1, w_2, \dots, w_N)}}
  \]
</div>

<p>In TensorFlow, since average cross-entropy loss equals the negative log-likelihood per token, perplexity is simply $\exp(\text{mean cross-entropy loss})$:</p>

```python
def compute_perplexity(logits, targets):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    loss = loss_fn(targets, logits)
    perplexity = np.exp(np.mean(loss))
    return perplexity
```

<h3 id="bleu-score">4.2 BLEU Score</h3>

<p><strong>BLEU (Bilingual Evaluation Understudy)</strong> measures n-gram overlap between generated captions and human reference captions, combined with a <strong>Brevity Penalty</strong> $p$ to prevent the model from gaming precision by outputting overly short fragments:</p>

<div class="equation-block" role="math" aria-label="BLEU score and brevity penalty equations">
  \[
    P = \frac{m}{w_t}, \qquad
    p = \begin{cases}
      1 & \text{if } c > r \\
      e^{1 - \frac{r}{c}} & \text{if } c \le r
    \end{cases}, \qquad
    \text{BLEU} = p \cdot \exp\!\left(\sum_{n=1}^{N} \frac{1}{N} \log P_n\right)
  \]
</div>

<p>where $m$ is the number of matched n-grams, $w_t$ is the total number of predicted n-grams, $c$ is the candidate caption length, and $r$ is the reference caption length:</p>

```python
from nltk.translate.bleu_score import corpus_bleu


def compute_bleu(predicted, targets):
    predicted_strings = []
    for seq in predicted:
        seq = np.argmax(seq, axis=1)
        string_seq = " ".join(
            [tokenizer.sequences_to_texts([[token]])[0] for token in seq if token != 0]
        )
        predicted_strings.append(string_seq)

    target_strings = []
    for seq in targets:
        seq = seq.numpy().tolist()
        string_seq = " ".join(
            [tokenizer.sequences_to_texts([[token]])[0] for token in seq if token != 0]
        )
        target_strings.append([string_seq])

    return corpus_bleu(target_strings, predicted_strings)
```

<h2 id="summary">5. Summary and References</h2>

<p>To recap the complete Image Captioning workflow:</p>

<ul>
  <li><strong>End-to-end data pipeline:</strong> Automated dataset creation via Selenium paired with regex text normalization, image augmentation, and <code>tf.data</code> batching.</li>
  <li><strong>Multimodal architecture:</strong> Pretrained VGG16 extracts spatial visual tokens $(H' \times W', 512)$ while a custom Transformer encoder-decoder aligns image regions with caption words through cross-attention.</li>
  <li><strong>Stable sequence training:</strong> Warmup learning-rate scheduling and padding-aware masked loss/accuracy keep gradient updates focused strictly on real tokens.</li>
  <li><strong>Quantitative evaluation:</strong> Combining Perplexity (probabilistic fluency) and BLEU (n-gram overlap against references) provides a balanced view of caption quality.</li>
</ul>

<h3 id="references">References</h3>

<ul>
  <li>Ashish Vaswani et al., <a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noopener noreferrer">Attention Is All You Need</a> (NeurIPS 2017).</li>
  <li>Karen Simonyan and Andrew Zisserman, <a href="https://arxiv.org/abs/1409.1556" target="_blank" rel="noopener noreferrer">Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG16)</a> (ICLR 2015).</li>
  <li>Kishore Papineni et al., <a href="https://aclanthology.org/P02-1040/" target="_blank" rel="noopener noreferrer">BLEU: a Method for Automatic Evaluation of Machine Translation</a> (ACL 2002).</li>
  <li>TensorFlow Official Tutorial, <a href="https://www.tensorflow.org/text/tutorials/transformer" target="_blank" rel="noopener noreferrer">Neural Machine Translation with a Transformer and Keras</a>.</li>
  <li>Dinh-Thang Duong (AI VIETNAM All-in-One Course), <em>Module 7: Image Captioning Project</em> lecture slides.</li>
</ul>
