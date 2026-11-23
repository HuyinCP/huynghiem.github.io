---
title: Visual Question Answering (VQA): From Baselines to LVLMs
description: A complete lecture walkthrough of Visual Question Answering (VQA), covering problem framing, VQAv2 and ViInfographicVQA datasets, MLP, CNN+BiLSTM, and ViT+BERT implementations in PyTorch, fusion and post-fusion ablation studies, language bias debiasing, and 4-bit LLaVA inference.
lead: Building and benchmarking Visual Question Answering systems step by step: from flattened-pixel MLP and ResNet50 + BiLSTM baselines to fine-tuned ViT + BERT, fusion ablations, Bottom-Up Top-Down attention, counterfactual debiasing, and zero-shot LLaVA-1.5 inference.
date: 2026
difficulty: 4
output: blog/visual-question-answering.html
back_url: ../projects.html
back_label: Projects
---

<p>In this project, we walk step by step through <strong>Visual Question Answering (VQA)</strong>, one of the foundational benchmarks of multimodal deep learning. Given an input image and a free-form natural language question about that image, a VQA system must extract visual semantics, understand linguistic intent, align both modalities, and output an accurate answer. Following a structured lecture progression, we start from core problem framing and datasets (<strong>VQAv2</strong> and <strong>ViInfographicVQA</strong>), implement three progressively stronger architectures in PyTorch (<strong>MLP</strong>, <strong>CNN + BiLSTM</strong>, and <strong>ViT + BERT</strong>), run controlled ablations on fusion operators and post-fusion networks, and conclude with modern research directions including <strong>Bottom-Up and Top-Down Attention</strong>, <strong>language bias debiasing (VQA-CP, CSS, RUBi)</strong>, and <strong>Large Vision-Language Models (LLaVA)</strong>.</p>

<nav class="article-toc" aria-label="Article table of contents">
  <p><strong>Contents</strong></p>
  <ol>
    <li><a href="#introduction">Introduction to Visual Question Answering and Datasets</a>
      <ol>
        <li><a href="#problem-framing">What Is VQA and How Is It Formulated?</a></li>
        <li><a href="#datasets">Benchmark Datasets: VQAv2 and ViInfographicVQA</a></li>
      </ol>
    </li>
    <li><a href="#mlp-approach">Approach 1: Multi-Layer Perceptron (MLP) Baseline and Metrics</a>
      <ol>
        <li><a href="#mlp-dataset">Dataset Structure and PyTorch DataLoader</a></li>
        <li><a href="#mlp-model">Flattened Pixels + TF-IDF Architecture</a></li>
        <li><a href="#vqa-metrics">Evaluation Metrics: VQA Accuracy and ANLS</a></li>
      </ol>
    </li>
    <li><a href="#cnn-lstm-approach">Approach 2: CNN + BiLSTM Baseline</a>
      <ol>
        <li><a href="#cnn-lstm-architecture">Pretrained ResNet50 and Bidirectional LSTM</a></li>
        <li><a href="#cnn-lstm-results">MLP vs. CNN + BiLSTM Results</a></li>
      </ol>
    </li>
    <li><a href="#vit-bert-approach">Approach 3: ViT + BERT Baseline</a>
      <ol>
        <li><a href="#vit-bert-architecture">Vision Transformer and BERT / PhoBERT Encoders</a></li>
        <li><a href="#frozen-vs-unfrozen">Frozen vs. Fine-Tuned Encoders</a></li>
      </ol>
    </li>
    <li><a href="#experiments">Experiments and Ablation Studies</a>
      <ol>
        <li><a href="#main-results">Main Benchmark Results on VQAv2 and ViInfographicVQA</a></li>
        <li><a href="#ablation-fusion">Ablation Study 1: Multimodal Fusion Operations</a></li>
        <li><a href="#ablation-post-fusion">Ablation Study 2: Post-Fusion Networks</a></li>
      </ol>
    </li>
    <li><a href="#discussion">Discussion and Modern Directions</a>
      <ol>
        <li><a href="#butd-attention">Bottom-Up and Top-Down Attention</a></li>
        <li><a href="#language-bias">Language Bias and Debiasing (VQA-CP, CSS, RUBi)</a></li>
        <li><a href="#lvlms-llava">Large Vision-Language Models and 4-Bit LLaVA Inference</a></li>
      </ol>
    </li>
    <li><a href="#summary">Summary and References</a></li>
  </ol>
</nav>

<h2 id="introduction">1. Introduction to Visual Question Answering and Datasets</h2>

<h3 id="problem-framing">1.1 What Is VQA and How Is It Formulated?</h3>

<p><strong>Visual Question Answering (VQA)</strong> is a multimodal machine learning task that receives an image \(I\) together with a natural language question \(Q\) and predicts an answer \(A\) grounded in the visual scene. Unlike single-modality tasks such as image classification or text sentiment analysis, VQA demands fine-grained cross-modal reasoning: even two nearly identical scenes can require opposite answers when a single visual relationship changes.</p>

<figure class="article-figure">
  <img src="../assets/images/vqa/1.png" alt="Visual Question Answering definition with two complementary scenes of a girl and a bicycle yielding opposite yes and no answers">
  <figcaption>Figure 1. Visual Question Answering on complementary scenes: answering "Is the girl walking the bike?" requires grounding both objects and their spatial interaction.</figcaption>
</figure>

<p>Almost every classical VQA architecture follows a three-stage blueprint: (1) <strong>Features Extraction</strong>, where an image encoder maps pixels to a visual representation and a question encoder maps tokens to a textual representation; (2) <strong>Features Fusion</strong>, which combines both modalities into a joint representation; and (3) an <strong>Answer Generator</strong>, which maps the fused features to the final answer.</p>

<figure class="article-figure">
  <img src="../assets/images/vqa/avt.png" alt="Three-stage Visual Question Answering framework: Features Extraction, Features Fusion, and Answer Generator">
  <figcaption>Figure 2. The standard three-stage VQA pipeline: Features Extraction, Features Fusion, and Answer Generation.</figcaption>
</figure>

<p>When designing a VQA system, two high-level architectural choices shape the entire training pipeline:</p>

<ul>
  <li><strong>Classification vs. Sequence-to-Sequence (Seq-to-Seq):</strong> Treating VQA as multi-class classification restricts predictions to a fixed vocabulary of the top-\(K\) most frequent training answers (for example, <code>"yes"</code>, <code>"no"</code>, <code>"2"</code>, <code>"Feverfew"</code>). This simplifies optimization with standard Cross-Entropy loss and works well for short factual responses, whereas a Seq-to-Seq decoder generates open-ended full sentences token by token (such as <code>"This is a Tanacetum Parthenium."</code>).</li>
  <li><strong>Early Fusion vs. Late Fusion:</strong> In <em>Early Fusion</em>, visual and textual inputs are merged before deep joint processing so cross-modal interactions can occur across all layers. In <em>Late Fusion</em>, each modality passes through its own dedicated encoder first, and the high-level embeddings are aggregated near the decision head.</li>
</ul>

<figure class="article-figure">
  <img src="../assets/images/vqa/2.png" alt="Comparison between formulating VQA as closed-set classification versus open-ended sequence-to-sequence generation">
  <figcaption>Figure 3. Formulating VQA as closed-vocabulary classification (left) versus autoregressive sequence-to-sequence generation (right).</figcaption>
</figure>

<figure class="article-figure">
  <img src="../assets/images/vqa/3.png" alt="Diagram contrasting Early Fusion with a Joint Modality Encoder against Late Fusion with separate modality networks and late aggregation">
  <figcaption>Figure 4. Multimodal fusion strategies: Early Fusion (left) versus Late Fusion (right).</figcaption>
</figure>

<p>Even for humans, visual questions span a wide spectrum of reasoning difficulty: object recognition, color and attribute identification, spatial relations, reading embedded scene text (OCR), and object counting. Counting questions illustrate this gap clearly: counting all giraffes in a scene already requires object localization, while counting only the giraffes currently drinking water requires combining localization with fine-grained action recognition.</p>

<figure class="article-figure">
  <img src="../assets/images/vqa/4.png" alt="Comparison of simple counting (How many giraffes?) versus complex conditional counting (How many giraffes are drinking water?)">
  <figcaption>Figure 5. Simple object counting (left) versus conditional counting requiring action recognition and visual reasoning (right).</figcaption>
</figure>

<h3 id="datasets">1.2 Benchmark Datasets: VQAv2 and ViInfographicVQA</h3>

<p>To evaluate models under very different visual regimes, our experiments use curated subsets of two complementary benchmarks:</p>

<ul>
  <li><strong>VQAv2 (English, Natural Scenes):</strong> Built on real-world MS COCO photographs, covering Yes/No, numeric, and short open-ended questions. A key design contribution of VQAv2 is <em>balanced pairs</em>: for almost every question, the dataset includes two visually similar images that yield different answers (for example, <code>"Is the umbrella upside down?"</code> answered <code>"yes"</code> on one photo and <code>"no"</code> on another), penalizing models that guess purely from text statistics.</li>
  <li><strong>ViInfographicVQA (Vietnamese, Dense Text Infographics):</strong> Composed of Vietnamese infographics, posters, and statistical charts where answers frequently depend on reading numbers, legends, charts, and structured layout rather than recognizing natural objects alone.</li>
</ul>

<figure class="article-figure">
  <img src="../assets/images/vqa/5.png" alt="Visual comparison of natural scene photos from VQAv2 on the left and dense Vietnamese charts from ViInfographicVQA on the right">
  <figcaption>Figure 6. Domain contrast between natural photographs in VQAv2 (left) and text-dense charts and posters in ViInfographicVQA (right).</figcaption>
</figure>

<figure class="article-figure">
  <img src="../assets/images/vqa/6.png" alt="VQAv2 balanced pairs showing pairs of similar scenes with identical questions and contrasting ground-truth answers">
  <figcaption>Figure 7. VQAv2 balanced pairs mitigate language shortcuts by pairing each question with visually similar scenes that have opposite answers.</figcaption>
</figure>

<div class="article-table-wrap">
  <table>
    <thead>
      <tr>
        <th>Dataset</th>
        <th>Language</th>
        <th class="num">Original Images</th>
        <th class="num">Original QA Pairs</th>
        <th class="num">Project Images</th>
        <th class="num">Project QA Pairs</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>VQAv2</strong></td>
        <td>English</td>
        <td class="num">~204,000</td>
        <td class="num">~1,100,000</td>
        <td class="num">4,417</td>
        <td class="num">7,000</td>
      </tr>
      <tr>
        <td><strong>ViInfographicVQA</strong></td>
        <td>Vietnamese</td>
        <td class="num">6,747</td>
        <td class="num">20,409</td>
        <td class="num">1,570</td>
        <td class="num">7,000</td>
      </tr>
    </tbody>
  </table>
  <p class="article-table-caption">Table 1. Statistics of the full benchmark datasets and the balanced 7,000-QA project subsets used throughout our experiments.</p>
</div>

<p>Across all three baseline approaches below, we follow a consistent engineering workflow: (1) load libraries and set random seeds, (2) build the dataset, image transforms, and answer vocabulary, (3) construct the model architecture, (4) define training and evaluation routines, (5) run optimization, and (6) evaluate on held-out validation and test splits.</p>

<h2 id="mlp-approach">2. Approach 1: Multi-Layer Perceptron (MLP) Baseline and Metrics</h2>

<p>Before introducing convolutional networks or Transformers, it is instructive to build the simplest possible end-to-end multimodal baseline: resize and flatten the raw RGB image into a 1D pixel vector, encode the question with a bag-of-words <strong>TF-IDF</strong> vector, concatenate both vectors, and classify the answer using a Multi-Layer Perceptron (MLP).</p>

<figure class="article-figure">
  <img src="../assets/images/vqa/7.png" alt="MLP baseline architecture combining resized flattened image pixels with TF-IDF question features into a fusion classifier">
  <figcaption>Figure 8. Overview of the MLP baseline: flattened RGB pixels and TF-IDF question representations concatenated and fed into an MLP classifier.</figcaption>
</figure>

<figure class="article-figure">
  <img src="../assets/images/vqa/8.png" alt="Diagram showing how a 3D RGB image tensor of shape (C, H, W) is unrolled into a single 1D feature vector">
  <figcaption>Figure 9. Flattening a 3D image tensor of shape (C, H, W) into a 1D vector destroys 2D spatial neighborhood structure.</figcaption>
</figure>

<h3 id="mlp-dataset">2.1 Dataset Structure and PyTorch DataLoader</h3>

<p>Each split (<code>train.json</code>, <code>val.json</code>, <code>test.json</code>) is stored as a JSON list of records containing the relative <code>image_path</code>, the <code>question</code> string, the canonical majority <code>answer</code>, and the full list of 10 human annotator <code>answers</code> needed to compute official VQA Accuracy:</p>

```text
[
  {
    "image_path": "images/COCO_train2014_000000262465.jpg",
    "question": "What is on top of the skis?",
    "answer": "snow",
    "answers": [
      "snow", "snow", "snow", "snow", "snow",
      "snow", "snow", "snow", "snow", "snow"
    ]
  }
]
```

<p>A shared PyTorch <code>VQADataset</code> loads each RGB image, applies resizing and normalization transforms, maps the canonical answer string to its integer class ID via an answer vocabulary dictionary (<code>vocab</code>), and preserves the raw annotator list for metric evaluation:</p>

```python
import json
import os
from PIL import Image
from torch.utils.data import Dataset


class VQADataset(Dataset):
    def __init__(self, json_path, data_dir, transform=None, vocab=None):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.data_dir = data_dir
        self.transform = transform
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load RGB image and apply transforms
        img_path = os.path.join(self.data_dir, item["image_path"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        question = item["question"]
        answer = item["answer"]
        answer_idx = self.vocab.get(answer, -1)
        answers = item.get("answers", [answer])

        return {
            "image": image,
            "question": question,
            "answer": answer_idx,
            "raw_answer": answer,
            "answers": answers,
        }
```

<h3 id="mlp-model">2.2 Flattened Pixels + TF-IDF Architecture</h3>

<p>For an input image resized to \(H \times W\) with 3 color channels, flattening produces a vector of size \(3 \cdot H \cdot W\) (for example, \(3 \times 224 \times 224 = 150{,}528\) dimensions). On the text side, a scikit-learn <code>TfidfVectorizer(max_features=1000)</code> fit on training questions transforms each question into a sparse 1,000-dimensional vector. Both vectors are concatenated along <code>dim=1</code>, projected through a bottleneck layer with ReLU and Dropout (0.5), and passed to a two-layer classification head:</p>

```python
import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    def __init__(self, num_classes, img_size, tfidf_dim, tfidf_vectorizer, hidden_dim=512):
        super().__init__()
        self.img_size = img_size
        self.tfidf_dim = tfidf_dim
        self.img_flatten_dim = 3 * img_size * img_size
        self.tfidf_vectorizer = tfidf_vectorizer

        fusion_dim = self.img_flatten_dim + tfidf_dim
        self.bottleneck = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def encode_questions(self, questions):
        tfidf_features = self.tfidf_vectorizer.transform(questions).toarray()
        return torch.FloatTensor(tfidf_features)

    def forward(self, images, questions):
        batch_size = images.size(0)
        img_features = images.view(batch_size, -1)
        text_features = self.encode_questions(questions).to(images.device)
        combined = torch.cat([img_features, text_features], dim=1)
        fused_features = self.bottleneck(combined)
        output = self.classifier(fused_features)
        return output
```

<h3 id="vqa-metrics">2.3 Evaluation Metrics: VQA Accuracy and ANLS</h3>

<p>Natural language answers often have legitimate surface variations or annotator disagreement. Accordingly, VQA research relies on two specialized evaluation metrics tailored to our two datasets:</p>

<p><strong>1. VQA Accuracy (used for VQAv2):</strong> Each VQAv2 question is annotated independently by 10 human annotators. A predicted answer \(\hat{a}\) receives full credit (1.0) if at least 3 of the 10 human responses match it identically, and proportional partial credit otherwise:</p>

<div class="equation-block" role="math" aria-label="VQA Accuracy formula">
  \[
    \text{VQA Accuracy}(\hat{a}) = \min\!\left(\frac{\text{number of matching human answers}}{3},\; 1\right)
  \]
</div>

```python
def compute_vqa_accuracy(prediction, ground_truth_answers):
    if not ground_truth_answers:
        return 0.0
    count = 0
    pred = prediction.lower().strip()
    for ans in ground_truth_answers:
        if ans.lower().strip() == pred:
            count += 1
    return min(1.0, count / 3.0)
```

<p><strong>2. Average Normalized Levenshtein Similarity, ANLS (used for ViInfographicVQA):</strong> Infographic answers are often multi-word phrases, numbers, or OCR strings where minor character typos should not incur a complete zero penalty as long as the predicted string is close to the target. Normalized Levenshtein distance divides the character edit distance by the longer string length, and ANLS applies a cutoff threshold \(\tau = 0.5\):</p>

<div class="equation-block" role="math" aria-label="Normalized Levenshtein and ANLS formulas">
  \[
    \mathrm{NL}(\hat{a}, a) = \frac{\mathrm{Levenshtein}(\hat{a}, a)}{\max(\mathrm{len}(\hat{a}),\, \mathrm{len}(a))},
    \qquad
    \mathrm{ANLS}_i =
    \begin{cases}
      1 - \mathrm{NL}(\hat{a}_i, a_i), & \text{if } 1 - \mathrm{NL}(\hat{a}_i, a_i) \ge 0.5 \\
      0, & \text{otherwise}
    \end{cases}
  \]
</div>

<div class="equation-block" role="math" aria-label="Average ANLS across dataset">
  \[
    \text{Avg ANLS} = \frac{1}{N} \sum_{i=1}^{N} \mathrm{ANLS}_i
  \]
</div>

```python
def compute_anls(target, prediction):
    target = target.lower().strip()
    prediction = prediction.lower().strip()

    if not target or not prediction:
        return 0.0 if target != prediction else 1.0

    dist = levenshtein_distance(target, prediction)
    max_len = max(len(target), len(prediction))

    score = 1.0 - float(dist) / float(max_len)
    return score if score >= 0.5 else 0.0
```

<p>When trained on our VQAv2 subset for 10 epochs, the MLP baseline collapses near random guessing: only <strong>3.27% Train</strong>, <strong>2.93% Val</strong>, and <strong>1.27% Test VQA Accuracy</strong> (7.40% Top-5, 0.0144 ANLS). Flattening raw RGB pixels throws away translation invariance and local visual patterns, while TF-IDF ignores word order and syntactic roles (for instance, treating <em>"man bites dog"</em> identically to <em>"dog bites man"</em>).</p>

<h2 id="cnn-lstm-approach">3. Approach 2: CNN + BiLSTM Baseline</h2>

<h3 id="cnn-lstm-architecture">3.1 Pretrained ResNet50 and Bidirectional LSTM</h3>

<p>To repair both representation bottlenecks, our second architecture replaces each naive encoder with a modality-appropriate deep backbone:</p>

<ul>
  <li><strong>Visual Encoder (Frozen Pretrained ResNet50):</strong> Using <code>timm.create_model('resnet50', pretrained=True, num_classes=0)</code> extracts a compact 2,048-dimensional semantic feature vector capturing objects, textures, and spatial hierarchy. Keeping the CNN weights frozen speeds up training and prevents overfitting on a 7,000-QA dataset.</li>
  <li><strong>Question Encoder (Embedding + 2-Layer Bi-LSTM):</strong> Questions are tokenized against a word vocabulary built from the training split (with <code>&lt;PAD&gt;: 0</code> and <code>&lt;UNK&gt;: 1</code>), padded or truncated to <code>max_len=64</code>, embedded into 300-dimensional vectors, and passed through a 2-layer bidirectional LSTM (<code>hidden_dim=512</code> per direction).</li>
</ul>

<figure class="article-figure">
  <img src="../assets/images/vqa/9.png" alt="CNN + BiLSTM VQA architecture combining ResNet50 image features with concatenated bidirectional LSTM hidden states">
  <figcaption>Figure 10. CNN + BiLSTM architecture: ResNet50 extracts semantic visual features while a Bidirectional LSTM encodes sequential question context.</figcaption>
</figure>

<p>Because a bidirectional LSTM processes the question both left-to-right and right-to-left, concatenating the final forward hidden state \(\vec{h}_T\) (<code>hidden[-2]</code>) and final backward hidden state \(\overleftarrow{h}_1\) (<code>hidden[-1]</code>) yields a 1,024-dimensional language vector summarizing full-sentence context:</p>

<figure class="article-figure">
  <img src="../assets/images/vqa/10.png" alt="Unrolled Bidirectional LSTM showing forward hidden states h_1 to h_6 and backward hidden states h'_6 to h'_1 concatenated into a Language Vector Feature">
  <figcaption>Figure 11. Concatenating the final forward and backward hidden states of a Bi-LSTM produces a unified question representation.</figcaption>
</figure>

```python
import timm
import torch
import torch.nn as nn


class CNNLSTMBaseline(nn.Module):
    def __init__(
        self,
        num_classes,
        vocab_size,
        word_to_idx,
        hidden_dim=512,
        lstm_layers=2,
        embed_dim=300,
        max_len=64,
    ):
        super().__init__()

        # Frozen pretrained ResNet50 visual backbone
        self.cnn = timm.create_model("resnet50", pretrained=True, num_classes=0)
        for param in self.cnn.parameters():
            param.requires_grad = False
        cnn_output_dim = self.cnn.num_features

        # Trainable token embeddings and 2-layer Bi-LSTM question encoder
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        lstm_output_dim = hidden_dim * 2

        fusion_dim = cnn_output_dim + lstm_output_dim
        self.bottleneck = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.word_to_idx = word_to_idx
        self.max_len = max_len

    def tokenize(self, questions):
        batch_tokens = []
        for q in questions:
            words = q.lower().split()
            tokens = [self.word_to_idx.get(w, 1) for w in words]
            if len(tokens) < self.max_len:
                tokens = tokens + [0] * (self.max_len - len(tokens))
            else:
                tokens = tokens[: self.max_len]
            batch_tokens.append(tokens)
        return torch.LongTensor(batch_tokens)

    def forward(self, images, questions):
        img_features = self.cnn(images)
        question_tokens = self.tokenize(questions).to(images.device)
        question_embeds = self.embedding(question_tokens)

        _, (hidden, _) = self.lstm(question_embeds)
        text_features = torch.cat([hidden[-2], hidden[-1]], dim=1)

        combined = torch.cat([img_features, text_features], dim=1)
        fused_features = self.bottleneck(combined)
        output = self.classifier(fused_features)
        return output
```

<h3 id="cnn-lstm-results">3.2 MLP vs. CNN + BiLSTM Results</h3>

<p>Switching from raw pixels + TF-IDF to pretrained ResNet50 + Bi-LSTM produces a dramatic leap in generalization across every split: Test VQA Accuracy surges more than 40x from <strong>1.27%</strong> to <strong>53.00%</strong> (and up to 56.23% after tuning batch size and image resolution).</p>

<figure class="article-figure">
  <img src="../assets/images/vqa/11.png" alt="Side-by-side bar charts comparing MLP accuracy (3.27% Train, 2.93% Val, 1.27% Test) with CNN + LSTM accuracy (69.59% Train, 52.23% Val, 53.00% Test)">
  <figcaption>Figure 12. VQA Accuracy comparison between MLP (left) and CNN + BiLSTM (right) across Train, Validation, and Test splits.</figcaption>
</figure>

<h2 id="vit-bert-approach">4. Approach 3: ViT + BERT Baseline</h2>

<h3 id="vit-bert-architecture">4.1 Vision Transformer and BERT / PhoBERT Encoders</h3>

<p>Our third architecture upgrades both modality encoders to pretrained Transformer backbones:</p>

<ul>
  <li><strong>Vision Transformer (ViT):</strong> <code>vit_base_patch16_224</code> splits the image into \(16 \times 16\) patches and applies global self-attention across all patches, capturing long-range visual relationships without convolutional locality constraints.</li>
  <li><strong>Pretrained Language Model (BERT / PhoBERT):</strong> Using HuggingFace <code>AutoModel</code> and <code>AutoTokenizer</code>, we encode English VQAv2 questions with BERT (or RoBERTa) and Vietnamese ViInfographicVQA questions with <code>vinai/phobert-base</code>, extracting the final-layer <code>[CLS]</code> token embedding <code>outputs.last_hidden_state[:, 0, :]</code>.</li>
</ul>

<figure class="article-figure">
  <img src="../assets/images/vqa/12.png" alt="ViT + BERT/PhoBERT architecture concatenating ViT image features with the final [CLS] embedding before the fusion classifier">
  <figcaption>Figure 13. ViT + BERT / PhoBERT baseline fusing global Vision Transformer features with contextual [CLS] question embeddings.</figcaption>
</figure>

```python
import timm
import torch
import torch.nn as nn
from transformers import AutoModel


class ViTBERTBaseline(nn.Module):
    def __init__(
        self,
        num_classes,
        bert_model,
        bert_tokenizer,
        hidden_dim=512,
        img_size=224,
        max_len=64,
        freeze_encoders=True,
    ):
        super().__init__()
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0,
            img_size=img_size,
        )
        vit_output_dim = self.vit.num_features

        self.tokenizer = bert_tokenizer
        self.bert = AutoModel.from_pretrained(bert_model)
        bert_output_dim = self.bert.config.hidden_size

        if freeze_encoders:
            for param in self.vit.parameters():
                param.requires_grad = False
            for param in self.bert.parameters():
                param.requires_grad = False

        fusion_dim = vit_output_dim + bert_output_dim
        self.bottleneck = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.max_len = max_len

    def encode_questions(self, questions, device):
        encoding = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

    def forward(self, images, questions):
        img_features = self.vit(images)
        text_features = self.encode_questions(questions, images.device)
        combined = torch.cat([img_features, text_features], dim=1)
        fused_features = self.bottleneck(combined)
        output = self.classifier(fused_features)
        return output
```

<h3 id="frozen-vs-unfrozen">4.2 Frozen vs. Fine-Tuned Encoders</h3>

<p>An unexpected finding emerges when both ViT and BERT are kept completely frozen (<code>requires_grad = False</code>) and only the linear bottleneck + classifier is trained for 10 epochs at <code>lr = 1e-3</code>: <strong>frozen ViT + BERT underperforms CNN + BiLSTM</strong> (~45.6% Val / ~46.8% Test V-Acc vs. 53.4% Val / 56.2% Test for CNN + BiLSTM) and shows earlier validation loss divergence.</p>

<figure class="article-figure">
  <img src="../assets/images/vqa/13.png" alt="Training and validation loss and accuracy learning curves comparing CNN + LSTM (left) against frozen ViT + BERT (right) over 10 epochs">
  <figcaption>Figure 14. Learning curves over 10 epochs: frozen ViT + BERT (right) overfits its linear classifier faster and plateaus below CNN + BiLSTM (left).</figcaption>
</figure>

<p>Why does this happen? In CNN + BiLSTM, the entire question encoder (<code>nn.Embedding</code> + 2-layer Bi-LSTM) is <em>trainable</em>, so it actively reshapes its language representation to query ResNet50's visual features. By contrast, when both ViT and BERT are frozen, the ImageNet-trained ViT <code>[CLS]</code> token and the masked-language-model BERT <code>[CLS]</code> token live in completely separate representation spaces that were never adapted to answer questions jointly. Once we <strong>unfreeze both ViT and BERT</strong> using differential learning rates (a gentle <code>lr = 1e-5</code> for the pretrained Transformer encoders and <code>lr = 1e-3</code> for the classifier head), ViT + BERT surpasses every baseline by a wide margin.</p>

<h2 id="experiments">5. Experiments and Ablation Studies</h2>

<h3 id="main-results">5.1 Main Benchmark Results on VQAv2 and ViInfographicVQA</h3>

<p>We first benchmark all three frozen-backbone configurations across batch sizes \(\{16, 32\}\) and image resolutions \(\{224, 448\}\) for 10 epochs with learning rate \(10^{-3}\), then compare against our fine-tuned (unfrozen) ViT + BERT configuration.</p>

<div class="article-table-wrap">
  <table>
    <thead>
      <tr>
        <th rowspan="2">Architecture</th>
        <th rowspan="2" class="num">Batch</th>
        <th rowspan="2" class="num">Image</th>
        <th rowspan="2" class="num">LR</th>
        <th colspan="2" class="num">VQAv2 V-Acc (%)</th>
        <th colspan="2" class="num">VQAv2 Top-5 (%)</th>
        <th colspan="2" class="num">ViInfo Acc (%)</th>
        <th colspan="2" class="num">ViInfo ANLS</th>
      </tr>
      <tr>
        <th class="num">Val</th>
        <th class="num">Test</th>
        <th class="num">Val</th>
        <th class="num">Test</th>
        <th class="num">Val</th>
        <th class="num">Test</th>
        <th class="num">Val</th>
        <th class="num">Test</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>MLP</td>
        <td class="num">16</td>
        <td class="num">224</td>
        <td class="num">1e-3</td>
        <td class="num">3.10</td>
        <td class="num">2.20</td>
        <td class="num">10.33</td>
        <td class="num">8.33</td>
        <td class="num">1.90</td>
        <td class="num">0.63</td>
        <td class="num">0.0400</td>
        <td class="num">0.0140</td>
      </tr>
      <tr>
        <td>CNN + LSTM</td>
        <td class="num">16</td>
        <td class="num">224</td>
        <td class="num">1e-3</td>
        <td class="num">52.77</td>
        <td class="num">54.70</td>
        <td class="num">83.57</td>
        <td class="num">82.80</td>
        <td class="num">6.40</td>
        <td class="num">3.17</td>
        <td class="num">0.0849</td>
        <td class="num">0.0893</td>
      </tr>
      <tr>
        <td>ViT + BERT (frozen)</td>
        <td class="num">16</td>
        <td class="num">224</td>
        <td class="num">1e-3</td>
        <td class="num">44.50</td>
        <td class="num">47.87</td>
        <td class="num">73.33</td>
        <td class="num">76.23</td>
        <td class="num">4.10</td>
        <td class="num">1.69</td>
        <td class="num">0.0549</td>
        <td class="num">0.0304</td>
      </tr>
      <tr>
        <td>MLP</td>
        <td class="num">32</td>
        <td class="num">224</td>
        <td class="num">1e-3</td>
        <td class="num">2.97</td>
        <td class="num">1.30</td>
        <td class="num">9.60</td>
        <td class="num">6.70</td>
        <td class="num">1.80</td>
        <td class="num">0.63</td>
        <td class="num">0.0420</td>
        <td class="num">0.0140</td>
      </tr>
      <tr>
        <td>CNN + LSTM</td>
        <td class="num">32</td>
        <td class="num">224</td>
        <td class="num">1e-3</td>
        <td class="num">53.43</td>
        <td class="num">56.23</td>
        <td class="num">82.43</td>
        <td class="num">84.93</td>
        <td class="num">6.80</td>
        <td class="num">4.86</td>
        <td class="num">0.0876</td>
        <td class="num">0.0893</td>
      </tr>
      <tr>
        <td>ViT + BERT (frozen)</td>
        <td class="num">32</td>
        <td class="num">224</td>
        <td class="num">1e-3</td>
        <td class="num">45.57</td>
        <td class="num">46.80</td>
        <td class="num">74.60</td>
        <td class="num">77.33</td>
        <td class="num">4.20</td>
        <td class="num">3.48</td>
        <td class="num">0.0561</td>
        <td class="num">0.0461</td>
      </tr>
      <tr>
        <td>CNN + LSTM</td>
        <td class="num">32</td>
        <td class="num">448</td>
        <td class="num">1e-3</td>
        <td class="num">52.27</td>
        <td class="num">56.07</td>
        <td class="num">82.43</td>
        <td class="num">81.37</td>
        <td class="num">5.90</td>
        <td class="num">4.86</td>
        <td class="num">0.0757</td>
        <td class="num">0.0981</td>
      </tr>
      <tr>
        <td><strong>ViT + BERT (unfrozen)</strong></td>
        <td class="num"><strong>32</strong></td>
        <td class="num"><strong>448</strong></td>
        <td class="num"><strong>1e-5 / 1e-3</strong></td>
        <td class="num"><strong>58.40</strong></td>
        <td class="num"><strong>60.73</strong></td>
        <td class="num"><strong>85.00</strong></td>
        <td class="num"><strong>84.63</strong></td>
        <td class="num"><strong>9.90</strong></td>
        <td class="num"><strong>18.16</strong></td>
        <td class="num"><strong>0.1579</strong></td>
        <td class="num"><strong>0.2601</strong></td>
      </tr>
    </tbody>
  </table>
  <p class="article-table-caption">Table 2. Main benchmark results across VQAv2 and ViInfographicVQA (10 epochs). Unfreezing ViT + BERT with a 1e-5 backbone learning rate unlocks the highest performance on both datasets.</p>
</div>

<p>Two key observations stand out from Table 2. First, unfreezing ViT + BERT improves VQAv2 Test V-Acc by nearly <strong>+14 percentage points</strong> over frozen ViT + BERT (60.73% vs. 46.80%) and more than <strong>triples</strong> ViInfographicVQA Test Accuracy (18.16% vs. 4.86% for CNN + LSTM) and ANLS (0.2601 vs. 0.0981). Second, even our strongest closed-vocabulary classifier reaches only 18.16% on ViInfographicVQA because infographic questions depend heavily on fine-grained OCR text extraction, which classification models without OCR tokens cannot easily read.</p>

<h3 id="ablation-fusion">5.2 Ablation Study 1: Multimodal Fusion Operations</h3>

<p>Given extracted visual features \(v \in \mathbb{R}^{d_v}\) and textual features \(q \in \mathbb{R}^{d_q}\), how should we combine them? We test four fusion operators while holding the rest of each pipeline constant (Batch = 32, Image = 224, 10 epochs, frozen backbones): direct <code>Concat</code> \([v; q]\), projected <code>Linear -> Concat</code> \([W_v v; W_q q]\), element-wise addition <code>Linear -> Plus</code> \((W_v v + W_q q)\), and element-wise Hadamard multiplication <code>Linear -> Multiply</code> \((W_v v \odot W_q q)\).</p>

<figure class="article-figure">
  <img src="../assets/images/vqa/14.png" alt="Diagram highlighting the Features Fusion stage comparing Addition, Multiplication, and Concatenation of visual and textual features">
  <figcaption>Figure 15. Ablation Study 1 isolates the Features Fusion operator: Concatenation, Addition, and Element-wise Multiplication.</figcaption>
</figure>

<div class="article-table-wrap">
  <table>
    <thead>
      <tr>
        <th rowspan="2">Architecture</th>
        <th rowspan="2">Fusion Operation</th>
        <th colspan="2" class="num">VQAv2 V-Acc (%)</th>
        <th colspan="2" class="num">VQAv2 Top-5 (%)</th>
        <th colspan="2" class="num">ViInfo Acc (%)</th>
        <th colspan="2" class="num">ViInfo ANLS</th>
      </tr>
      <tr>
        <th class="num">Val</th>
        <th class="num">Test</th>
        <th class="num">Val</th>
        <th class="num">Test</th>
        <th class="num">Val</th>
        <th class="num">Test</th>
        <th class="num">Val</th>
        <th class="num">Test</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>MLP</td>
        <td>Concat</td>
        <td class="num">2.97</td>
        <td class="num">1.30</td>
        <td class="num">9.60</td>
        <td class="num">6.70</td>
        <td class="num">1.80</td>
        <td class="num">0.63</td>
        <td class="num">0.0420</td>
        <td class="num">0.0140</td>
      </tr>
      <tr>
        <td>MLP</td>
        <td>Linear -> Concat</td>
        <td class="num">2.73</td>
        <td class="num">1.63</td>
        <td class="num">10.43</td>
        <td class="num">7.63</td>
        <td class="num">1.60</td>
        <td class="num">0.53</td>
        <td class="num">0.0365</td>
        <td class="num">0.0259</td>
      </tr>
      <tr>
        <td>MLP</td>
        <td>Linear -> Plus</td>
        <td class="num">3.17</td>
        <td class="num">1.93</td>
        <td class="num">10.50</td>
        <td class="num">8.57</td>
        <td class="num">2.20</td>
        <td class="num">0.74</td>
        <td class="num">0.0423</td>
        <td class="num">0.0265</td>
      </tr>
      <tr>
        <td><strong>MLP</strong></td>
        <td><strong>Linear -> Multiply</strong></td>
        <td class="num"><strong>17.87</strong></td>
        <td class="num"><strong>17.27</strong></td>
        <td class="num"><strong>42.73</strong></td>
        <td class="num"><strong>45.17</strong></td>
        <td class="num"><strong>3.30</strong></td>
        <td class="num"><strong>2.11</strong></td>
        <td class="num"><strong>0.0558</strong></td>
        <td class="num"><strong>0.0480</strong></td>
      </tr>
      <tr>
        <td><strong>CNN + LSTM</strong></td>
        <td><strong>Concat</strong></td>
        <td class="num"><strong>53.43</strong></td>
        <td class="num"><strong>56.23</strong></td>
        <td class="num"><strong>82.43</strong></td>
        <td class="num"><strong>84.93</strong></td>
        <td class="num"><strong>6.80</strong></td>
        <td class="num">4.86</td>
        <td class="num"><strong>0.0876</strong></td>
        <td class="num">0.0893</td>
      </tr>
      <tr>
        <td>CNN + LSTM</td>
        <td>Linear -> Concat</td>
        <td class="num">51.50</td>
        <td class="num">54.73</td>
        <td class="num">81.63</td>
        <td class="num">82.63</td>
        <td class="num">6.70</td>
        <td class="num"><strong>5.70</strong></td>
        <td class="num">0.0837</td>
        <td class="num">0.0956</td>
      </tr>
      <tr>
        <td>CNN + LSTM</td>
        <td>Linear -> Plus</td>
        <td class="num">50.63</td>
        <td class="num">53.43</td>
        <td class="num">81.20</td>
        <td class="num">81.63</td>
        <td class="num">5.40</td>
        <td class="num">4.22</td>
        <td class="num">0.0825</td>
        <td class="num">0.0908</td>
      </tr>
      <tr>
        <td>CNN + LSTM</td>
        <td>Linear -> Multiply</td>
        <td class="num">49.30</td>
        <td class="num">50.10</td>
        <td class="num">79.43</td>
        <td class="num">80.60</td>
        <td class="num">5.20</td>
        <td class="num">5.28</td>
        <td class="num">0.0725</td>
        <td class="num"><strong>0.1108</strong></td>
      </tr>
      <tr>
        <td>ViT + BERT (frozen)</td>
        <td>Concat</td>
        <td class="num">45.57</td>
        <td class="num">46.80</td>
        <td class="num">74.60</td>
        <td class="num">77.33</td>
        <td class="num">4.20</td>
        <td class="num">3.48</td>
        <td class="num">0.0561</td>
        <td class="num">0.0461</td>
      </tr>
      <tr>
        <td><strong>ViT + BERT (frozen)</strong></td>
        <td><strong>Linear -> Concat</strong></td>
        <td class="num"><strong>48.03</strong></td>
        <td class="num"><strong>49.93</strong></td>
        <td class="num"><strong>77.60</strong></td>
        <td class="num"><strong>79.53</strong></td>
        <td class="num"><strong>4.70</strong></td>
        <td class="num">2.64</td>
        <td class="num"><strong>0.0620</strong></td>
        <td class="num"><strong>0.0491</strong></td>
      </tr>
      <tr>
        <td>ViT + BERT (frozen)</td>
        <td>Linear -> Plus</td>
        <td class="num">47.17</td>
        <td class="num">47.77</td>
        <td class="num">76.73</td>
        <td class="num">77.67</td>
        <td class="num">4.40</td>
        <td class="num">2.96</td>
        <td class="num">0.0564</td>
        <td class="num">0.0459</td>
      </tr>
      <tr>
        <td>ViT + BERT (frozen)</td>
        <td>Linear -> Multiply</td>
        <td class="num">47.00</td>
        <td class="num">48.63</td>
        <td class="num">75.93</td>
        <td class="num">78.47</td>
        <td class="num">4.10</td>
        <td class="num"><strong>3.59</strong></td>
        <td class="num">0.0574</td>
        <td class="num">0.0446</td>
      </tr>
    </tbody>
  </table>
  <p class="article-table-caption">Table 3. Ablation on multimodal fusion operations (Batch = 32, Image = 224, 10 epochs, frozen backbones).</p>
</div>

<p>Look at the remarkable jump for MLP under <code>Linear -> Multiply</code>: VQAv2 Test Accuracy shoots from <strong>1.30% to 17.27%</strong> (and Top-5 from 6.70% to 45.17%)! Because raw pixel concatenation is dominated in dimension by 150,528 pixel values versus 1,000 TF-IDF features, projecting both modalities to a shared dimension and multiplying them forces direct multiplicative gating between image and question signals. Meanwhile, for frozen ViT + BERT, adding linear alignment projections (<code>Linear -> Concat</code>) lifts VQAv2 Test Accuracy from 46.80% to 49.93% by bridging the mismatch between the two frozen feature spaces.</p>

<h3 id="ablation-post-fusion">5.3 Ablation Study 2: Post-Fusion Networks</h3>

<p>Next, we fix concatenation as the fusion step and vary the Answer Generator architecture placed after fusion: (1) a standard <code>MLP (None)</code>, (2) <code>LSTM + MLP</code>, and (3) <code>Attention + MLP</code>.</p>

<figure class="article-figure">
  <img src="../assets/images/vqa/15.png" alt="Diagram isolating the Answer Generator stage comparing MLP (None), LSTM + MLP, and Attention + MLP post-fusion networks">
  <figcaption>Figure 16. Ablation Study 2 compares three post-fusion Answer Generator designs on top of concatenated multimodal features.</figcaption>
</figure>

<div class="article-table-wrap">
  <table>
    <thead>
      <tr>
        <th rowspan="2">Architecture</th>
        <th rowspan="2">Post-Fusion</th>
        <th colspan="2" class="num">VQAv2 V-Acc (%)</th>
        <th colspan="2" class="num">VQAv2 Top-5 (%)</th>
        <th colspan="2" class="num">ViInfo Acc (%)</th>
        <th colspan="2" class="num">ViInfo ANLS</th>
      </tr>
      <tr>
        <th class="num">Val</th>
        <th class="num">Test</th>
        <th class="num">Val</th>
        <th class="num">Test</th>
        <th class="num">Val</th>
        <th class="num">Test</th>
        <th class="num">Val</th>
        <th class="num">Test</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>MLP</td>
        <td>None (MLP)</td>
        <td class="num">2.97</td>
        <td class="num">1.30</td>
        <td class="num">9.60</td>
        <td class="num">6.70</td>
        <td class="num">1.80</td>
        <td class="num">0.63</td>
        <td class="num">0.0420</td>
        <td class="num">0.0140</td>
      </tr>
      <tr>
        <td><strong>MLP</strong></td>
        <td><strong>LSTM</strong></td>
        <td class="num"><strong>5.60</strong></td>
        <td class="num"><strong>6.40</strong></td>
        <td class="num"><strong>19.50</strong></td>
        <td class="num"><strong>20.17</strong></td>
        <td class="num"><strong>2.30</strong></td>
        <td class="num"><strong>1.48</strong></td>
        <td class="num"><strong>0.0522</strong></td>
        <td class="num"><strong>0.0425</strong></td>
      </tr>
      <tr>
        <td>MLP</td>
        <td>Attention</td>
        <td class="num">3.17</td>
        <td class="num">1.57</td>
        <td class="num">10.50</td>
        <td class="num">7.70</td>
        <td class="num">2.00</td>
        <td class="num">0.42</td>
        <td class="num">0.0383</td>
        <td class="num">0.0180</td>
      </tr>
      <tr>
        <td><strong>CNN + LSTM</strong></td>
        <td><strong>None (MLP)</strong></td>
        <td class="num"><strong>53.43</strong></td>
        <td class="num"><strong>56.23</strong></td>
        <td class="num"><strong>82.43</strong></td>
        <td class="num"><strong>84.93</strong></td>
        <td class="num">6.80</td>
        <td class="num">4.86</td>
        <td class="num">0.0876</td>
        <td class="num">0.0893</td>
      </tr>
      <tr>
        <td>CNN + LSTM</td>
        <td>LSTM</td>
        <td class="num">50.23</td>
        <td class="num">51.67</td>
        <td class="num">80.83</td>
        <td class="num">82.63</td>
        <td class="num">6.60</td>
        <td class="num">5.39</td>
        <td class="num">0.0862</td>
        <td class="num">0.0953</td>
      </tr>
      <tr>
        <td>CNN + LSTM</td>
        <td>Attention</td>
        <td class="num">52.30</td>
        <td class="num">54.27</td>
        <td class="num">80.43</td>
        <td class="num">82.50</td>
        <td class="num"><strong>7.00</strong></td>
        <td class="num"><strong>6.02</strong></td>
        <td class="num"><strong>0.0906</strong></td>
        <td class="num"><strong>0.1101</strong></td>
      </tr>
      <tr>
        <td>ViT + BERT (frozen)</td>
        <td>None (MLP)</td>
        <td class="num">45.57</td>
        <td class="num">46.80</td>
        <td class="num">74.60</td>
        <td class="num">77.33</td>
        <td class="num">4.20</td>
        <td class="num"><strong>3.48</strong></td>
        <td class="num">0.0561</td>
        <td class="num">0.0461</td>
      </tr>
      <tr>
        <td>ViT + BERT (frozen)</td>
        <td>LSTM</td>
        <td class="num">45.67</td>
        <td class="num">47.10</td>
        <td class="num">75.33</td>
        <td class="num">77.57</td>
        <td class="num"><strong>4.80</strong></td>
        <td class="num">2.96</td>
        <td class="num">0.0604</td>
        <td class="num">0.0465</td>
      </tr>
      <tr>
        <td><strong>ViT + BERT (frozen)</strong></td>
        <td><strong>Attention</strong></td>
        <td class="num"><strong>47.37</strong></td>
        <td class="num"><strong>49.37</strong></td>
        <td class="num"><strong>75.50</strong></td>
        <td class="num"><strong>77.97</strong></td>
        <td class="num">4.10</td>
        <td class="num">2.85</td>
        <td class="num"><strong>0.0628</strong></td>
        <td class="num"><strong>0.0510</strong></td>
      </tr>
    </tbody>
  </table>
  <p class="article-table-caption">Table 4. Ablation on post-fusion Answer Generator networks (Batch = 32, Image = 224, 10 epochs, frozen backbones).</p>
</div>

<p>Adding post-fusion Attention lifts frozen ViT + BERT on VQAv2 (+2.57% Test V-Acc) and pushes CNN + LSTM to its highest ViInfographicVQA scores (6.02% Test Acc and 0.1101 ANLS), while a clean MLP head remains the strongest and least overfit option for CNN + LSTM on VQAv2.</p>

<h2 id="discussion">6. Discussion and Modern Directions</h2>

<p>Our three baselines reveal three fundamental limitations of simple global-feature VQA classifiers: (1) pooling an entire image into one vector loses localized object details, (2) classifiers easily memorize statistical shortcuts in the question text instead of looking at the image, and (3) closed-set answer vocabularies cannot generalize to open-world visual reasoning. Modern VQA research tackles each limitation directly.</p>

<h3 id="butd-attention">6.1 Bottom-Up and Top-Down Attention</h3>

<p>Earlier spatial attention models overlaid a uniform grid (such as \(7 \times 7\) CNN feature cells) onto an image. Rigid grid cells slice across object boundaries and struggle to focus precisely on a specific object like a person's shirt. <strong>Bottom-Up and Top-Down Attention</strong> (Anderson et al., CVPR 2018) fixes this by replacing uniform grids with object-level regions proposed by a Faster R-CNN detector.</p>

<figure class="article-figure">
  <img src="../assets/images/vqa/16.png" alt="Comparison between uniform spatial grid attention on the left and Bottom-Up object bounding box attention on the right for the question What color is his shirt?">
  <figcaption>Figure 17. Uniform spatial grid attention (left) cuts across object boundaries, whereas Bottom-Up attention (right) attends directly over detected objects.</figcaption>
</figure>

<p>In the Bottom-Up and Top-Down architecture:</p>

<ol>
  <li><strong>Bottom-Up Visual Proposals:</strong> Faster R-CNN extracts \(k\) salient object-region features \(V = \{v_1, \dots, v_k\} \in \mathbb{R}^{k \times 2048}\).</li>
  <li><strong>Question Encoding:</strong> A GRU encodes the embedded question into a single vector \(q \in \mathbb{R}^{512}\).</li>
  <li><strong>Top-Down Attention Weights:</strong> Each region feature \(v_i\) is concatenated with \(q\), scored by a nonlinear layer, and normalized via softmax to compute attention weights \(\alpha_i\), yielding an attended visual vector \(\hat{v} = \sum_{i=1}^{k} \alpha_i v_i\).</li>
  <li><strong>Multimodal Fusion:</strong> The attended visual vector \(\hat{v}\) and question vector \(q\) undergo element-wise product fusion before predicting candidate answer scores.</li>
</ol>

<figure class="article-figure">
  <img src="../assets/images/vqa/17.png" alt="Bottom-Up and Top-Down Attention VQA architecture combining Faster R-CNN k x 2048 object features with GRU question-guided top-down attention weights">
  <figcaption>Figure 18. Bottom-Up and Top-Down Attention (Anderson et al., 2018): object-region proposals weighted by question-guided top-down attention.</figcaption>
</figure>

<h3 id="language-bias">6.2 Language Bias and Debiasing (VQA-CP, CSS, RUBi)</h3>

<p>A notorious failure mode in VQA is <strong>language bias</strong>: models learn statistical shortcuts from question phrasing alone and ignore the visual evidence. For example, if 80% of bananas in the training set are yellow, a biased model answers <code>"yellow"</code> to <code>"What color is the banana?"</code> even when shown a green banana. Similarly, if sky photos in training are predominantly blue, the model predicts <code>"blue"</code> on a red sunset sky.</p>

<figure class="article-figure">
  <img src="../assets/images/vqa/18.png" alt="Illustration of language bias where a VQA model answers yellow for a green banana due to training priors versus a debiased model using visual evidence">
  <figcaption>Figure 19. Language bias causes models to rely on majority answer priors ("yellow") instead of inspecting the actual image ("green").</figcaption>
</figure>

<figure class="article-figure">
  <img src="../assets/images/vqa/19.png" alt="Training distribution bias showing mostly blue skies leading to an incorrect blue prediction on a red sunset test image">
  <figcaption>Figure 20. Skewed QA co-occurrences in training splits encourage models to treat frequent answers as the safest bet regardless of the image.</figcaption>
</figure>

<p>To measure and eliminate language shortcuts, the research community introduced both stricter benchmarks and explicit debiasing objectives:</p>

<ul>
  <li><strong>VQA-CP (VQA under Changing Priors):</strong> Re-splits train and test sets so that the answer distribution for each question type shifts substantially between training and evaluation. Models relying on text priors collapse sharply on VQA-CP, exposing whether visual grounding is genuine.</li>
  <li><strong>CSS (Counterfactual Samples Synthesizing):</strong> Automatically constructs counterfactual training pairs by masking critical visual objects (V-CSS) or key question words (Q-CSS) and inverting the ground-truth target (for instance, masking a green tie changes the label to <code>NOT green</code>). Jointly training on original and counterfactual examples forces the network to depend on causal visual and textual evidence.</li>
  <li><strong>RUBi (Reducing Unimodal Biases):</strong> Adds an auxiliary <em>question-only</em> branch during training that predicts answers purely from text. Its predictions dynamically reweight the main VQA branch via a sigmoid gate, down-weighting examples solvable by text shortcuts alone and forcing gradients onto visually grounded examples. At inference time, the auxiliary branch is removed completely.</li>
</ul>

<figure class="article-figure">
  <img src="../assets/images/vqa/20.png" alt="Distribution bar chart of VQA-CP showing shifted answer priors across question types between train and test splits">
  <figcaption>Figure 21. VQA-CP deliberately shifts per-question answer distributions between train and test splits to evaluate bias-robust visual reasoning.</figcaption>
</figure>

<figure class="article-figure">
  <img src="../assets/images/vqa/21.png" alt="Counterfactual Samples Synthesizing (CSS) diagram illustrating V-CSS region masking and Q-CSS word masking with inverted labels">
  <figcaption>Figure 22. Counterfactual Samples Synthesizing (CSS) masks critical objects or question terms to enforce visually grounded decision boundaries.</figcaption>
</figure>

<figure class="article-figure">
  <img src="../assets/images/vqa/22.png" alt="RUBi architecture with an auxiliary question-only branch gating the main multimodal VQA prediction during training">
  <figcaption>Figure 23. RUBi uses a training-only question-only auxiliary branch to detect language-biased samples and suppress unimodal shortcuts.</figcaption>
</figure>

<h3 id="lvlms-llava">6.3 Large Vision-Language Models and 4-Bit LLaVA Inference</h3>

<p>The modern paradigm shift in VQA moves away from task-specific classifiers entirely by connecting pretrained vision encoders directly to <strong>Large Language Models (LLMs)</strong>, forming <strong>Large Vision-Language Models (LVLMs)</strong>. Because an LLM only accepts token embeddings, an LVLM inserts a lightweight <strong>Projection Layer</strong> (a linear matrix \(W\) or small MLP) that maps visual patch embeddings from a frozen vision encoder (such as CLIP ViT) into the exact word-embedding space of the LLM.</p>

<figure class="article-figure">
  <img src="../assets/images/vqa/23.png" alt="General architecture of a Large Vision-Language Model combining an Image Encoder and Projection layer with Text Embeddings inside an LLM">
  <figcaption>Figure 24. Typical LVLM architecture: visual patch features are projected into visual tokens and concatenated with text tokens inside an LLM.</figcaption>
</figure>

<p><strong>LLaVA (Large Language-and-Vision Assistant, Liu et al., NeurIPS 2023)</strong> popularized this clean, modular design: an input image \(X_v\) passes through a Vision Encoder to produce visual features \(Z_v\), which a trainable projection \(W\) converts into visual tokens \(H_v = W \cdot Z_v\). Concatenated with tokenized language instructions \(H_q\), the combined sequence is processed autoregressively by the language model \(f_\phi\) to generate free-form answers \(X_a\).</p>

<figure class="article-figure">
  <img src="../assets/images/vqa/24.png" alt="LLaVA network architecture showing Vision Encoder features Z_v projected via matrix W into visual tokens H_v alongside language instruction tokens H_q fed into LLM f_phi">
  <figcaption>Figure 25. LLaVA architecture mapping image features into the LLM embedding space via a lightweight projection W.</figcaption>
</figure>

<p>Thanks to HuggingFace <code>transformers</code> and <code>bitsandbytes</code> 4-bit quantization, we can load and run zero-shot inference with <code>llava-hf/llava-1.5-7b-hf</code> directly on a single consumer GPU in just a few lines of Python:</p>

```python
import os
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationConfig,
    LlavaForConditionalGeneration,
)

# 1. Configure 4-bit quantization for memory-efficient GPU inference
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model_id = "llava-hf/llava-1.5-7b-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map=device,
)


# 2. Construct an instruction-following multimodal prompt
def create_prompt(question):
    prompt = f"""### INSTRUCTION:
Your task is to answer the question based on the given image.
### USER: <image>
{question}
### ASSISTANT:"""
    return prompt


# 3. Configure autoregressive decoding parameters
generation_config = GenerationConfig(
    max_new_tokens=10,
    do_sample=True,
    temperature=0.1,
    top_p=0.95,
    top_k=50,
    eos_token_id=model.config.eos_token_id,
    pad_token_id=model.config.pad_token_id,
)

# 4. Run zero-shot VQA inference on a test sample
idx = 0
question = test_data[idx]["question"]
image_name = test_data[idx]["image_path"]
image_path = os.path.join("val2014-resized", image_name)
label = test_data[idx]["answer"]

image = Image.open(image_path).convert("RGB")
prompt = create_prompt(question)
inputs = processor(
    prompt,
    image,
    padding=True,
    return_tensors="pt",
).to(device)

output = model.generate(**inputs, generation_config=generation_config)
generated_text = processor.decode(output[0], skip_special_tokens=True)

print(f"Question: {question}")
print(f"Label: {label}")
print(f"Prediction: {generated_text.split('### ASSISTANT:')[-1].strip()}")
```

<p>Qualitative comparison highlights the practical contrast between compact specialist models and zero-shot LVLMs: a fine-tuned <strong>ViT + RoBERTa</strong> classifier achieves strong closed-vocabulary VQA accuracy at a fraction of the compute cost, whereas a pretrained <strong>LLaVA-7B</strong> model handles diverse, unseen questions and nuanced visual scenes out of the box without any task-specific retraining.</p>

<figure class="article-figure">
  <img src="../assets/images/vqa/25.png" alt="Qualitative comparison showing predictions from a trained ViT+RoBERTa classifier on a snowy mountain scene and a zero-shot pretrained LLaVA-7B model on an indoor bedroom scene">
  <figcaption>Figure 26. Qualitative comparison between a task-trained ViT + RoBERTa classifier (left) and zero-shot pretrained LLaVA-7B (right).</figcaption>
</figure>

<h2 id="summary">7. Summary and References</h2>

<p>Walking through VQA from first principles reveals a clear progression in multimodal representation learning:</p>

<ul>
  <li><strong>Representation quality dominates:</strong> Flattening RGB pixels and TF-IDF bags of words fails completely (~1.3% Test V-Acc), whereas pretrained ResNet50 + trainable Bi-LSTM immediately reaches 56.23% V-Acc (84.93% Top-5) on VQAv2.</li>
  <li><strong>Cross-modal alignment matters for Transformers:</strong> Freezing both ViT and BERT underperforms CNN + BiLSTM (~46.8% Test V-Acc) because two independently pretrained <code>[CLS]</code> spaces are poorly aligned; unfreezing both encoders with differential learning rates (<code>1e-5</code> for encoders, <code>1e-3</code> for the classifier) yields our best results across both benchmarks (<strong>60.73% V-Acc</strong> on VQAv2 and <strong>18.16% Acc / 0.2601 ANLS</strong> on ViInfographicVQA).</li>
  <li><strong>Fusion geometry can unlock weak features:</strong> Element-wise multiplication (<code>Linear -> Multiply</code>) boosts MLP accuracy more than 13x (from 1.30% to 17.27%) by introducing direct multiplicative gating across modalities.</li>
  <li><strong>Modern VQA moves beyond global classification:</strong> Object-level attention (Bottom-Up and Top-Down), counterfactual and auxiliary-branch debiasing (VQA-CP, CSS, RUBi), and generative LVLMs (LLaVA) overcome the spatial, statistical, and vocabulary ceilings of classical classifiers.</li>
</ul>

<h3>References</h3>

<ol>
  <li>Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., &amp; Parikh, D. (2017). <em>Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering (VQAv2).</em> CVPR 2017.</li>
  <li>Anderson, P., He, X., Buehler, C., Teney, D., Johnson, M., Gould, S., &amp; Zhang, L. (2018). <em>Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering.</em> CVPR 2018.</li>
  <li>Agrawal, A., Batra, D., Parikh, D., &amp; Kembhavi, A. (2018). <em>Don't Just Assume; Look and Answer: Overcoming Priors for Visual Question Answering (VQA-CP).</em> CVPR 2018.</li>
  <li>Cadene, R., Dancette, C., Ben-Younes, H., Cord, M., &amp; Parikh, D. (2019). <em>RUBi: Reducing Unimodal Biases for Visual Question Answering.</em> NeurIPS 2019.</li>
  <li>Chen, L., Yan, X., Xiao, J., Zhang, H., Pu, S., &amp; Zhuang, Y. (2020). <em>Counterfactual Samples Synthesizing for Robust Visual Question Answering (CSS).</em> CVPR 2020.</li>
  <li>Liu, H., Li, C., Wu, Q., &amp; Lee, Y. J. (2023). <em>Visual Instruction Tuning (LLaVA).</em> NeurIPS 2023.</li>
</ol>
