---
layout: post
title: "The Story of Embedding — Deep Dive: From Bag of Words to Sentence Transformers"
date: 2026-02-20 09:00:00 +0530
categories: [AI, LLM, Embedding]
tags: [Embedding, Word2Vec, Sentence Transformers]
author: Girijesh Prasad
excerpt: "The mathematical intuitions, architectural decisions, and production lessons behind 70 years of teaching machines to understand language — from Bag of Words to Sentence Transformers."
---

# The Story of Embedding — Deep Dive: From Bag of Words to Sentence Transformers

*The mathematical intuitions, architectural decisions, and production lessons behind 70 years of teaching machines to understand language.*

---

## Why This Post Exists

There are hundreds of "intro to embeddings" posts out there. Most of them tell you *what* Word2Vec and BERT are. Very few explain *why* each generation of embeddings emerged, what mathematical insight drove each breakthrough, and what actually matters when you're deploying these systems in production.

This post is for engineers who want to go deeper — who want to understand not just the "what" but the "why" and the "how it actually works under the hood."

Let's trace the full arc, starting from first principles.

---

## 1. The Representation Problem: Why Vectors?

Before we count a single word, we need to answer a fundamental question: **why represent text as vectors at all?**

The answer is deceptively simple: **vectors give us geometry**, and geometry gives us the ability to *measure*. Once you have text as vectors, you can compute distances (how different are two documents?), find nearest neighbours (what's the most similar sentence?), and perform operations (what's halfway between "happy" and "sad"?).

The entire history of embeddings is really the history of making these geometric operations *meaningful* — making the geometry of the vector space mirror the semantics of language.

<pre class="mermaid">
timeline
    title The Evolution of Text Embeddings
    section Count-Based
        1950s : Bag of Words
            : Simple frequency counting
        1990 : LSA (SVD)
            : Latent semantic structure
        1992 : TF-IDF
            : Information-theoretic weighting
    section Neural Static
        2003 : Bengio NPLM
            : First neural word embeddings
        2013 : Word2Vec
            : Negative sampling breakthrough
        2014 : GloVe
            : Global co-occurrence factorisation
        2016 : FastText
            : Subword n-grams
    section Contextual
        2017 : Transformer
            : Self-attention architecture
        2018 : ELMo
            : Layer-wise contextual representations
        2018 : BERT
            : Pretraining-finetuning paradigm
    section Sentence-Level
        2019 : Sentence-BERT
            : Siamese bi-encoders
        2020 : ColBERT
            : Late interaction
        2022 : Matryoshka
            : Adaptive dimensionality
        2023-24 : E5 / BGE / NV-Embed
            : Instruction-tuned embeddings
</pre>

---

## 2. The Counting Era: BoW, TF-IDF, and Their Hidden Mathematics

### Bag of Words (1950s)

BoW maps each document to an |V|-dimensional vector, where |V| is the vocabulary size. Simple frequency counting. But here's what most tutorials skip: **BoW is actually performing a projection from the infinite-dimensional space of possible utterances onto a finite vector space** — and it's a lossy projection that discards word order, syntax, and semantics.

The fundamental limitation isn't just "no semantics." It's the **curse of dimensionality for sparse vectors**. With |V| = 100,000, every document lives in a 100,000-dimensional space where cosine similarity becomes almost meaningless — in high-dimensional sparse spaces, all pairwise distances converge, a phenomenon known as the **concentration of measure**.

### TF-IDF: Information-Theoretic Weighting

TF-IDF is more interesting than most people realise. The IDF component:

$$
\text{IDF}(t) = \log\frac{N}{df(t)}
$$

is essentially an **information-theoretic** quantity. A word that appears in every document (df(t) = N) has IDF = 0 — zero information value. A rare word has high IDF. This connects directly to Shannon's self-information: rare events carry more information.

But TF-IDF still builds on the **independence assumption** — it treats each word as statistically independent of every other word. "New York" is just "New" + "York". This is where the paradigm needed to break.

### LSA: The Forgotten Bridge (1990)

Most "embedding history" posts jump from TF-IDF to Word2Vec, skipping the critical intermediate step: **Latent Semantic Analysis (LSA)** by Deerwester et al. (1990).

LSA takes the term-document matrix and applies **Singular Value Decomposition (SVD)**:

$$
X \approx U_k \Sigma_k V_k^T
$$

By keeping only the top-k singular values, you project documents into a k-dimensional space (typically k=100-300) where **synonyms collapse together** and **polysemy partially resolves**. LSA was the first demonstration that **dimensionality reduction on co-occurrence data captures latent semantic structure**.

This insight — that meaning hides in the statistical structure of co-occurrence — is the intellectual ancestor of everything that follows.

---

## 3. The Neural Turn: Bengio's NPLM (2003) — The Forgotten Origin

The standard narrative says Word2Vec (2013) started neural embeddings. **That's wrong.** The actual origin is Yoshua Bengio's **Neural Probabilistic Language Model (NPLM)**, published in 2003 — a full decade earlier.

Bengio's key insight: assign each word a **learned distributed representation** (a dense vector), then train a neural network to predict the next word from the concatenation of the previous n words' vectors.

The model had three components:

1. **Embedding lookup table** C: a |V| × d matrix mapping word indices to d-dimensional vectors
2. **Hidden layer**: `h = tanh(H · [C(w_{t-n+1}); ...; C(w_{t-1})] + b)`
3. **Output softmax**: probability distribution over all |V| words

The genius was that the **embedding table C was learned jointly** with the prediction task. Words that could appear in similar contexts would naturally get similar embeddings, because similar embeddings would produce similar predictions through the hidden layer.

<pre class="mermaid">
graph LR
    subgraph Input["Input: Previous n words"]
        W1["w(t-3)"] --> E1["Embedding C(w(t-3))"]
        W2["w(t-2)"] --> E2["Embedding C(w(t-2))"]
        W3["w(t-1)"] --> E3["Embedding C(w(t-1))"]
    end
    E1 --> CONCAT["Concatenate"]
    E2 --> CONCAT
    E3 --> CONCAT
    CONCAT --> HIDDEN["Hidden Layer - tanh Hx + b"]
    HIDDEN --> SOFTMAX["Softmax over V words - O(V) bottleneck"]
    SOFTMAX --> PRED["P w_t = next word"]
    style SOFTMAX fill:#ff6b6b,stroke:#333,color:#fff
    style PRED fill:#51cf66,stroke:#333,color:#fff
</pre>

**Why did it take 10 years to become mainstream?** Bengio's model was computationally expensive. The softmax output layer required computing a |V|-way classification *for every position in the training data*. With V = 100K words and billions of training positions, this was intractable in 2003.

Word2Vec's real contribution wasn't the idea of neural embeddings — it was making them **computationally feasible**.

---

## 4. Word2Vec (2013): The Trick Was in the Training

### The Skip-gram Objective

Skip-gram's true objective function maximises:

$$
J = \frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} | w_t)
$$

where T is the total words in the corpus, c is the context window size, and:

$$
P(w_O | w_I) = \frac{\exp(\tilde{v}_{w_O}^{\,T} \cdot v_{w_I})}{\sum_{w=1}^{V} \exp(\tilde{v}_w^{\,T} \cdot v_{w_I})}
$$

The denominator is a **sum over the entire vocabulary** — this is the bottleneck that killed Bengio's model. With V = 100K+, computing this for every training example is absurdly expensive.

### Negative Sampling: The Actual Innovation

Mikolov's key contribution was **negative sampling**, which replaces the expensive softmax with a much cheaper binary classification:

$$
\log \sigma(\tilde{v}_{w_O}^{\,T} \cdot v_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} [\log \sigma(-\tilde{v}_{w_i}^{\,T} \cdot v_{w_I})]
$$

Instead of computing probabilities over all V words, you:

1. Take the actual context word (positive) — push its vector **towards** the target
2. Sample k random "noise" words (negatives, typically k=5-15) — push their vectors **away** from the target

The noise distribution P_n(w) is the unigram distribution raised to the 3/4 power: `P_n(w) = U(w)^{3/4}/Z`. The 3/4 exponent is an empirical choice that slightly upweights rare words relative to their frequency — preventing extremely common words from dominating the negative samples.

This reduced training from O(V) per example to O(k) per example. **That's the real reason Word2Vec succeeded where Bengio's NPLM struggled** — not a fundamentally different idea, but a training trick that made it 10,000x faster.

<pre class="mermaid">
graph TD
    subgraph FULL["Full Softmax (Bengio)"]
        direction LR
        TGT1["Target word"] --> COMP1["Compute score against\nALL V words"]
        COMP1 --> NORM1["Normalise\n(expensive!)"]
        NORM1 --> COST1["O(V) per example\n❌ ~100K operations"]
    end

    subgraph NEG["Negative Sampling (Word2Vec)"]
        direction LR
        TGT2["Target word"] --> POS["✅ 1 positive\n(actual context word)"]
        TGT2 --> NEGS["❌ k=5 negatives\n(random noise words)"]
        POS --> COST2["O(k) per example\n✅ ~5 operations"]
        NEGS --> COST2
    end

    FULL -.->|"replaced by"| NEG

    style COST1 fill:#ff6b6b,stroke:#333,color:#fff
    style COST2 fill:#51cf66,stroke:#333,color:#fff
</pre>

### Why King - Man + Woman ≈ Queen Actually Works

This isn't magic. It's a consequence of the linear structure that skip-gram implicitly learns.

If "king" and "queen" appear in similar royal/monarchical contexts, and "man" and "woman" appear in similar gender-differentiated contexts, then the model learns embeddings where the **gender direction** (man → woman) and the **royalty direction** (commoner → royal) are approximately independent linear subspaces.

Mathematically:

- `v(king) ≈ v(royalty) + v(male)`
- `v(queen) ≈ v(royalty) + v(female)`
- `v(king) - v(man) + v(woman) ≈ v(royalty) + v(male) - v(male) + v(female) ≈ v(royalty) + v(female) ≈ v(queen)`

Levy and Goldberg (2014) proved that **Skip-gram with negative sampling is implicitly factorising a shifted PMI matrix** — the pointwise mutual information between words and contexts, shifted by log(k). This connects Word2Vec back to the distributional semantics tradition and explains *why* the embeddings capture semantic relationships: PMI is a well-understood measure of statistical association.

---

## 5. GloVe: Making the Implicit Explicit

### The Objective Function

Pennington et al. at Stanford asked: if Word2Vec is implicitly factorising a co-occurrence matrix, why not do it **explicitly**?

GloVe's objective:

$$
J = \sum_{i,j=1}^{V} f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2
$$

where X_ij is the co-occurrence count of words i and j, and f(x) is a weighting function:

$$
f(x) = \begin{cases} (x/x_{max})^{0.75} & \text{if } x < x_{max} \\ 1 & \text{otherwise} \end{cases}
$$

The weighting function f() is crucial: it prevents extremely frequent co-occurrences (like "the" + anything) from dominating the objective, whilst giving zero weight to word pairs that never co-occur (X_ij = 0).

**Key insight:** The model asks that the **dot product of two word vectors** should approximate the **log of their co-occurrence count**. Words that co-occur frequently → high dot product → similar vectors.

### When to Choose GloVe vs Word2Vec

In practice, the difference is marginal for most downstream tasks (Levy et al., 2015 showed they perform similarly when hyperparameters are properly tuned). The real trade-off is:

- **GloVe**: Single-pass over co-occurrence matrix, deterministic, easier to parallelise
- **Word2Vec**: Online learning (can update with new data), stochastic, works well with streaming data

---

## 6. FastText: Morphology Matters

FastText's innovation isn't just "handles OOV words." The deeper insight is about **morphological compositionality**.

The word vector is the sum of its character n-gram vectors:

$$
v_{w} = \sum_{g \in \mathcal{G}(w)} z_g
$$

where G(w) is the set of n-grams (n=3-6 typically) for word w, plus the word itself.

This means:

- "unhappy" ≈ "un" + "happy" → the "un-" prefix carries negation information
- "running", "runner", "ran" share subword features
- Misspelled "embeddding" shares most n-grams with "embedding"

**Why this matters for production:** In real-world data, you encounter typos, domain-specific neologisms, code-mixed text (Hindi + English), and morphologically rich languages. FastText handles all of these gracefully, whilst Word2Vec and GloVe would return a zero/random vector.

---

## 7. ELMo: The Layer-Wise Revelation

### Architecture

ELMo (Peters et al., 2018) uses a **2-layer bidirectional LSTM** trained as a language model. The critical insight wasn't just "context-dependent vectors" — it was what each layer captures.

The ELMo representation for a token k is:

$$
\text{ELMo}_k^{task} = \gamma^{task} \sum_{j=0}^{L} s_j^{task} h_{k,j}
$$

where:

- h_{k,0} = character-level CNN (subword features)
- h_{k,1} = first LSTM layer (syntactic features)
- h_{k,2} = second LSTM layer (semantic features)
- s_j = softmax-normalised weights (learned per task)
- γ = task-specific scaling factor

**The revelation:** Peters et al. showed that **different layers encode different linguistic properties**. Lower layers capture syntax (POS tags, syntactic dependencies), higher layers capture semantics (word sense, sentiment). This was the first hard evidence for **hierarchical language representation** in neural networks — an insight that would prove fundamental for understanding Transformers.

<pre class="mermaid">
graph BT
    INPUT["Raw Text: I went to the bank"] --> CHAR["Layer 0: Character CNN - Subword features, morphology"]
    CHAR --> L1["Layer 1: Bidirectional LSTM - Syntax: POS tags, dependencies"]
    L1 --> L2["Layer 2: Bidirectional LSTM - Semantics: word sense, sentiment"]
    L2 --> COMBINE["Task-Specific Weighted Sum"]
    CHAR --> COMBINE
    L1 --> COMBINE
    COMBINE --> TASK["Downstream Task"]
    style CHAR fill:#74c0fc,stroke:#333
    style L1 fill:#748ffc,stroke:#333,color:#fff
    style L2 fill:#9775fa,stroke:#333,color:#fff
    style COMBINE fill:#ffd43b,stroke:#333
</pre>

### The Feature-Based vs Fine-Tuning Distinction

ELMo was used as a **feature extractor** — you'd freeze ELMo and concatenate its outputs with your task-specific model's inputs. This is different from BERT's approach of fine-tuning the entire model. The debate between feature-based and fine-tuning approaches continues even today (prefix tuning, adapters, LoRA all revisit this tension).

---

## 8. Attention Is All You Need (2017): The Foundation

Before BERT, we need to understand the **Transformer** (Vaswani et al., 2017), because it's the architectural foundation for everything that follows.

### Self-Attention: The Core Mechanism

The attention function:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Three things to understand here:

**1. Why Q, K, V?** These come from information retrieval. Query (what am I looking for?), Key (what does each position offer?), Value (what information does each position contain?). Each word generates all three by multiplying with learned weight matrices: Q = XW_Q, K = XW_K, V = XW_V.

**2. Why scale by √d_k?** Without scaling, when d_k is large, the dot products QK^T can become very large in magnitude, pushing the softmax into regions where it has **extremely small gradients** (saturation). Scaling by √d_k keeps the variance of the dot products at ~1 regardless of dimensionality. This is subtle but critical for training stability.

**3. Why multi-head?** Instead of a single attention function with d_model dimensions, use h attention heads, each with d_k = d_model/h dimensions:

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

Each head can attend to different aspects of the input (one head for syntactic relations, another for semantic similarity, another for coreference, etc.). This is not just a performance trick — it enables **different representational subspaces**.

<pre class="mermaid">
graph LR
    subgraph Input
        X["Input Embeddings + Positional Encoding"]
    end
    X --> WQ["W_Q"] --> Q["Queries"]
    X --> WK["W_K"] --> K["Keys"]
    X --> WV["W_V"] --> V["Values"]
    Q --> DOT["QK_T / sqrt d_k"]
    K --> DOT
    DOT --> SM["Softmax attention weights"]
    SM --> MUL["Multiply with V"]
    V --> MUL
    MUL --> H1["Head 1 - syntax"]
    MUL --> H2["Head 2 - semantics"]
    MUL --> H3["Head 3 - coreference"]
    MUL --> Hn["Head h - ..."]
    H1 --> CAT["Concat"]
    H2 --> CAT
    H3 --> CAT
    Hn --> CAT
    CAT --> WO["W_O"] --> OUT["Output"]
    style DOT fill:#ffd43b,stroke:#333
    style SM fill:#ff922b,stroke:#333,color:#fff
    style OUT fill:#51cf66,stroke:#333,color:#fff
</pre>

### Positional Encoding: The Unsung Hero

Attention is **permutation-invariant** — it doesn't know word order. The positional encoding adds order information using sinusoidal functions:

$$
PE_{(pos,2i)} = \sin(pos/10000^{2i/d_{model}})
$$

$$
PE_{(pos,2i+1)} = \cos(pos/10000^{2i/d_{model}})
$$

Why sinusoids? Because `PE(pos+k)` can be expressed as a linear function of `PE(pos)`, meaning the model can learn to attend to **relative positions** — "the word 3 positions back" — rather than absolute positions. Later models (RoPE, ALiBi) improved on this, but the intuition remains.

---

## 9. BERT (2018): The Paradigm Shift

### What Most People Get Wrong About BERT

BERT's contribution is often summarised as "bidirectional Transformers." That's deeply incomplete. BERT's actual innovation was **the pretraining-finetuning paradigm for NLP**:

1. **Pre-train** a massive model on unlabelled text using self-supervised objectives
2. **Fine-tune** the entire model on your specific task with minimal labelled data

This was revolutionary because **labelled data is expensive**; unlabelled text is effectively infinite.

### The Two Pre-training Objectives

**Masked Language Modelling (MLM):** Randomly mask 15% of input tokens and predict them. But here's the subtlety — of the 15% selected tokens:

- 80% are replaced with [MASK]
- 10% are replaced with a random word
- 10% are kept unchanged

Why this mixed strategy? If all selected tokens were replaced with [MASK], the model would never see [MASK] during fine-tuning, creating a **train-test mismatch**. The random replacement and unchanged tokens mitigate this.

**Next Sentence Prediction (NSP):** Given sentence A, predict whether sentence B is the actual next sentence or a random one. **This objective was later shown to be mostly harmful.** RoBERTa (2019) removed NSP and improved performance, showing that cross-sentence reasoning emerges naturally from MLM alone when trained on longer sequences.

### The [CLS] Token Problem

BERT prepends a special [CLS] token and trains it via NSP to represent the "whole input." Many people use `output[CLS]` as a sentence embedding. **This is a terrible idea for similarity tasks.**

Reimers and Gurevych (2019) showed that using BERT [CLS] embeddings for semantic similarity gives results **worse than GloVe averaged embeddings**. Why? Because BERT's [CLS] was trained for NSP (a binary classification), not for producing meaningful continuous representations of sentence meaning. The embedding space is not isometric — distances don't correspond to semantic similarity.

This fact is critical and widely misunderstood. It's exactly why Sentence-BERT was necessary.

---

## 10. Cross-Encoders vs Bi-Encoders: The Fundamental Trade-off

This is the single most important architectural distinction in modern embeddings, and it's astonishingly under-discussed.

### Cross-Encoder

```
Input: [CLS] Sentence A [SEP] Sentence B [SEP]
      → BERT → Classification Head → Similarity Score
```

Both sentences are processed **together** through the Transformer. Every token in A can attend to every token in B. This gives maximum accuracy because the model can perform fine-grained token-level matching.

**Problem:** You cannot pre-compute embeddings. To compare a query against 1M documents, you must run BERT 1M times with (query, doc_i) as input. For 10K sentences, finding the most similar pair requires C(10000,2) = 49,995,000 forward passes → **~65 hours**.

### Bi-Encoder (Sentence Transformers)

```
Sentence A → BERT → Pool → Embedding_A
Sentence B → BERT → Pool → Embedding_B
→ cosine_similarity(Embedding_A, Embedding_B)
```

Each sentence is processed **independently**. You can pre-compute all embeddings once, then compare using fast vector operations.

**For 10K sentences:** 10,000 forward passes to encode all (seconds), then cosine similarity on 100M pairs is trivial (milliseconds with FAISS).

<pre class="mermaid">
graph TB
    subgraph CE["Cross-Encoder"]
        direction LR
        IN_CE["CLS + Sent A + SEP + Sent B"] --> BERT_CE["BERT full cross-attention"]
        BERT_CE --> CLS_CE["CLS to Score"]
    end
    subgraph BE["Bi-Encoder Sentence-BERT"]
        direction LR
        SA["Sentence A"] --> BERT_A["BERT"]
        SB["Sentence B"] --> BERT_B["BERT shared weights"]
        BERT_A --> POOL_A["Mean Pool emb_A"]
        BERT_B --> POOL_B["Mean Pool emb_B"]
        POOL_A --> COS["cosine_sim"]
        POOL_B --> COS
    end
    CE --- COMPARE{"Trade-off"}
    BE --- COMPARE
    COMPARE --> ACC["Cross-Encoder: Higher accuracy, 65 hours for 10K"]
    COMPARE --> SPD["Bi-Encoder: 5 seconds for 10K, ~5-10% less accurate"]
    style CE fill:#ff8787,stroke:#333
    style BE fill:#69db7c,stroke:#333
    style ACC fill:#ffe3e3,stroke:#333
    style SPD fill:#d3f9d8,stroke:#333
</pre>

### The Quality Gap and How to Close It

Bi-encoders are ~5-10% less accurate than cross-encoders for similarity tasks. The standard production pattern is the **retrieve-then-rerank pipeline**:

1. **Retrieve** top-100 candidates using bi-encoder (fast, milliseconds)
2. **Rerank** the 100 candidates using cross-encoder (accurate, still fast with only 100 pairs)

This gives you cross-encoder quality at bi-encoder speed. It's how virtually every production search system works today.

<pre class="mermaid">
graph LR
    QUERY["User Query"] --> EMBED["Bi-Encoder embed query"]
    EMBED --> ANN["ANN Search FAISS / Qdrant"]
    DB[("Vector DB 10M+ docs")] --> ANN
    ANN -->|"Top 100 ~5ms"| RERANK["Cross-Encoder Reranking"]
    RERANK -->|"Top 10 ~50ms"| RESULT["Final Results"]
    style QUERY fill:#74c0fc,stroke:#333
    style ANN fill:#ffd43b,stroke:#333
    style RERANK fill:#ff922b,stroke:#333,color:#fff
    style RESULT fill:#51cf66,stroke:#333,color:#fff
    style DB fill:#e599f7,stroke:#333
</pre>

```python
from sentence_transformers import SentenceTransformer, CrossEncoder, util

# Stage 1: Bi-encoder retrieval
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
corpus_embeddings = bi_encoder.encode(corpus, convert_to_tensor=True)
query_embedding = bi_encoder.encode(query, convert_to_tensor=True)

# Fast approximate nearest neighbours
hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=100)[0]

# Stage 2: Cross-encoder reranking
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
cross_inp = [[query, corpus[hit['corpus_id']]] for hit in hits]
cross_scores = cross_encoder.predict(cross_inp)

# Sort by cross-encoder scores
for idx in range(len(cross_scores)):
    hits[idx]['cross_score'] = cross_scores[idx]
hits = sorted(hits, key=lambda x: x['cross_score'], reverse=True)
```

---

## 11. Sentence-BERT: Architecture Details That Matter

### Pooling Strategy Matters

SBERT experiments showed three pooling strategies produce very different results:

| Pooling                | STS Benchmark (Spearman) |
| ---------------------- | ------------------------ |
| [CLS] token            | 29.19                    |
| Max pooling            | 82.32                    |
| **Mean pooling** | **83.18**          |

Mean pooling (averaging all token embeddings) won. [CLS] was catastrophically worse. This empirical result destroyed the common practice of using [CLS] as a sentence representation.

### Training Data Combination

SBERT's training strategy was: first train on **NLI data** (SNLI + MultiNLI, 570K sentence pairs with entailment/contradiction/neutral labels), then fine-tune on **STS data** (semantic textual similarity with continuous 0-5 scores).

The NLI stage gives the model a coarse understanding of sentence relationships. The STS stage calibrates the similarity scores. **This two-stage approach outperforms training on either dataset alone** — a lesson that transfers to most fine-tuning scenarios.

### The Objective Function

For NLI training, SBERT concatenates the two sentence embeddings and their element-wise difference, then classifies:

$$
o = \text{softmax}(W_t \cdot [u; v; |u-v|])
$$

where u and v are the sentence embeddings. The **|u-v|** term is crucial — it explicitly encodes the difference between the two representations, helping the model learn what makes sentences similar or different.

---

## 12. Fine-Tuning Embeddings: A Production Engineer's Guide

### Loss Functions — The Mathematics

**Contrastive Loss:**

$$
L = \frac{1}{2}(1-y) \cdot D^2 + \frac{1}{2}y \cdot \max(0, m - D)^2
$$

where D is the distance between embeddings, y=0 for similar pairs, y=1 for dissimilar pairs, m is the margin. Similar items are pulled together unconditionally; dissimilar items are pushed apart only if they're closer than margin m.

**Triplet Loss:**

$$
L = \max(0, \|a - p\|^2 - \|a - n\|^2 + \alpha)
$$

where a=anchor, p=positive, n=negative, α=margin. The model learns to keep the positive closer to the anchor than the negative by at least margin α.

**Multiple Negatives Ranking Loss (MNRL):**

$$
L = -\log \frac{e^{sim(a_i, p_i)/\tau}}{\sum_{j=1}^{N} e^{sim(a_i, p_j)/\tau}}
$$

This is an **in-batch softmax**. For a batch of N (anchor, positive) pairs, each anchor's positive is treated as a positive, and all other N-1 positives in the batch are treated as negatives. With batch size 64, you get 63 free negatives per example.

**Why MNRL dominates in practice:**

1. You only need positive pairs (cheaper to curate)
2. Larger batch sizes = more negatives = better gradients
3. Temperature τ controls the hardness of the distribution

### Hard Negative Mining: The 10x Multiplier

Random negatives are easy to distinguish — "What causes diabetes?" vs "How to cook pasta?" doesn't teach the model much. **Hard negatives** are semantically close but actually different:

- Query: "What causes type 2 diabetes?"
- Easy negative: "Best Italian restaurants in Mumbai"
- **Hard negative**: "What are the symptoms of type 2 diabetes?"

Hard negative mining strategies:

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives
from datasets import load_dataset

model = SentenceTransformer("all-MiniLM-L6-v2")
dataset = load_dataset("natural-questions", split="train")

# Mine hard negatives using current model's top-k
# These are passages the model currently ranks highly
# but are actually irrelevant
dataset = mine_hard_negatives(
    dataset=dataset,
    model=model,
    range_min=10,    # Skip top-10 (likely true positives)
    range_max=50,    # Use ranks 10-50 as hard negatives
    num_negatives=5, # 5 hard negatives per example
)
```

### Data Requirements — What Actually Works

| Training Data Size | Expected Impact                                  |
| ------------------ | ------------------------------------------------ |
| 100-500 pairs      | Noticeable domain adaptation                     |
| 1K-5K pairs        | Significant improvement                          |
| 10K-50K pairs      | Near-optimal for most domains                    |
| 100K+ pairs        | Diminishing returns (unless very diverse domain) |

**Critical rule:** Quality > Quantity. 1,000 carefully curated pairs from your domain outperform 100,000 noisy automatically-generated pairs.

---

## 13. The Embedding Anisotropy Problem

Here's something most tutorials completely ignore: **pre-trained embedding spaces are often anisotropic**, meaning embeddings cluster in a narrow cone of the high-dimensional space rather than being uniformly distributed.

**Why this matters:**

- In an anisotropic space, cosine similarity between random sentences averages ~0.6-0.8 instead of ~0.0
- This means similarity scores are less discriminative — the gap between "truly similar" and "random" is compressed
- High baseline similarity makes thresholding unreliable

**Detection:**

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
random_sentences = [...]  # 1000 random sentences

embeddings = model.encode(random_sentences)
# Compute mean pairwise cosine similarity
similarities = np.dot(embeddings, embeddings.T)
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
cosine_sim = similarities / (norms @ norms.T)
np.fill_diagonal(cosine_sim, 0)

avg_similarity = cosine_sim.sum() / (len(random_sentences) * (len(random_sentences) - 1))
print(f"Average pairwise cosine similarity: {avg_similarity:.4f}")
# Isotropic: ~0.0, Anisotropic: ~0.5-0.8
```

**Mitigation strategies:**

1. **Whitening** (Su et al., 2021): Apply PCA whitening to normalise the embedding distribution
2. **Fine-tuning with contrastive loss**: Naturally spreads the distribution
3. **Use models trained with better objectives**: Models trained with MNRL tend to be more isotropic

---

## 14. ColBERT: Late Interaction — A Third Way

Beyond cross-encoders and bi-encoders, there's a third architecture: **late interaction** (Khattab & Zaharia, 2020).

```
Query: "What causes diabetes?"
        → BERT → [q1, q2, q3, q4]    # Keep ALL token embeddings

Document: "Diabetes results from insulin resistance..."
        → BERT → [d1, d2, d3, d4, d5, d6]  # Keep ALL token embeddings

Score = Σ max_j(q_i · d_j)   # MaxSim operation
```

Instead of compressing to a single vector (bi-encoder) or cross-attending (cross-encoder), ColBERT:

1. Encodes query and document **independently** (like bi-encoder)
2. But keeps **all token embeddings** (unlike bi-encoder's pooling)
3. Computes a **MaxSim** score: for each query token, find its best-matching document token

<pre class="mermaid">
graph TB
    subgraph QE["Query Encoding"]
        QT["What causes diabetes?"] --> QB["BERT"] --> QV["q1, q2, q3, q4"]
    end
    subgraph DE["Document Encoding pre-computed"]
        DT["Diabetes results from..."] --> DB["BERT"] --> DV["d1, d2, d3, d4, d5, d6"]
    end
    subgraph MS["MaxSim Scoring"]
        direction LR
        M1["q1 best match among d1..d6"]
        M2["q2 best match among d1..d6"]
        M3["q3 best match among d1..d6"]
        M4["q4 best match among d1..d6"]
    end
    QV --> MS
    DV --> MS
    MS --> SUM["Score = Sum of MaxSim"]
    style MS fill:#ffd43b,stroke:#333
    style SUM fill:#51cf66,stroke:#333,color:#fff
</pre>

This achieves ~95% of cross-encoder quality whilst being **100x faster** at retrieval because document token embeddings can be pre-computed and indexed.

**The trade-off:** Storage. Instead of storing one 768-dim vector per document, you store N×128-dim vectors (N = number of tokens, dimensions compressed from 768 to 128). A 100M document index might require 100-200 GB.

---

## 15. Sparse-Dense Hybrid: SPLADE and the Best of Both Worlds

Pure dense retrieval (Sentence-BERT) misses **exact keyword matching**. The query "iPhone 15 Pro Max specifications" should match documents containing those exact terms, even if the dense embedding focuses on the general "phone specs" semantics.

**SPLADE** (Sparse Lexical and Expansion) learns **sparse representations** using BERT:

```python
# Conceptually:
# Instead of BERT → mean pool → 768d dense vector
# SPLADE does: BERT → MLM head → |V|-dimensional sparse vector
# where non-zero entries represent "expanded" terms

# A query about "ML deployment" might expand to:
# {"ML": 2.1, "machine": 1.8, "learning": 1.5,
#  "deployment": 2.3, "production": 1.2, "inference": 0.9,
#  "serving": 0.7, ...}
# Note: "production", "inference", "serving" weren't in the query
# but SPLADE learned they're relevant!
```

Modern production systems (Vespa, Weaviate, Qdrant) support **hybrid search** that combines dense and sparse scores:

$$
\text{score} = \alpha \cdot \text{dense\_score} + (1-\alpha) \cdot \text{sparse\_score}
$$

with α tuned per use case. This consistently outperforms either approach alone.

---

## 16. Matryoshka Embeddings: Adaptive Dimensionality

### The Core Idea

Standard models produce fixed-size embeddings (768d, 1024d). Matryoshka Representation Learning (Kusupati et al., 2022) trains the model so that **the first d dimensions form a valid embedding for any d**.

This is achieved by adding a multi-scale loss during training:

$$
L = \sum_{d \in \{32, 64, 128, 256, 512, 1024\}} L_d(\text{truncate}(e, d))
$$

The model simultaneously optimises for all truncation sizes. The result: the first 256 dimensions capture ~95% of the full-size performance, and even 64 dimensions retain ~85%.

### Production Impact

| Dimensions | Performance (Relative) | Storage (per embedding) | ANN Search Speed |
| ---------- | ---------------------- | ----------------------- | ---------------- |
| 1024       | 100%                   | 4 KB                    | 1x               |
| 256        | ~95%                   | 1 KB                    | ~4x faster       |
| 64         | ~85%                   | 256 B                   | ~16x faster      |

**Practical pattern:** Use 64d for fast initial candidate retrieval (top-1000), then re-score with full 1024d for the final ranking. You get maximum precision with minimum latency.

OpenAI's `text-embedding-3-small` and `text-embedding-3-large` both support this. The `dimensions` parameter lets you truncate at inference time — the model is already trained with the Matryoshka objective.

---

## 17. Instruction-Tuned Embeddings: E5 and BGE

A critical 2023-2024 development: **instruction-tuned embedding models** that accept a task description alongside the input text.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/e5-large-v2")

# The instruction prefix tells the model HOW to embed
query = "query: What causes Type 2 diabetes?"
passage = "passage: Type 2 diabetes results from insulin resistance..."

# vs for classification:
text = "classification: This patient shows signs of hyperglycaemia"
```

**Why this matters:** The same sentence should be embedded differently depending on the task. For retrieval, you want to capture the "query intent." For classification, you want to capture the "topic." For clustering, you want broad semantic features. Instruction tuning lets one model handle all tasks.

Models like **E5** (Wang et al., 2023), **BGE** (Xiao et al., 2023), and **NV-Embed-v2** (NVIDIA, 2024) use this approach and dominate the MTEB leaderboard.

---

## 18. Production Deployment: What Tutorials Never Tell You

### Quantisation: Shrinking Embeddings for Scale

Float32 embeddings (768d = 3KB per embedding) are expensive at scale. **Quantisation** reduces this:

| Format           | Bytes per 768d | Quality Retention | Speed-up       |
| ---------------- | -------------- | ----------------- | -------------- |
| Float32          | 3,072          | 100% (baseline)   | 1x             |
| Float16          | 1,536          | ~99.9%            | ~2x            |
| Int8             | 768            | ~99%              | ~4x            |
| **Binary** | **96**   | **~92-95%** | **~32x** |

**Binary quantisation** is particularly interesting: convert each dimension to 0/1, then use Hamming distance instead of cosine similarity. FAISS, Qdrant, and Weaviate all support this.

```python
import numpy as np

def binary_quantize(embedding):
    """Convert float embedding to binary."""
    return (embedding > 0).astype(np.uint8)

def hamming_similarity(a, b):
    """Fast binary similarity using bitwise XOR."""
    return 1.0 - np.count_nonzero(a != b) / len(a)

# 32x less storage, 10-30x faster search
binary_emb = binary_quantize(model.encode("query"))
```

### Embedding Drift and Index Maintenance

Models get updated. Your fine-tuned model improves. New data distributions emerge. **All of these invalidate your existing index.**

Production checklist:

1. **Version your embedding model**: Every index must track which model version generated it
2. **Blue-green index deployment**: Build new index with new model whilst old one serves traffic, then swap
3. **Monitor retrieval quality**: Track Recall@K, MRR on a golden evaluation set weekly
4. **Detect distribution drift**: Compare embedding statistics (mean, variance, average pairwise similarity) between batches

### Latency Budget Breakdown

For a typical RAG system targeting <200ms end-to-end:

```
Embedding query:           10-30ms  (GPU) / 50-100ms (CPU)
ANN search (FAISS/Qdrant): 1-5ms   (for 10M vectors)
Reranking (top-50):        30-80ms  (cross-encoder on GPU)
LLM generation:            100-500ms
─────────────────────────────
Total:                     141-615ms
```

**Key optimisations:**

- **Cache frequent query embeddings** (LRU cache with TTL)
- **Pre-compute and index document embeddings** (batch job, not real-time)
- **Use ONNX Runtime / TensorRT** for embedding model inference (~3x speed-up over PyTorch)
- **Matryoshka truncation** for first-pass retrieval, full dimensions for reranking

---

## 19. The Evaluation Problem: MTEB and Beyond

### MTEB (Massive Text Embedding Benchmark)

MTEB evaluates models across 8 task categories and 56+ datasets. But there are important caveats:

**Leaderboard position ≠ best model for you.** A model scoring highest on average might underperform on your specific task. Always evaluate on your own data.

**MTEB overweights English.** The recently launched **MMTEB** (Multilingual MTEB) addresses this with 250+ datasets across 200+ languages.

**Key metrics by task:**

- **Retrieval**: NDCG@10, Recall@100
- **STS**: Spearman correlation
- **Classification**: Accuracy, F1
- **Clustering**: V-measure

### How to Evaluate Your Own Embeddings

```python
from sentence_transformers import SentenceTransformer, evaluation

model = SentenceTransformer("your-fine-tuned-model")

# Retrieval evaluation
evaluator = evaluation.InformationRetrievalEvaluator(
    queries={"q1": "What is diabetes?", ...},
    corpus={"d1": "Diabetes is a chronic condition...", ...},
    relevant_docs={"q1": ["d1", "d5"], ...},  # Ground truth
    name="my-domain-eval",
    mrr_at_k=[10],
    ndcg_at_k=[10],
    recall_at_k=[10, 100]
)

results = evaluator(model)
print(f"MRR@10: {results['my-domain-eval_mrr@10']:.4f}")
print(f"NDCG@10: {results['my-domain-eval_ndcg@10']:.4f}")
print(f"Recall@100: {results['my-domain-eval_recall@100']:.4f}")
```

---

## 20. Where This Story Goes Next

The embedding landscape is evolving rapidly. Key directions:

**Multimodal Embeddings (CLIP, SigLIP, ImageBind):** Shared embedding spaces for text + images + audio + video. CLIP's contrastive training aligned 400M image-text pairs into a single space. This enables "search images with text" and vice versa.

**Multilingual at Scale:** LaBSE (Language-agnostic BERT Sentence Embedding) and mE5 create embeddings that are comparable across 100+ languages — you can search English documents with Hindi queries.

**LLM-based Embeddings:** Using decoder-only LLMs (Mistral, LLaMA) as embedding backbones instead of encoder-only BERT. Models like GritLM simultaneously perform generation and embedding with one model.

**Mixture-of-Experts Embeddings:** Routing different types of text to specialised embedding sub-networks, combining specialist quality with generalist coverage.

---

## The Arc of This Story

From counting words to understanding meaning. From sparse, high-dimensional vectors to dense, geometric spaces. From static representations to contextual, task-aware embeddings.

Each generation didn't just improve on the previous one — it revealed something new about how language and meaning can be computationally represented:

- **LSA** showed that meaning hides in co-occurrence statistics
- **Word2Vec** showed that prediction is a better training signal than counting
- **ELMo** showed that language has hierarchical structure (syntax → semantics)
- **BERT** showed that bidirectional context + transfer learning changes everything
- **SBERT** showed that practical efficiency matters as much as theoretical quality
- **Matryoshka** showed that information is not uniformly distributed across dimensions

The story of embeddings is the story of building better mirrors for meaning — and we're still learning what those mirrors can reflect.

---

## References

1. Deerwester, S. et al. (1990). *Indexing by Latent Semantic Analysis.* JASIS.
2. Bengio, Y. et al. (2003). *A Neural Probabilistic Language Model.* JMLR.
3. Mikolov, T. et al. (2013). *Efficient Estimation of Word Representations in Vector Space.* [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)
4. Mikolov, T. et al. (2013). *Distributed Representations of Words and Phrases and their Compositionality.* [NeurIPS](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality)
5. Pennington, J. et al. (2014). *GloVe: Global Vectors for Word Representation.* [EMNLP](https://nlp.stanford.edu/pubs/glove.pdf)
6. Levy, O. & Goldberg, Y. (2014). *Neural Word Embedding as Implicit Matrix Factorization.* [NeurIPS](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization)
7. Bojanowski, P. et al. (2017). *Enriching Word Vectors with Subword Information.* [TACL](https://arxiv.org/abs/1607.04606)
8. Vaswani, A. et al. (2017). *Attention Is All You Need.* [NeurIPS](https://arxiv.org/abs/1706.03762)
9. Peters, M.E. et al. (2018). *Deep Contextualized Word Representations.* [NAACL](https://arxiv.org/abs/1802.05365)
10. Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers.* [NAACL](https://arxiv.org/abs/1810.04805)
11. Liu, Y. et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* [arXiv:1907.11692](https://arxiv.org/abs/1907.11692)
12. Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* [EMNLP](https://arxiv.org/abs/1908.10084)
13. Khattab, O. & Zaharia, M. (2020). *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction.* [SIGIR](https://arxiv.org/abs/2004.12832)
14. Su, J. et al. (2021). *Whitening Sentence Representations for Better Semantics and Faster Retrieval.* [arXiv:2103.15316](https://arxiv.org/abs/2103.15316)
15. Kusupati, A. et al. (2022). *Matryoshka Representation Learning.* [NeurIPS](https://arxiv.org/abs/2205.13147)
16. Wang, L. et al. (2023). *Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5).* [ACL](https://arxiv.org/abs/2212.03533)
17. Muennighoff, N. et al. (2023). *MTEB: Massive Text Embedding Benchmark.* [EACL](https://arxiv.org/abs/2210.07316)
18. Lee, C. et al. (2024). *NV-Embed: Improved Techniques for Training LLM-based Embedding Models.* [arXiv:2405.17428](https://arxiv.org/abs/2405.17428)

---

*Written by Girijesh Prasad*
*20 February 2026*
