---
name: word2vec
description: Word2Vec embeddings using gensim. Use for learning vector representations from sequences, semantic similarity, action embeddings, and NLP feature engineering. Works with text, game actions, or any sequential data.
---

# Word2Vec Embeddings Guide

Word2Vec learns dense vector representations where semantically similar items are close in vector space. Uses skip-gram or CBOW neural architectures via gensim.

## Quick Start

### Installation

```bash
pip install gensim
```

### Basic Usage

```python
from gensim.models import Word2Vec

# Prepare sequences (list of lists of tokens)
sequences = [
    ["action_a", "action_b", "action_c"],
    ["action_b", "action_d", "action_a"],
    # ... more sequences
]

# Train model
model = Word2Vec(
    sequences,
    vector_size=100,    # Embedding dimensionality
    window=5,           # Context window size
    min_count=1,        # Minimum token frequency
    workers=4,          # Parallel threads
    sg=1,               # 1=skip-gram, 0=CBOW
    epochs=10,          # Training iterations
)

# Get vector for a token
vector = model.wv["action_a"]

# Find similar tokens
similar = model.wv.most_similar("action_a", topn=5)
```

## Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `vector_size` | 100 | Embedding dimensions (50-300 typical) |
| `window` | 5 | Context window on each side |
| `min_count` | 5 | Minimum token frequency |
| `sg` | 0 | 0=CBOW, 1=skip-gram |
| `negative` | 5 | Negative sampling count |
| `epochs` | 5 | Training passes over data |

### Skip-gram vs CBOW

- **Skip-gram** (`sg=1`): Better for rare tokens, smaller datasets
- **CBOW** (`sg=0`): Faster training, better for frequent tokens

## Working with Vectors

### Similarity Operations

```python
# Cosine similarity between two tokens
similarity = model.wv.similarity("token_a", "token_b")

# Most similar tokens
similar = model.wv.most_similar("token_a", topn=10)
# Returns: [('token_b', 0.89), ('token_c', 0.85), ...]

# Analogy: A is to B as C is to ?
result = model.wv.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=1
)
```

### Vector Arithmetic

```python
import numpy as np

# Average multiple vectors
tokens = ["token_a", "token_b", "token_c"]
vectors = [model.wv[t] for t in tokens if t in model.wv]
avg_vector = np.mean(vectors, axis=0)

# Weighted combination
weighted = 0.7 * model.wv["token_a"] + 0.3 * model.wv["token_b"]
```

## Saving and Loading

```python
# Save full model (can continue training)
model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")

# Save just vectors (smaller, read-only)
model.wv.save("word2vec.wordvectors")

from gensim.models import KeyedVectors
wv = KeyedVectors.load("word2vec.wordvectors")
```

### Standard Word2Vec Format

```python
# Save in standard format (compatible with other tools)
model.wv.save_word2vec_format("vectors.txt", binary=False)
model.wv.save_word2vec_format("vectors.bin", binary=True)

# Load pre-trained vectors
wv = KeyedVectors.load_word2vec_format("vectors.bin", binary=True)
```

## Training on Large Data

### Streaming from Files

```python
from gensim.models.word2vec import LineSentence

# One sentence per line, space-separated tokens
sentences = LineSentence("corpus.txt")
model = Word2Vec(sentences, vector_size=100)
```

### Custom Iterator

```python
class MyCorpus:
    def __init__(self, data_path):
        self.data_path = data_path

    def __iter__(self):
        for line in open(self.data_path):
            yield line.strip().split()

corpus = MyCorpus("data.txt")
model = Word2Vec(corpus, vector_size=100)
```

### Incremental Training

```python
# Build vocabulary first
model.build_vocab(sequences)

# Train in batches
for batch in batches:
    model.train(batch, total_examples=len(batch), epochs=1)
```

## Feature Engineering

### Document/Sequence Embeddings

```python
def sequence_embedding(tokens, model):
    """Average token vectors for sequence embedding."""
    vectors = [model.wv[t] for t in tokens if t in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)

# Apply to dataset
embeddings = np.array([
    sequence_embedding(seq, model) for seq in sequences
])
```

### For Downstream ML

```python
from sklearn.preprocessing import normalize

# Get all vocabulary vectors
vocab = list(model.wv.key_to_index.keys())
X = np.array([model.wv[w] for w in vocab])

# L2 normalize for cosine similarity with dot product
X_normalized = normalize(X, norm='l2')
```

## Application: Game Action Embeddings

```python
# For Texas 42: embed game actions
action_sequences = [
    ["bid_30", "pass", "pass", "bid_31", "declare_trumps"],
    ["lead_5-5", "follow_5-4", "follow_5-3", "follow_5-0"],
    # ... sequences from game logs
]

model = Word2Vec(
    action_sequences,
    vector_size=64,
    window=3,      # Actions within same trick/phase
    min_count=1,   # Keep all actions
    sg=1,          # Skip-gram for sparse actions
)

# Find similar actions
model.wv.most_similar("lead_5-5")
# Returns other high-trump leads
```

## Visualization

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Get vectors for visualization
words = list(model.wv.key_to_index.keys())[:100]
vectors = np.array([model.wv[w] for w in words])

# Reduce to 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
coords = tsne.fit_transform(vectors)

# Plot
plt.figure(figsize=(12, 8))
plt.scatter(coords[:, 0], coords[:, 1], alpha=0.6)
for i, word in enumerate(words):
    plt.annotate(word, coords[i], fontsize=8)
plt.title("Word2Vec Embedding Space (t-SNE)")
plt.show()
```

## Evaluation

### Intrinsic Evaluation

```python
# Word analogy accuracy (if applicable)
model.wv.evaluate_word_analogies("analogies.txt")

# Word similarity correlation
model.wv.evaluate_word_pairs("similarity.tsv")
```

### Cluster Quality

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Cluster embeddings
vocab = list(model.wv.key_to_index.keys())
X = np.array([model.wv[w] for w in vocab])

kmeans = KMeans(n_clusters=10, random_state=42)
labels = kmeans.fit_predict(X)

score = silhouette_score(X, labels)
print(f"Silhouette score: {score:.3f}")
```

## Best Practices

| DO | DON'T |
|----|-------|
| Preprocess tokens consistently | Mix cased/uncased tokens |
| Set `min_count` based on corpus size | Keep all rare tokens (noise) |
| Use skip-gram for small/sparse data | Default to CBOW blindly |
| Evaluate on downstream task | Only use intrinsic metrics |
| Normalize vectors for similarity | Use raw vectors with cosine |
| Set `random_state` for reproducibility | Expect exact same results |

## Hyperparameter Tuning

```python
from itertools import product

# Grid search
param_grid = {
    'vector_size': [50, 100, 200],
    'window': [3, 5, 10],
    'min_count': [1, 5],
    'sg': [0, 1],
}

best_score = 0
best_params = None

for vs, w, mc, sg in product(*param_grid.values()):
    model = Word2Vec(
        sequences,
        vector_size=vs, window=w,
        min_count=mc, sg=sg,
        epochs=10
    )
    # Evaluate on downstream task
    score = evaluate_downstream(model)
    if score > best_score:
        best_score = score
        best_params = {'vector_size': vs, 'window': w, 'min_count': mc, 'sg': sg}

print(f"Best params: {best_params}, score: {best_score:.3f}")
```

## Common Issues

| Issue | Solution |
|-------|----------|
| OOV (out-of-vocabulary) tokens | Use `min_count=1` or handle missing |
| Poor similarity results | Increase `vector_size`, more training data |
| Slow training | Use `workers` for parallelism, reduce `epochs` |
| Memory issues | Use streaming iterator, not list |

## Resources

- **Gensim Documentation**: https://radimrehurek.com/gensim/models/word2vec.html
- **Word2Vec Tutorial**: https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
- **Original Paper**: Mikolov et al. (2013) - "Efficient Estimation of Word Representations in Vector Space"
