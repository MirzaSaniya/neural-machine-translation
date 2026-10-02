# Neural Machine Translation with Seq2Seq and Attention

A neural machine translation system implemented in PyTorch using sequence-to-sequence modeling, learned word embeddings, recurrent neural networks, and attention.

> **Portfolio Project**  |  **NLP • Deep Learning • PyTorch • Machine Translation**

## Overview

Neural machine translation maps a source-language sentence to a target-language sentence using an end-to-end neural model. This project implements the core components of a sequence-to-sequence translation system and exposes the mechanics behind encoding, decoding, attention, training, and inference.

Rather than treating translation as a collection of independent word-level decisions, the model learns a conditional sequence model that uses the source sentence to generate the target sequence one token at a time.

## What This Project Demonstrates

- Sequence-to-sequence neural modeling
- Encoder-decoder architecture
- Word embedding layers
- Recurrent sequence modeling
- Attention-based decoding
- Teacher-forced training
- Autoregressive inference
- Tensor shape management and batching in PyTorch
- NLP model training and evaluation

## Architecture

```text
Source sentence
      ↓
Source embeddings
      ↓
Encoder RNN
      ↓
Encoded source representations
      ↓
Attention mechanism  ←  Decoder hidden state
      ↓
Context vector
      ↓
Decoder RNN
      ↓
Target vocabulary distribution
      ↓
Next translated token
      ↺
```

During decoding, attention allows the model to form a context vector from the encoded source states instead of compressing the entire source sentence into a single fixed-size representation.

## Repository Structure

```text
neural-machine-translation/
├── README.md
├── .gitignore
├── src/
│   ├── model_embeddings.py
│   ├── nmt_model.py
│   ├── utils.py
│   └── __init__.py
└── results/
    └── README.md
```

## Core Components

### `model_embeddings.py`
Defines the source and target embedding layers used to transform token IDs into dense vector representations.

### `nmt_model.py`
Contains the main neural machine translation model, including the encoder-decoder pipeline and attention-based decoding logic.

### `utils.py`
Provides supporting utilities used by the translation workflow.

## Technology Stack

| Area | Technology |
| --- | --- |
| Language | Python |
| Deep learning | PyTorch |
| NLP | Neural machine translation |
| Sequence model | Recurrent encoder-decoder |
| Mechanism | Attention |

## Why Attention Matters

A sequence-to-sequence model benefits from being able to reference different source positions while generating different target words. Attention provides that dynamic alignment mechanism, helping the decoder focus on the source representations most relevant to the next prediction.

## Learning Outcomes

This project connects the mathematical ideas behind seq2seq translation with an executable PyTorch implementation. It also provides practical experience with sequence lengths, padding, masks, hidden states, batched computation, and autoregressive decoding.

## Course Context

Developed as part of **Stanford XCS224N: Natural Language Processing with Deep Learning** coursework.

## Notes on Reproducibility

Large trained parameter files, caches, logs, temporary outputs, and submission archives are excluded from the public repository by default. The repository is intended to showcase the implementation and the underlying modeling ideas.

## Status

**Completed coursework implementation - portfolio packaged**

## Author

**Saniya Mirza**
