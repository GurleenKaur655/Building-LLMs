# Building LLMs

A small hands-on learning repo where I explore how large language models work by building the early pieces in Python.

This project is not about training a production-scale model like ChatGPT. It is about understanding the building blocks step by step: tokenization, byte pair encoding, and preparing input-target pairs for next-token prediction.

## What This Repo Covers

- A simple regex-based tokenizer built from scratch
- A second tokenizer version with special tokens like `<|endoftext|>` and `<|unk|>`
- GPT-2 style byte pair encoding with `tiktoken`
- Creating shifted input-target token pairs for language model training

## Project Structure

```text
Building-LLMs/
├── README.md
└── Building LLMs/
    ├── 01_tokenizer.py
    ├── 02_BPE.py
    ├── 03_input-target-pairs.py
    └── the-verdict.txt
```

## Files

### `01_tokenizer.py`

Builds a basic word-level tokenizer using regular expressions. This script:

- reads `the-verdict.txt`
- splits the text into tokens
- creates a vocabulary and token IDs
- implements `SimpleTokenizerV1`
- extends it into `SimpleTokenizerV2` with special tokens for unknown words and end-of-text markers

### `02_BPE.py`

Introduces GPT-2 byte pair encoding using `tiktoken`. This script shows:

- how GPT-style subword tokenization works
- how special tokens are handled
- how unknown-looking words can still be broken into smaller token pieces

### `03_input-target-pairs.py`

Prepares training examples for next-token prediction. This script:

- tokenizes the text with GPT-2 encoding
- creates a context window
- shifts the sequence by one token to build input-target pairs

### `the-verdict.txt`

Sample text used as the corpus for the tokenizer and training-data examples.

## Getting Started

### 1. Install Python dependencies

```bash
python3 -m pip install tiktoken
```

### 2. Run the scripts from inside the `Building LLMs` folder

The scripts read `the-verdict.txt` using a relative path, so run them from that directory:

```bash
cd "Building LLMs"
python3 "01_tokenizer.py"
python3 "02_BPE.py"
python3 "03_input-target-pairs.py"
```

## Important Note

`01_tokenizer.py` runs locally without extra downloads after Python is available.

`02_BPE.py` and `03_input-target-pairs.py` use `tiktoken`. On first run, `tiktoken` may need access to download GPT-2 encoding files unless they are already cached on your machine.

## Why I Made This

I wanted to understand how LLMs work beyond the buzzwords. Even if I cannot train a massive model from scratch, I can still learn the core ideas by building smaller pieces myself and documenting the process.

This repo is that learning journey.

## What's Next

I will keep learning and expanding this repo as I go.

My next goal is to build two projects with Ollama and add AI integration into them so I can apply these concepts in more practical, real-world projects.
