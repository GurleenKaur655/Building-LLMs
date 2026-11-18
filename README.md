# Building LLMs

A hands-on learning repo where I build a GPT-style large language model from scratch in Python and PyTorch, one concept per script.

This isn't about training a production-scale model like ChatGPT. It's about understanding every building block myself: going from raw text to tokens, through attention and the full GPT-2 architecture, to pretraining, loading OpenAI's real GPT-2 weights, and fine-tuning the model to classify SMS messages as spam or not spam.

## Progress So Far

- [x] **Stage 1: Working with text data.** Tokenization, BPE, input-target pairs, data loaders, embeddings
- [x] **Stage 2: Attention mechanisms.** Self-attention, causal attention, multi-head attention
- [x] **Stage 3: GPT architecture.** Layer norm, GELU, shortcut connections, transformer block, full 124M-parameter GPT-2
- [x] **Stage 4: Pretraining.** Loss function, evaluation, training loop, temperature and top-k sampling
- [x] **Stage 5: Pretrained weights.** Loading OpenAI's GPT-2 (124M) weights into my own model
- [x] **Stage 6: Fine-tuning for classification.** SMS spam classifier built on top of GPT-2

## Project Structure

```text
Building-LLMs/
├── README.md
├── 01_tokenizer.py ... 24_finetuningforclassification.py   # one script per concept
├── gpt_download3.py              # downloads and loads OpenAI GPT-2 weights
├── the-verdict.txt               # small text corpus used for pretraining
├── sms_spam_collection/          # SMS Spam Collection dataset
└── train.csv / validation.csv / test.csv   # balanced spam dataset splits
```

## Scripts

### Stage 1: Working with text data

| Script | What it does |
| --- | --- |
| `01_tokenizer.py` | Regex-based word tokenizer built from scratch (`SimpleTokenizerV1`), then extended with `<\|unk\|>` and `<\|endoftext\|>` special tokens (`SimpleTokenizerV2`) |
| `02_BPE.py` | GPT-2 byte pair encoding with `tiktoken`, showing how unknown words split into subword pieces |
| `03_input-target-pairs.py` | Creates shifted input-target token pairs for next-token prediction |
| `04_data_loader.py` | `GPTDatasetV1` and a PyTorch `DataLoader` using a sliding window with `max_length` and `stride` |
| `05_token_embeddings.py` | Turns token IDs into dense vectors with `nn.Embedding` |
| `06_position_encoding.py` | Adds absolute positional embeddings to token embeddings |

### Stage 2: Attention mechanisms

| Script | What it does |
| --- | --- |
| `07_attention.py` | Simplified attention: attention scores, softmax weights, context vectors |
| `08_selfattention.py` | Self-attention with trainable query, key and value weight matrices |
| `09_casualattention.py` | Causal (masked) attention so tokens can't see the future, plus dropout |
| `10_multiheadattention.py` | Multi-head attention by stacking several causal attention heads |
| `11_multiheadattentionwithweightsplits.py` | Efficient `MultiHeadAttention` class that splits one set of weights across heads |

### Stage 3: GPT architecture

| Script | What it does |
| --- | --- |
| `12_gptarchitecture.py` | GPT config and a placeholder "dummy" GPT model to see the overall structure |
| `13_layernormalization.py` | Layer normalization from scratch |
| `14_geluactivation.py` | GELU activation and the feed-forward network |
| `15_shortcutconnections.py` | Shortcut (residual) connections and how they help gradients flow during backpropagation |
| `16_transformerblock.py` | Full transformer block: attention + feed-forward + layer norm + shortcuts |
| `17_predictnexttoken.py` | Complete `GPTModel` (124M parameters) and greedy text generation with untrained weights |

### Stage 4: Pretraining

| Script | What it does |
| --- | --- |
| `18_llmlossfunction.py` | Cross-entropy loss and perplexity for next-token prediction |
| `19_evaluatingllmperformance.py` | Training/validation loss on `the-verdict.txt`. Also the shared module other scripts import (model, data loader, loss helpers) |
| `20_pretrainingllm.py` | Training loop with AdamW. Pretrains the model for 10 epochs and saves `model.pth` |
| `21_temperaturescaling.py` | Temperature scaling to control randomness in generation |
| `22_topksampling.py` | Top-k sampling and an improved `generate()` function to reduce memorized output |

### Stage 5: Loading pretrained weights

| Script | What it does |
| --- | --- |
| `gpt_download3.py` | Downloads OpenAI's GPT-2 TensorFlow checkpoint and converts it to NumPy params |
| `23_pretrainedweights.py` | Maps OpenAI's GPT-2 (124M) weights into my own `GPTModel` and generates coherent text |

### Stage 6: Fine-tuning for classification

`24_finetuningforclassification.py` turns GPT-2 into an SMS spam classifier:

1. **Data:** downloads the UCI SMS Spam Collection, balances spam/ham by undersampling, and splits into 70% train / 10% validation / 20% test (`train.csv`, `validation.csv`, `test.csv`)
2. **Dataset:** a `SpamDataset` class that tokenizes and pads messages to the same length
3. **Model:** loads pretrained GPT-2 (124M) and replaces the 50,257-word output head with a 2-class head
4. **Freezing:** freezes the model and only trains the last transformer block, the final layer norm and the new output head
5. **Training:** uses the last token's output for classification, cross-entropy loss, and AdamW (lr 5e-5) for 5 epochs, tracking accuracy each epoch
6. **Inference:** a `classify_review()` function that labels new messages as `spam` or `not spam`, then saves the model as `review_classifier.pth`

Before fine-tuning, the model scores about 46% accuracy (close to random guessing). Fine-tuning teaches it to tell spam apart from normal messages.

## Getting Started

### 1. Install dependencies

```bash
python3 -m pip install torch tiktoken numpy pandas tensorflow tqdm requests
```

`tensorflow` is only needed to read OpenAI's GPT-2 checkpoint files (scripts 23 and 24).

### 2. Run scripts from the repo root

The scripts use relative paths and import each other with `importlib`, so run them from the repo root:

```bash
python3 "01_tokenizer.py"
```

Then work through them in order, for example:

```bash
python3 "20_pretrainingllm.py"
```

```bash
python3 "24_finetuningforclassification.py"
```

## Notes

- `02_BPE.py` and later scripts use `tiktoken`, which may download the GPT-2 encoding files on first run.
- Scripts 23 and 24 download the GPT-2 124M weights (about 500 MB) into a `gpt2/` folder on first run.
- Trained weights (`*.pth`) and the `gpt2/` folder are git-ignored because they're too large to commit.
- Training runs on CPU, but a GPU is much faster for script 24.

## Why I Made This

I wanted to understand how LLMs work beyond the buzzwords. I can't train a massive model from scratch, but I can learn the core ideas by building every piece myself and documenting the process.

This repo is that learning journey, from a simple regex tokenizer to a fine-tuned GPT-2 classifier.

## What's Next

- Instruction fine-tuning, so the model can follow instructions like a chatbot
- Building two projects with Ollama and adding AI features to them, so I can use these ideas in practical, real-world projects
