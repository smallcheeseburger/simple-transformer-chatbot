# Transformer-based Chatbot

A simple Transformer chatbot trained from scratch on the [Blended Skill Talk](https://huggingface.co/datasets/blended_skill_talk) dataset, with support for `word-level` and `SentencePiece BPE tokenizer`.  
This project demonstrates the full workflow of building, training, evaluating, and inferring responses from a transformer model for dialogue generation.

## Project Structure

```text
models/
├── inference.py # Inference entry point
├── save_corpus.py # Export corpus.txt for BPE tokenizer training
├── tokenizer.py # Tokenizer definition (Word-level or SentencePiece)
├── train.py # Training script (Word-level)
├── train_bpe.py # Training script (BPE)
├── transformer_model.py # Transformer architecture definition
bpe.model # Trained SentencePiece model
bpe.vocab # Trained SentencePiece vocab
corpus.txt # Exported corpus used for SentencePiece training
transformer.pth # Saved best model weights
best_accuracy.pkl # Saved best validation accuracy
```

## Features

- End-to-end training on dialogue data  
- Word-level tokenizer and BPE tokenizer support  
- Early stopping and learning rate scheduler  
- Top-k sampling & temperature control during inference  
- Validation loss & accuracy tracking


## 🛠️ Requirements

You can install requirements using:

```bash
pip install torch tqdm sentencepiece datasets
```

⚙️ Usage

1️⃣ Prepare corpus for BPE training (optional, if you want SentencePiece tokenizer)
```bash
python models/save_corpus.py
```
2️⃣ Train SentencePiece tokenizer (optional)
```bash
python models/train_bpe.py
```
3️⃣ Train model
Word-level tokenizer:
```bash
python models/train.py
```
BPE tokenizer: Make sure tokenizer.py loads bpe.model correctly.

4️⃣ Inference (generate chatbot replies)
```bash
python models/inference.py
```