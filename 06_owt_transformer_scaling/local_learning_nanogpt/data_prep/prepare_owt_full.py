"""
Prepare the full OpenWebText dataset (~9B tokens) for large-scale experiments.

This streams the entire OpenWebText dataset and saves it as binary files.
Expected: ~9B train tokens + 5M val tokens.
Estimated download + tokenize time: 30-60 minutes.
Disk usage: ~18 GB (uint16).
"""
import os
import numpy as np
import tiktoken
from datasets import load_dataset

from local_learning_nanogpt.paths import project_data_dir

TARGET_TOKENS = None  # None = use all available
VAL_TOKENS = 5_000_000  # 5M val tokens (10× the small subset)

enc = tiktoken.get_encoding("gpt2")
out_dir = project_data_dir("owt_full")
os.makedirs(out_dir, exist_ok=True)

print("Loading OpenWebText (streaming mode) — full dataset...")
print("This will take 30-60 minutes to download and tokenize.")
dataset = load_dataset("openwebtext", split="train", streaming=True)

# Use chunked writing to avoid memory issues with 9B tokens
train_path = os.path.join(out_dir, 'train.bin')
val_path = os.path.join(out_dir, 'val.bin')

val_tokens = []
train_chunks = []
train_total = 0
chunk_size = 50_000_000  # 50M tokens per chunk, flush to disk
chunk_idx = 0

# Write in append mode
if os.path.exists(train_path):
    os.remove(train_path)

import time
t0 = time.time()

for example in dataset:
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)

    if len(val_tokens) < VAL_TOKENS:
        val_tokens.extend(ids)
    else:
        train_chunks.extend(ids)
        train_total += len(ids)

        # Flush chunk to disk
        if len(train_chunks) >= chunk_size:
            arr = np.array(train_chunks[:chunk_size], dtype=np.uint16)
            with open(train_path, 'ab') as f:
                arr.tofile(f)
            train_chunks = train_chunks[chunk_size:]
            chunk_idx += 1
            elapsed = time.time() - t0
            rate = train_total / elapsed / 1e6
            print(f"  Chunk {chunk_idx}: {train_total/1e9:.2f}B tokens, "
                  f"{rate:.1f}M tok/s, {elapsed/60:.0f}min elapsed")

# Flush remaining
if train_chunks:
    arr = np.array(train_chunks, dtype=np.uint16)
    with open(train_path, 'ab') as f:
        arr.tofile(f)

# Save val
val_tokens = val_tokens[:VAL_TOKENS]
val_arr = np.array(val_tokens, dtype=np.uint16)
val_arr.tofile(val_path)

elapsed = time.time() - t0
print(f"\nDone in {elapsed/60:.0f} minutes.")
print(f"Train: {train_total:,} tokens ({train_total/1e9:.2f}B)")
print(f"Val: {len(val_tokens):,} tokens")

# Save meta
import pickle
meta = {
    'vocab_size': enc.n_vocab,
    'tokenizer': 'gpt2',
    'n_train_tokens': train_total,
}
with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"Saved to {out_dir}/")
print(f"  train.bin: {os.path.getsize(train_path)/1e9:.2f}GB")
print(f"  val.bin: {os.path.getsize(val_path)/1e6:.1f}MB")
