"""
Prepare a small subset of OpenWebText (~50M tokens) for scaling law experiments.
Much faster than full OWT (9B tokens), but large enough to avoid data bottleneck.
"""
import os
import numpy as np
import tiktoken
from datasets import load_dataset

from local_learning_nanogpt.paths import project_data_dir

TARGET_TOKENS = 50_000_000  # 50M tokens (50x Shakespeare)
VAL_TOKENS = 500_000        # 500K val tokens

enc = tiktoken.get_encoding("gpt2")
out_dir = project_data_dir("owt_small")
os.makedirs(out_dir, exist_ok=True)

print("Loading OpenWebText (streaming mode)...")
dataset = load_dataset("openwebtext", split="train", streaming=True)

# Collect tokens until we have enough
train_tokens = []
val_tokens = []
total = 0

for example in dataset:
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)

    if len(val_tokens) < VAL_TOKENS:
        val_tokens.extend(ids)
    else:
        train_tokens.extend(ids)

    total += len(ids)
    if total % 5_000_000 == 0:
        print(f"  Collected {total/1e6:.1f}M tokens...")

    if len(train_tokens) >= TARGET_TOKENS and len(val_tokens) >= VAL_TOKENS:
        break

# Truncate to exact sizes
train_tokens = train_tokens[:TARGET_TOKENS]
val_tokens = val_tokens[:VAL_TOKENS]

print(f"Train: {len(train_tokens):,} tokens")
print(f"Val: {len(val_tokens):,} tokens")
print(f"Vocab size: {enc.n_vocab}")

# Save as binary
train_arr = np.array(train_tokens, dtype=np.uint16)
val_arr = np.array(val_tokens, dtype=np.uint16)
train_arr.tofile(os.path.join(out_dir, 'train.bin'))
val_arr.tofile(os.path.join(out_dir, 'val.bin'))

# Save meta
import pickle
meta = {
    'vocab_size': enc.n_vocab,  # 50257 (GPT-2 BPE)
    'tokenizer': 'gpt2',
}
with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"Saved to {out_dir}/")
print(f"  train.bin: {os.path.getsize(os.path.join(out_dir, 'train.bin'))/1e6:.1f}MB")
print(f"  val.bin: {os.path.getsize(os.path.join(out_dir, 'val.bin'))/1e6:.1f}MB")
