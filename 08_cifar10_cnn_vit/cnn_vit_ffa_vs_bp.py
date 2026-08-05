"""
CNN/ViT: FFA vs BP Fair Comparison on CIFAR-10 (10-class)
==========================================================
Proper implementation:
  - FFA: label-overlay protocol (Hinton 2022), 10-class inference
  - BP:  standard cross-entropy, same architecture
  - Metric: test accuracy (10-class)
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

NUM_CLASSES = 10
LABEL_DIM = 10  # one-hot label overlay dimension

# ==================== Data ====================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Loading CIFAR-10...")
train_dataset = datasets.CIFAR10(root='/tmp/cifar10_data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='/tmp/cifar10_data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")


def overlay_label(x_flat, y, num_classes=10):
    """Overlay one-hot label into first num_classes pixels (Hinton 2022 protocol)."""
    x_out = x_flat.clone()
    one_hot = torch.zeros(x_flat.shape[0], num_classes, device=x_flat.device)
    one_hot.scatter_(1, y.unsqueeze(1), 1.0)
    x_out[:, :num_classes] = one_hot
    return x_out


def make_negative_labels(y, num_classes=10):
    """Generate random wrong labels."""
    wrong = torch.randint(1, num_classes, (y.shape[0],), device=y.device)
    return (y + wrong) % num_classes


# ==================== CNN Architecture ====================
class CNN(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Linear(128 * 4 * 4, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)


# ==================== ViT Architecture ====================
class SimpleViT(nn.Module):
    def __init__(self, out_dim=256, patch_size=4, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        num_patches = (32 // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=256,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, out_dim)

    def forward(self, x):
        x = self.patch_embed(x)            # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)   # (B, num_patches, embed_dim)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.mean(dim=1)                  # global avg pool
        return self.head(x)


# ==================== FFA Layer ====================
class FFAHead(nn.Module):
    """FFA head: takes image features, overlays label, produces goodness."""
    def __init__(self, feat_dim, hidden_dim=512):
        super().__init__()
        # Input: feat_dim (backbone output) + LABEL_DIM (one-hot overlay)
        # But we overlay on flattened image, so backbone sees overlaid input
        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.relu(self.ln1(self.fc1(x)))
        h = self.relu(self.ln2(self.fc2(h)))
        return h

    def goodness(self, h):
        return (h ** 2).mean(dim=1)  # mean, not sum, to keep scale ~O(1)


# ==================== Training Functions ====================

def train_bp(model, classifier, train_loader, test_loader, epochs=50, lr=0.001):
    """Standard BP training with cross-entropy."""
    optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        classifier.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            feat = model(x)
            logits = classifier(feat)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.shape[0]

        train_acc = correct / total

        # Test
        model.eval()
        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                feat = model(x)
                logits = classifier(feat)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.shape[0]
        test_acc = correct / total

        if (epoch + 1) % 10 == 0:
            print(f"    [BP] Epoch {epoch+1}/{epochs}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

    return test_acc


def train_ffa(backbone, ffa_head, train_loader, test_loader, epochs=50, lr=0.001):
    """FFA training with label-overlay protocol."""
    optimizer = optim.Adam(list(backbone.parameters()) + list(ffa_head.parameters()), lr=lr)

    for epoch in range(epochs):
        backbone.train()
        ffa_head.train()
        total_loss = 0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x_flat = x.reshape(x.shape[0], -1)

            # Positive: correct label overlay
            x_pos = overlay_label(x_flat, y).reshape(x.shape)
            feat_pos = backbone(x_pos)
            h_pos = ffa_head(feat_pos)
            g_pos = ffa_head.goodness(h_pos)

            # Negative: wrong label overlay
            y_neg = make_negative_labels(y)
            x_neg = overlay_label(x_flat, y_neg).reshape(x.shape)
            feat_neg = backbone(x_neg)
            h_neg = ffa_head(feat_neg)
            g_neg = ffa_head.goodness(h_neg)

            # Dynamic theta: midpoint of pos/neg means
            theta = (g_pos.mean() + g_neg.mean()).detach() / 2

            # FFA loss: push g_pos above theta, g_neg below theta
            loss_pos = torch.log(1 + torch.exp(-(g_pos - theta))).mean()
            loss_neg = torch.log(1 + torch.exp(g_neg - theta)).mean()
            loss = loss_pos + loss_neg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(ffa_head.parameters()), 1.0
            )
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Test: try all 10 labels, pick highest goodness
        backbone.eval()
        ffa_head.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                x_flat = x.reshape(x.shape[0], -1)
                batch_size = x.shape[0]

                # Try all 10 labels
                best_g = torch.full((batch_size,), -float('inf'), device=device)
                best_label = torch.zeros(batch_size, dtype=torch.long, device=device)

                for c in range(NUM_CLASSES):
                    y_try = torch.full((batch_size,), c, dtype=torch.long, device=device)
                    x_try = overlay_label(x_flat, y_try).reshape(x.shape)
                    feat = backbone(x_try)
                    h = ffa_head(feat)
                    g = ffa_head.goodness(h)

                    mask = g > best_g
                    best_g[mask] = g[mask]
                    best_label[mask] = c

                correct += (best_label == y).sum().item()
                total += batch_size

        test_acc = correct / total

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / n_batches
            print(f"    [FFA] Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, test_acc={test_acc:.4f}")

    return test_acc


# ==================== Run Experiments ====================
print("\n" + "="*70)
print("CNN/ViT: FFA vs BP on CIFAR-10 (10-class, proper comparison)")
print("="*70)

results = {}
EPOCHS = 50
FEAT_DIM = 256

configs = [
    ('CNN', lambda: CNN(out_dim=FEAT_DIM).to(device)),
    ('ViT', lambda: SimpleViT(out_dim=FEAT_DIM).to(device)),
]

for arch_name, make_backbone in configs:
    print(f"\n{'='*50}")
    print(f"Architecture: {arch_name}")
    print(f"{'='*50}")

    # --- BP Baseline ---
    print(f"\n  Training {arch_name} with BP (cross-entropy)...")
    backbone_bp = make_backbone()
    classifier = nn.Linear(FEAT_DIM, NUM_CLASSES).to(device)
    bp_acc = train_bp(backbone_bp, classifier, train_loader, test_loader, epochs=EPOCHS, lr=0.001)
    print(f"  BP final test accuracy: {bp_acc:.4f}")

    # --- FFA ---
    print(f"\n  Training {arch_name} with FFA (label-overlay)...")
    backbone_ffa = make_backbone()
    ffa_head = FFAHead(feat_dim=FEAT_DIM, hidden_dim=512).to(device)
    ffa_acc = train_ffa(backbone_ffa, ffa_head, train_loader, test_loader, epochs=EPOCHS, lr=0.001)
    print(f"  FFA final test accuracy: {ffa_acc:.4f}")

    results[arch_name] = {
        'bp_acc': float(bp_acc),
        'ffa_acc': float(ffa_acc),
        'gap': float(bp_acc - ffa_acc),
        'epochs': EPOCHS,
    }

    print(f"\n  >>> {arch_name}: BP={bp_acc:.2%} vs FFA={ffa_acc:.2%} (gap={bp_acc-ffa_acc:.2%})")

# Save results
results_path = Path(__file__).parent.parent / 'results' / 'cnn_vit_ffa_vs_bp.json'
results_path.parent.mkdir(parents=True, exist_ok=True)
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")

# Summary
print("\n" + "="*70)
print("SUMMARY: FFA vs BP on Modern Architectures")
print("="*70)
print(f"\n{'Architecture':<12} {'BP Acc':>10} {'FFA Acc':>10} {'Gap':>10}")
print("-" * 45)
for arch, r in results.items():
    print(f"{arch:<12} {r['bp_acc']:>9.2%} {r['ffa_acc']:>9.2%} {r['gap']:>9.2%}")

print("\nKey question: Can FFA compete with BP on CNN/ViT?")
print("Answer: See results above for quantitative comparison.")
