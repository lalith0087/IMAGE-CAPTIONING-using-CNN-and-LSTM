"""Scores a trained checkpoint with BLEU-1..4 on the validation split that train.py held out.

The split is rebuilt exactly as in train.py (same seed, same 10%), so no training image is scored.
Note the same split also selected checkpoints/best.pth (lowest val loss), so treat the numbers as
validation scores rather than a fully untouched test set.

Usage:
    python scripts/evaluate.py                       # checkpoints/best.pth, full split
    python scripts/evaluate.py --limit 64            # quick smoke test
"""
import argparse
import json
import os
import sys
import textwrap

import torch
from nltk.translate.bleu_score import corpus_bleu
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.encoder import image_transform
from data_loader import IMAGE_DIR, VOCAB_FILE, load_captions
from vocabulary import END, START, Vocabulary, tokenize
from predict import load_models

BLEU_WEIGHTS = {"bleu1": (1, 0, 0, 0), "bleu2": (.5, .5, 0, 0),
                "bleu3": (1 / 3, 1 / 3, 1 / 3, 0), "bleu4": (.25, .25, .25, .25)}


def val_indices(n_images, val_split=0.1, seed=42):
    """Indices train.py put in its validation set: random_split over the sorted image list."""
    val_size = int(n_images * val_split)
    generator = torch.Generator().manual_seed(seed)
    return list(random_split(range(n_images), [n_images - val_size, val_size], generator=generator)[1].indices)


def heldout_images(val_split=0.1, seed=42):
    """Filenames in the validation split (same filtering as CaptionDataset: image must exist)."""
    files = sorted(f for f in load_captions() if os.path.exists(os.path.join(IMAGE_DIR, f)))
    return [files[i] for i in val_indices(len(files), val_split, seed)]


@torch.no_grad()
def greedy_batch(decoder, image_embeds, start_idx, end_idx, max_len=20):
    """Batched version of DecoderRNN.sample (same input layout, same greedy rule)."""
    batch, device = image_embeds.size(0), image_embeds.device
    _, states = decoder.lstm(image_embeds.unsqueeze(1))
    tokens = torch.full((batch,), start_idx, dtype=torch.long, device=device)
    steps = []
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    for _ in range(max_len):
        hidden, states = decoder.lstm(decoder.embed(tokens).unsqueeze(1), states)
        tokens = decoder.linear(hidden.squeeze(1)).argmax(dim=1)
        steps.append(tokens)
        finished |= tokens == end_idx
        if finished.all():
            break
    grid = torch.stack(steps, dim=1).tolist()
    return [row[:row.index(end_idx)] if end_idx in row else row for row in grid]


def compute_bleu(references, hypotheses):
    """references: per image, a list of token lists. hypotheses: per image, one token list."""
    return {name: corpus_bleu(references, hypotheses, weights=w) for name, w in BLEU_WEIGHTS.items()}


def distinct_pct(captions):
    return 100.0 * len(set(captions)) / max(len(captions), 1)


class ImageFiles(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return image_transform(Image.open(os.path.join(IMAGE_DIR, self.files[i])).convert("RGB"))


def sample_grid(files, preds, refs, path, columns=4, cell=(330, 330)):
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    rows = (len(files) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell[0], rows * cell[1]), "white")
    draw = ImageDraw.Draw(canvas)
    for k, (f, pred, ref) in enumerate(zip(files, preds, refs)):
        x, y = (k % columns) * cell[0], (k // columns) * cell[1]
        img = Image.open(os.path.join(IMAGE_DIR, f)).convert("RGB")
        img.thumbnail((cell[0] - 20, 200))
        canvas.paste(img, (x + 10, y + 8))
        lines = textwrap.wrap("model: " + pred, 44) + [""] + textwrap.wrap("human: " + ref, 44)
        draw.text((x + 10, y + 215), "\n".join(lines), fill="black", font=font)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best.pth"))
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=20)
    parser.add_argument("--limit", type=int, help="score only the first N held-out images (smoke test)")
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    vocab = Vocabulary.load(VOCAB_FILE)
    encoder, decoder = load_models(args.checkpoint, vocab, device)
    epoch = torch.load(args.checkpoint, map_location="cpu").get("epoch")

    files = heldout_images(args.val_split, args.seed)[: args.limit]
    captions = load_captions()
    print(f"Scoring {len(files)} held-out images on {device} (checkpoint epoch {epoch})")

    preds = []
    loader = DataLoader(ImageFiles(files), batch_size=args.batch_size, num_workers=args.workers)
    with torch.no_grad():
        for batch in loader:
            ids = greedy_batch(decoder, encoder(batch.to(device)), vocab.stoi[START], vocab.stoi[END], args.max_len)
            preds += [vocab.decode(row) for row in ids]
    refs = [[tokenize(c) for c in captions[f]] for f in files]
    bleu = compute_bleu(refs, [p.split() for p in preds])

    os.makedirs(args.out_dir, exist_ok=True)
    metrics = {**{k: round(v, 4) for k, v in bleu.items()}, "n_images": len(files), "checkpoint": os.path.basename(args.checkpoint),
               "epoch": epoch, "decoding": f"greedy, max_len={args.max_len}", "split": f"seed={args.seed}, val_split={args.val_split}",
               "avg_caption_words": round(sum(len(p.split()) for p in preds) / max(len(preds), 1), 2),
               "distinct_captions_pct": round(distinct_pct(preds), 1)}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out_dir, "captions.json"), "w") as f:
        json.dump([{"image": fn, "model": p, "human": captions[fn][0]} for fn, p in zip(files, preds)], f, indent=1)
    pick = [int(i * len(files) / args.samples) for i in range(min(args.samples, len(files)))]
    sample_grid([files[i] for i in pick], [preds[i] for i in pick], [captions[files[i]][0] for i in pick],
                os.path.join(args.out_dir, "samples.png"))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
