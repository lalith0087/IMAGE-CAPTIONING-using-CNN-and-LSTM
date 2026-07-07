# Image Captioning using CNN + LSTM

Generates natural-language captions for images using a pretrained CNN
(InceptionV3) encoder and an LSTM decoder, trained on the Flickr8k dataset.

## 1. Get the data

Download the Flickr8k dataset (images + captions) and place it like this:

```
data/
  Flickr8k_Dataset/        # all .jpg images
  Flickr8k.token.txt       # image_name#caption_id \t caption text
```

Common sources: Kaggle "Flickr8k" dataset, or the original Illinois request form.
Unzip so that `data/Flickr8k_Dataset/*.jpg` and `data/Flickr8k.token.txt` exist.

## 2. Install dependencies

```
pip install -r requirements.txt
```

## 3. Preprocess: build vocabulary

```
python scripts/build_vocab.py
```

This caches a `vocab.pkl` built from every caption's word frequencies.

## 4. Train

```
python scripts/train.py --epochs 20 --batch-size 32
```

The encoder fine-tunes the last Inception block (`Mixed_7a` onward) instead
of using frozen precomputed features, so every training step runs the full
CNN forward pass on raw images — slower per epoch than a frozen-feature
setup, but the visual features adapt to the captioning task and to whatever
images you train on. Use `--cnn-lr` to control the (lower) learning rate for
those fine-tuned CNN layers separately from `--lr` for the projection + LSTM.

Checkpoints are saved to `checkpoints/`.

## 5. Generate a caption for a new image

```
python scripts/predict.py --image path/to/image.jpg --checkpoint checkpoints/best.pth
```

## 6. Web demo

```
python webapp/app.py
```

Then open http://localhost:5000, upload an image, and see the generated caption.

## Architecture

- **Encoder**: pretrained InceptionV3 (ImageNet weights). Layers up to
  `Mixed_7a` stay frozen; `Mixed_7a` onward is fine-tuned so the CNN adapts
  to the captioning task. The 2048-d pooled output is projected to the
  embedding dimension.
- **Decoder**: LSTM that takes the image embedding as the first input, then
  generates the caption token-by-token, trained with teacher forcing and
  cross-entropy loss (padding tokens masked).
- **Inference**: greedy or beam-search decoding starting from `<start>`, ending
  at `<end>` or a max length.
