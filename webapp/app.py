"""Minimal Flask demo: upload an image, get a generated caption back."""
import os
import sys
import uuid

import torch
from flask import Flask, render_template, request

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from predict import load_models, generate_caption
from data_loader import VOCAB_FILE
from vocabulary import Vocabulary

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "static", "uploads")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best.pth")

app = Flask(__name__)
os.makedirs(UPLOAD_DIR, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
vocab = Vocabulary.load(VOCAB_FILE)
encoder, decoder = load_models(CHECKPOINT_PATH, vocab, device)


@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    image_url = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")
        if uploaded_file and uploaded_file.filename:
            ext = os.path.splitext(uploaded_file.filename)[1].lower()
            if ext not in (".jpg", ".jpeg", ".png"):
                return render_template("index.html", caption="Unsupported file type.", image_url=None)

            filename = f"{uuid.uuid4().hex}{ext}"
            save_path = os.path.join(UPLOAD_DIR, filename)
            uploaded_file.save(save_path)

            caption = generate_caption(save_path, encoder, decoder, vocab, device)
            image_url = f"static/uploads/{filename}"

    return render_template("index.html", caption=caption, image_url=image_url)


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
