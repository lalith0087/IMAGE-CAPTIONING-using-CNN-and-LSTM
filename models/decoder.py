"""LSTM decoder that generates a caption token-by-token, conditioned on an
image embedding fed in as the first timestep."""
import torch
import torch.nn as nn


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, image_features, captions):
        """Teacher-forced training pass over a padded batch.
        image_features: (batch, embed_size)
        captions: (batch, seq_len) token ids, including <start>. The image
                  feature is fed as the first timestep, and predictions are
                  compared against `captions` itself (shifted by one via the
                  extra image step), so the model learns image -> <start> ->
                  w1 -> ... -> <end>.
        Returns logits of shape (batch, seq_len, vocab_size).
        """
        embeddings = self.embed(captions[:, :-1])
        inputs = torch.cat((image_features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(inputs)
        return self.linear(hiddens)

    def sample(self, image_features, start_idx, end_idx, max_len=20):
        """Greedy decoding for a single image at inference time. Mirrors the
        training input layout: image feature first, then <start>, then each
        predicted token fed back in."""
        sampled_ids = []
        states = None
        inputs = image_features.unsqueeze(1)  # (1, 1, embed_size)
        _, states = self.lstm(inputs, states)

        next_token = torch.tensor([start_idx], device=image_features.device)
        for _ in range(max_len):
            inputs = self.embed(next_token).unsqueeze(1)
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            next_token = outputs.argmax(dim=1)
            if next_token.item() == end_idx:
                break
            sampled_ids.append(next_token.item())
        return sampled_ids
