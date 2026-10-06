import torch
from torch.utils.data import random_split

from evaluate import compute_bleu, distinct_pct, greedy_batch, val_indices
from models.decoder import DecoderRNN

START, END = 1, 2


def make_decoder(seed=0):
    torch.manual_seed(seed)
    return DecoderRNN(embed_size=8, hidden_size=16, vocab_size=12).eval()


def test_batched_greedy_matches_the_single_image_sampler():
    decoder = make_decoder()
    embeds = torch.randn(6, 8)
    batched = greedy_batch(decoder, embeds, START, END, max_len=10)
    single = [decoder.sample(embeds[i:i + 1], START, END, max_len=10) for i in range(6)]
    assert batched == single


def test_batched_greedy_stops_at_end_token():
    decoder = make_decoder()
    with torch.no_grad():
        decoder.linear.bias[END] = 100.0                        # make <end> the most likely word
    assert greedy_batch(decoder, torch.randn(3, 8), START, END, max_len=10) == [[], [], []]


def test_val_split_matches_what_train_py_holds_out():
    n = 8091
    expected = random_split(list(range(n)), [n - int(n * 0.1), int(n * 0.1)],
                            generator=torch.Generator().manual_seed(42))[1].indices
    ours = val_indices(n)
    assert ours == list(expected) and len(ours) == 809
    assert val_indices(n, seed=7) != ours                       # seed matters


def test_bleu_perfect_and_disjoint():
    refs = [[["a", "dog", "runs", "in", "the", "park"]], [["two", "kids", "play", "with", "a", "red", "ball"]]]
    perfect = compute_bleu(refs, [r[0] for r in refs])
    assert all(abs(v - 1.0) < 1e-9 for v in perfect.values())
    nothing = compute_bleu(refs, [["x"] * 6, ["y"] * 7])
    assert all(v == 0 for v in nothing.values())


def test_distinct_captions_flags_mode_collapse():
    assert distinct_pct(["a dog"] * 10) == 10.0
    assert distinct_pct(["a", "b", "c", "d"]) == 100.0


def test_fingerprint_depends_on_the_set_not_the_order():
    from evaluate import fingerprint
    assert fingerprint(["b.jpg", "a.jpg"]) == fingerprint(["a.jpg", "b.jpg"])
    assert fingerprint(["a.jpg", "b.jpg"]) != fingerprint(["a.jpg", "c.jpg"])
    assert len(fingerprint(["a.jpg"])) == 16


def test_verify_split_rejects_a_different_dataset_copy():
    import pytest
    from evaluate import verify_split
    verify_split("abc", "abc")                                  # match: fine
    verify_split("abc", None)                                   # nothing expected: fine
    with pytest.raises(ValueError, match="different train/val split"):
        verify_split("abc", "xyz")
