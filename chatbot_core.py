import re, json
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
MAX_LEN = 15          # max tokens per sentence for Cornell pairs


def normalize(text):
    """Lowercase, drop emojis / weird symbols, split punctuation into tokens."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9.!?']+", " ", text)   # keep letters, digits, . ! ? and '
    text = re.sub(r"([.!?])\1+", r"\1", text)      # "..." -> "."
    text = re.sub(r"([.!?])", r" \1 ", text)        # "hi!" -> "hi !"
    return re.sub(r"\s+", " ", text).strip()


class Vocab:
    """word <-> integer mapping with <PAD>, <SOS>, <EOS>, <UNK>."""
    def __init__(self, words=None):
        self.itos = list(SPECIAL_TOKENS) + list(words or [])
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, sentence, add_eos=False):
        ids = [self.stoi.get(w, UNK) for w in sentence.split()]
        return ids + [EOS] if add_eos else ids

    def detokenize(self, ids):
        words = [self.itos[i] for i in ids if i not in (PAD, SOS, EOS)]
        text = " ".join(words)
        text = re.sub(r"\s+([.!?])", r"\1", text)
        text = re.sub(r"\bi\b", "I", text)
        text = re.sub(r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
        return text[:1].upper() + text[1:]

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.itos, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            itos = json.load(f)
        return cls(itos[len(SPECIAL_TOKENS):])


def make_mask(lengths, max_len):
    """True for real tokens, False for padding. Shape (B, S)."""
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


class Encoder(nn.Module):
    """The Listener: bidirectional GRU that compresses the question."""
    def __init__(self, vocab_size, emb_dim, hidden, n_layers=2, dropout=0.2):
        super().__init__()
        self.hidden, self.n_layers = hidden, n_layers
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(emb_dim, hidden, n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0, bidirectional=True)

    def forward(self, src, lengths):
        emb = self.drop(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        out, h = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        out = out[:, :, :self.hidden] + out[:, :, self.hidden:]      # (B, S, H) sum both directions
        h = h.view(self.n_layers, 2, -1, self.hidden)
        h = h[:, 0] + h[:, 1]                                        # (layers, B, H) = context vector
        return out, h


class LuongAttention(nn.Module):
    """Luong 'general' attention: score(h_t, h_s) = h_t^T W h_s."""
    def __init__(self, hidden):
        super().__init__()
        self.W = nn.Linear(hidden, hidden, bias=False)

    def forward(self, dec_h, enc_out, mask):
        scores = torch.bmm(enc_out, self.W(dec_h).unsqueeze(2)).squeeze(2)   # (B, S)
        scores = scores.masked_fill(~mask, -1e9)                              # ignore <PAD>
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)         # (B, H)
        return context, weights


class Decoder(nn.Module):
    """The Speaker: one GRU step + attention over the encoder outputs."""
    def __init__(self, vocab_size, emb_dim, hidden, n_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(emb_dim, hidden, n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0)
        self.attention = LuongAttention(hidden)
        self.concat = nn.Linear(2 * hidden, hidden)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, token, hidden, enc_out, mask):
        emb = self.drop(self.embedding(token)).unsqueeze(1)          # (B, 1, E)
        out, hidden = self.gru(emb, hidden)
        out = out.squeeze(1)                                         # (B, H)
        context, weights = self.attention(out, enc_out, mask)
        attn_h = torch.tanh(self.concat(torch.cat([out, context], dim=1)))
        logits = self.out(self.drop(attn_h))                         # (B, V)
        return logits, hidden, weights


@torch.no_grad()
def generate(encoder, decoder, vocab, sentence, device, max_len=25,
             method="greedy", temperature=0.8, top_k=10, return_attention=False):
    """Inference loop: encode -> feed <SOS> -> pick next word -> feed it back -> stop at <EOS>."""
    encoder.eval(); decoder.eval()
    text = normalize(sentence)
    ids = vocab.encode(text, add_eos=True)[-(2 * MAX_LEN):]
    if len(ids) == 1:                       # empty / only symbols
        ids = [UNK, EOS]
    src = torch.tensor([ids], device=device)
    lengths = torch.tensor([len(ids)], device=device)
    enc_out, hidden = encoder(src, lengths)
    mask = make_mask(lengths, src.size(1))

    token = torch.tensor([SOS], device=device)
    out_ids, attn = [], []
    for _ in range(max_len):
        logits, hidden, w = decoder(token, hidden, enc_out, mask)
        logits[:, [PAD, SOS, UNK]] = -float("inf")           # never emit these
        if method == "greedy":
            nxt = logits.argmax(dim=-1)                      # most probable word
        else:                                                # temperature + top-k sampling
            logits = logits / max(temperature, 1e-3)
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < kth, -float("inf"))
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(1)
        if nxt.item() == EOS:
            break
        out_ids.append(nxt.item()); attn.append(w[0].cpu().numpy())
        token = nxt                                          # feed prediction back in
    reply = vocab.detokenize(out_ids)
    if return_attention:
        src_toks = text.split() + ["<EOS>"]
        return reply, src_toks[-len(ids):], [vocab.itos[i] for i in out_ids], attn
    return reply


def load_chatbot(ckpt_path="chatbot.pt", vocab_path="vocab.json", device="cpu"):
    """Rebuild Encoder/Decoder from a saved checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    hp = ckpt["hparams"]
    vocab = Vocab.load(vocab_path)
    enc = Encoder(len(vocab), hp["emb_dim"], hp["hidden"], hp["n_layers"], hp["dropout"]).to(device)
    dec = Decoder(len(vocab), hp["emb_dim"], hp["hidden"], hp["n_layers"], hp["dropout"]).to(device)
    enc.load_state_dict(ckpt["encoder"]); dec.load_state_dict(ckpt["decoder"])
    enc.eval(); dec.eval()
    return enc, dec, vocab
