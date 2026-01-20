###!pip install huggingface_hub --quiet
from huggingface_hub import login

login("your_key")

###!pip install wandb weave --quiet
import wandb
wandb.login("your_key")


from datasets import load_dataset

dataset = load_dataset("MRR24/English_to_Telugu_Bilingual_Sentence_Pairs")

split = dataset["train"].train_test_split(test_size=0.1, seed=42)

train_ds = split["train"]
test_ds  = split["test"]


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
###!pip install indic-nlp-library
from indicnlp.tokenize import indic_tokenize
from nltk.translate.bleu_score import corpus_bleu

def build_vocab(sentences, max_vocab=20_000):
    counter = Counter()
    for s in sentences:
        counter.update(s.split())
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    for word, _ in counter.most_common(max_vocab - len(vocab)):
        vocab[word] = len(vocab)
    return vocab

eng_vocab = build_vocab(train_ds["Input"])
tel_vocab = build_vocab(train_ds["Output"])

def encode(sentence, vocab, max_len):
    tokens = sentence.split()
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    ids = [vocab["<sos>"]] + ids[:max_len-2] + [vocab["<eos>"]]
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return torch.tensor(ids)


class TranslationDataset(Dataset):
    def __init__(self, ds, src_vocab, tgt_vocab, max_len):
        self.ds = ds
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src = encode(self.ds[idx]["Input"], self.src_vocab, self.max_len)
        tgt = encode(self.ds[idx]["Output"], self.tgt_vocab, self.max_len)
        return src, tgt

MAX_LEN = 40
BATCH_SIZE = 64

train_loader = DataLoader(
    TranslationDataset(train_ds, eng_vocab, tel_vocab, MAX_LEN),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    TranslationDataset(test_ds, eng_vocab, tel_vocab, MAX_LEN),
    batch_size=BATCH_SIZE
)

class MultiHeadedAttention(nn.Module):
  def __init__(self, d_model, n_heads):
    super(MultiHeadedAttention, self).__init__()
    assert d_model % n_heads == 0, "dimensions should be divisible by num heads"
    self.d_model = d_model
    self.n_heads = n_heads
    self.d_k = d_model // n_heads # Dimension of each head's key, query, and value


    self.W_q = nn.Linear(d_model, d_model) # Query transformation
    self.W_k = nn.Linear(d_model, d_model) # Key transformation
    self.W_v = nn.Linear(d_model, d_model) # Value transformation
    self.W_o = nn.Linear(d_model, d_model) # Output transformation

  def scaled_dot_product_attention(self, Q,K,V, mask = None):
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

    if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
    attn_probs = torch.softmax(attn_scores, dim= 1)

    output = torch.matmul(attn_probs, V)

    return output

  def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)

  def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

  def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.layer = nn.Sequential(nn.Linear(d_model, d_ff),
                                   nn.Linear(d_ff, d_model),
                                   nn.ReLU())
    def forward(self, x):
      return self.layer(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]



class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x



class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(d_model, num_heads)
        self.cross_attn = MultiHeadedAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x



class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        # Ensure nopeak_mask is on the same device as tgt
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=tgt.device), diagonal=1)).bool()

        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output



from nltk.translate.bleu_score import SmoothingFunction
smooth_fn = SmoothingFunction().method4

def decode_ids(ids, vocab):
    """
    Convert token IDs to a sentence string.
    Applies EOS filtering and removes special tokens.
    """
    inv_vocab = {v: k for k, v in vocab.items()}
    words = []

    for i in ids:
        token = inv_vocab.get(i, "<unk>")
        if token == "<eos>":
            break
        if token not in {"<pad>", "<sos>"}:
            words.append(token)

    return " ".join(words)



device = "cuda" if torch.cuda.is_available() else "cpu"

model = Transformer(
    src_vocab_size=len(eng_vocab),
    tgt_vocab_size=len(tel_vocab),
    d_model=256,
    num_heads=8,
    num_layers=4,
    d_ff=512,
    max_seq_length=MAX_LEN,
    dropout=0.1
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)



def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    global global_step # Declare global_step as global
    model.train()
    total_loss = 0

    for step, (src, tgt) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()

        output = model(src, tgt[:, :-1])
        loss = criterion(
            output.reshape(-1, output.size(-1)),
            tgt[:, 1:].reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        global_step += 1

        if step % 100 == 0:
            wandb.log({"train/batch_loss": loss.item()}, step=global_step)
            print(f"Epoch {epoch+1}, Batch {step+1}/{len(dataloader)}, Loss: {loss.item()}")


    avg_loss = total_loss / len(dataloader)

    # Log per epoch
    wandb.log(
    {"train/epoch_loss": avg_loss, "epoch": epoch +1}
)


    print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")

    return avg_loss



@torch.no_grad()
def eval_epoch(model, dataloader, criterion, device, src_vocab, tgt_vocab, epoch, prefix="test"):
    model.eval()
    total_loss = 0
    all_preds, all_refs = [], []

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        # Teacher-forced forward pass (UNCHANGED)
        output = model(src, tgt[:, :-1])
        loss = criterion(
            output.reshape(-1, output.size(-1)),
            tgt[:, 1:].reshape(-1)
        )
        total_loss += loss.item()

        # Greedy predictions (still teacher-forced context)
        pred_ids = output.argmax(dim=-1)

        for pred_seq, tgt_seq in zip(pred_ids, tgt[:, 1:]):
            # EOS-filtered decoding
            pred_sentence = decode_ids(pred_seq.cpu().tolist(), tgt_vocab)
            tgt_sentence  = decode_ids(tgt_seq.cpu().tolist(), tgt_vocab)

            # Telugu-friendly tokenization
            pred_tokens = indic_tokenize.trivial_tokenize(pred_sentence)
            tgt_tokens  = indic_tokenize.trivial_tokenize(tgt_sentence)

            # Skip empty sentences (important for BLEU stability)
            if len(pred_tokens) == 0 or len(tgt_tokens) == 0:
                continue

            all_preds.append(pred_tokens)
            all_refs.append([tgt_tokens])

    avg_loss = total_loss / len(dataloader)

    # BLEU with smoothing
    bleu = corpus_bleu(
        all_refs,
        all_preds,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smooth_fn
    )

    wandb.log({
        f"{prefix}/epoch_loss": avg_loss,
        f"{prefix}/epoch_BLEU": bleu,
        "epoch": epoch + 1
    })

    print(f"Epoch {epoch+1}, {prefix} Loss: {avg_loss:.4f}, BLEU: {bleu:.4f}")
    return avg_loss, bleu

wandb.init(
    project="english-to-telugu-transformer",
    config={
        "model": "Transformer",
        "dataset": "MRR24 English↔Telugu",
        "d_model": 256,
        "num_heads": 8,
        "num_layers": 4,
        "d_ff": 512,
        "dropout": 0.1,
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "batch_size": 64,
        "epochs": 5
    }
)


EPOCHS = 80
global_step = 0

for epoch in range(EPOCHS):
    train_loss = train_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        device,
        epoch
    )
    test_loss_avg, test_bleu = eval_epoch(
        model,
        test_loader,
        criterion,
        device,
        eng_vocab,
        tel_vocab,
        epoch,
        prefix="test"
    )

    print(f"Epoch {epoch+1}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Test  Loss: {test_loss_avg:.4f}")
    print(f"  Test  BLEU: {test_bleu:.4f}")


torch.save(model.state_dict(), "telugu_transformer.pth")

def inference_decode_ids(ids, vocab):
    inv_vocab = {v: k for k, v in vocab.items()}
    words = []

    for i in ids:
        if i == vocab["<eos>"]:
            break
        if i in (vocab["<pad>"], vocab["<sos>"]):
            continue
        words.append(inv_vocab.get(i, "<unk>"))

    return " ".join(words)

@torch.no_grad()
def translate_sentence(
    model,
    sentence,
    src_vocab,
    tgt_vocab,
    device,
    max_len=40
):
    model.eval()

    # Encode source
    src_ids = encode(sentence, src_vocab, max_len).unsqueeze(0).to(device)

    # Start with <sos>
    tgt_ids = torch.tensor([[tgt_vocab["<sos>"]]], device=device)

    for _ in range(max_len):
        output = model(src_ids, tgt_ids)
        next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(1)

        tgt_ids = torch.cat([tgt_ids, next_token], dim=1)

        if next_token.item() == tgt_vocab["<eos>"]:
            break

    # Decode (skip <sos>)
    return decode_ids(tgt_ids[0, 1:].cpu().tolist(), tgt_vocab)


sentence = "I like walking"
translation = translate_sentence(
    model,
    sentence,
    eng_vocab,
    tel_vocab,
    device
)

print("English :", sentence)
print("Telugu  :", translation)


###Optional autoregressive training

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    global global_step  # Declare global_step as global
    model.train()
    total_loss = 0

    for step, (src, tgt) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device);

        optimizer.zero_grad()

        # Autoregressive generation
        batch_size, tgt_len = tgt[:, :-1].size()
        outputs = torch.zeros(batch_size, tgt_len, model.fc.out_features, device=device)

        # Start token (usually index 0 or your SOS token)
        input_seq = tgt[:, :1]  # shape: (batch_size, 1)

        for t in range(tgt_len):
            out = model(src, input_seq)  # forward pass with generated tokens so far
            next_token_logits = out[:, -1, :]  # logits for the last timestep
            outputs[:, t, :] = next_token_logits

            # Greedy next token
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            input_seq = torch.cat([input_seq, next_token], dim=1)  # append to input_seq

        # Compute loss
        loss = criterion(
            outputs.reshape(-1, outputs.size(-1)),
            tgt[:, 1:].reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        global_step += 1

        if step % 100 == 0:
            wandb.log({"train/batch_loss": loss.item()}, step=global_step)
            print(f"Epoch {epoch+1}, Batch {step+1}/{len(dataloader)}, Loss: {loss.item()}")

    avg_loss = total_loss / len(dataloader)

    # Log per epoch
    wandb.log(
        {"train/epoch_loss": avg_loss, "epoch": epoch + 1}
    )

    print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")

    return avg_loss

@torch.no_grad()
def eval_epoch(model, dataloader, criterion, device, src_vocab, tgt_vocab, epoch, prefix="test"):
    model.eval()
    total_loss = 0
    all_preds, all_refs = [], []

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        # Autoregressive generation
        batch_size, tgt_len = tgt[:, :-1].size()
        outputs = torch.zeros(batch_size, tgt_len, model.fc.out_features, device=device)

        # Start token (usually index 0 or your specific SOS token)
        input_seq = tgt[:, :1]  # shape: (batch_size, 1)

        for t in range(tgt_len):
            out = model(src, input_seq)  # forward pass with generated tokens so far
            next_token_logits = out[:, -1, :]  # logits for the last timestep
            outputs[:, t, :] = next_token_logits

            # Greedy next token
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            input_seq = torch.cat([input_seq, next_token], dim=1)  # append to input_seq

        # Compute loss
        loss = criterion(
            outputs.reshape(-1, outputs.size(-1)),
            tgt[:, 1:].reshape(-1)
        )
        total_loss += loss.item()

        # Greedy predictions
        pred_ids = outputs.argmax(dim=-1)
        for pred_seq, tgt_seq in zip(pred_ids, tgt[:, 1:]):
            pred_sentence = decode_ids(pred_seq.cpu().tolist(), tgt_vocab)
            tgt_sentence  = decode_ids(tgt_seq.cpu().tolist(), tgt_vocab)

            pred_tokens = indic_tokenize.trivial_tokenize(pred_sentence)
            tgt_tokens  = indic_tokenize.trivial_tokenize(tgt_sentence)

            if len(pred_tokens) == 0 or len(tgt_tokens) == 0:
                continue

            all_preds.append(pred_tokens)
            all_refs.append([tgt_tokens])

    avg_loss = total_loss / len(dataloader)

    bleu = corpus_bleu(
        all_refs,
        all_preds,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smooth_fn
    )

    wandb.log({
        f"{prefix}/epoch_loss": avg_loss,
        f"{prefix}/epoch_BLEU": bleu,
        "epoch": epoch + 1
    })

    print(f"Epoch {epoch+1}, {prefix} Loss: {avg_loss:.4f}, BLEU: {bleu:.4f}")
    return avg_loss, bleu


wandb.init(
    project="english-to-telugu-transformer",
    config={
        "model": "Transformer",
        "dataset": "MRR24 English↔Telugu",
        "d_model": 256,
        "num_heads": 8,
        "num_layers": 4,
        "d_ff": 512,
        "dropout": 0.1,
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "batch_size": 64,
        "epochs": 5
    }
)

EPOCHS = 80
global_step = 0

for epoch in range(EPOCHS):
    train_loss = train_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        device,
        epoch
    )
    test_loss_avg, test_bleu = eval_epoch(
        model,
        test_loader,
        criterion,
        device,
        eng_vocab,
        tel_vocab,
        epoch,
        prefix="test"
    )

    print(f"Epoch {epoch+1}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Test  Loss: {test_loss_avg:.4f}")
    print(f"  Test  BLEU: {test_bleu:.4f}")



