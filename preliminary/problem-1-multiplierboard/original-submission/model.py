import torch
import torch.nn as nn

# ==========================================
# Solution for KRAFTON AI R&D Hackathon
# MultiplierBoard: 6-bit Binary Multiplication
# ==========================================

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=24):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MultiplyTransformer(nn.Module):
    def __init__(self, vocab_size=2, d_model=16, nhead=2, num_layers=2):
        """
        Extremely minimal Transformer Architecture capable of learning 6-bit multiplication.
        By utilizing Fixed Sinusoidal Positional Encoding and Weight Tying, we save parameters, requiring only 6,594 parameters.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        # Fixed Sinusoidal PE (0 Trainable Parameters per Hackathon Rules)
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=24)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            batch_first=True,
            norm_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.fc_out = nn.Linear(self.d_model, self.vocab_size)
        
        # Apply Weight Tying to save parameters
        self.fc_out.weight = self.embedding.weight
        
    def forward(self, x):
        seq_len = x.size(1)
        # Causal mask ensures autoregressive token generation
        src_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        emb = self.pos_encoder(self.embedding(x))
        out = self.transformer_encoder(emb, mask=src_mask, is_causal=True)
        logits = self.fc_out(out)
        return logits

def build_model():
    """
    Returns the untrained nn.Module for Problem 1-2.
    Evaluated by the fixed training protocol (100k random pairs, AdamW, 200 epochs).
    """
    return MultiplyTransformer(vocab_size=2, d_model=16, nhead=2, num_layers=2)

from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# Reproducibility / Training Protocol
# ==========================================

def generate_data(num_samples):
    a = torch.randint(0, 64, (num_samples,))
    b = torch.randint(0, 64, (num_samples,))
    p = a * b
    a_bits = (a.unsqueeze(1) >> torch.arange(6)) & 1
    b_bits = (b.unsqueeze(1) >> torch.arange(6)) & 1
    p_bits = (p.unsqueeze(1) >> torch.arange(12)) & 1
    return torch.cat([a_bits, b_bits, p_bits], dim=1)

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            seq = batch[0].to(device)
            current_seq = seq[:, :12]
            for i in range(12):
                logits = model(current_seq)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                current_seq = torch.cat([current_seq, next_token], dim=1)
            
            predicted_p = current_seq[:, 12:24]
            true_p = seq[:, 12:24]
            correct += (predicted_p == true_p).all(dim=1).sum().item()
            total += seq.size(0)
    return correct / total

def reproduce_results(epochs=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_seq = generate_data(100000)
    train_loader = DataLoader(TensorDataset(train_seq), batch_size=256, shuffle=True)
    test_seq = generate_data(10000)
    test_loader = DataLoader(TensorDataset(test_seq), batch_size=256, shuffle=False)
    
    model = build_model().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Unique Parameters (P_2): {param_count}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            seq = batch[0].to(device)
            optimizer.zero_grad()
            logits = model(seq[:, :-1]) 
            loss_logits = logits[:, 11:23, :].reshape(-1, 2)
            loss_targets = seq[:, 12:24].reshape(-1)
            loss = criterion(loss_logits, loss_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == epochs:
            acc = evaluate(model, test_loader, device)
            print(f"Epoch {epoch:03d}/{epochs} - Loss: {total_loss/len(train_loader):.4f} - Test Acc: {acc:.4f}")

if __name__ == '__main__':
    reproduce_results()
