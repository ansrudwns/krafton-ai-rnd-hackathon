import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import build_model
import argparse

def build_custom_model(d_model, nhead, num_layers):
    class CustomMultiplyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.vocab_size = 2
            self.d_model = d_model
            self.nhead = nhead
            self.num_layers = num_layers
            
            self.embedding = nn.Embedding(self.vocab_size, self.d_model)
            self.pos_emb = nn.Parameter(torch.randn(1, 24, self.d_model) * 0.02)
            
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
            
        def forward(self, x):
            seq_len = x.size(1)
            src_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
            
            emb = self.embedding(x) + self.pos_emb[:, :seq_len, :]
            out = self.transformer_encoder(emb, mask=src_mask, is_causal=True)
            logits = self.fc_out(out)
            return logits
    return CustomMultiplyTransformer()

def generate_data(num_samples):
    # a, b in [0, 63]
    a = torch.randint(0, 64, (num_samples,))
    b = torch.randint(0, 64, (num_samples,))
    p = a * b
    
    # 6-bit LSB-first
    a_bits = (a.unsqueeze(1) >> torch.arange(6)) & 1
    b_bits = (b.unsqueeze(1) >> torch.arange(6)) & 1
    
    # 12-bit LSB-first
    p_bits = (p.unsqueeze(1) >> torch.arange(12)) & 1
    
    # Concat all to form sequence of 24 tokens (A0..A5, B0..B5, P0..P11)
    seq = torch.cat([a_bits, b_bits, p_bits], dim=1)
    return seq

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            seq = batch[0].to(device)
            N = seq.size(0)
            
            # Start with 12 input tokens
            current_seq = seq[:, :12]
            
            # Generate 12 output tokens regressively
            for i in range(12):
                logits = model(current_seq) # (N, L, 2)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                current_seq = torch.cat([current_seq, next_token], dim=1)
            
            predicted_p = current_seq[:, 12:24]
            true_p = seq[:, 12:24]
            
            correct += (predicted_p == true_p).all(dim=1).sum().item()
            total += N
            
    return correct / total

def train_model(epochs=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 100,000 random pairs for training
    train_seq = generate_data(100000)
    train_dataset = TensorDataset(train_seq)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    # 10,000 pairs for testing metric
    test_seq = generate_data(10000)
    test_dataset = TensorDataset(test_seq)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    configs = [
        (16, 2, 2),
        (24, 2, 2),
        (32, 2, 2)
    ]
    
    for d_model, nhead, num_layers in configs:
        model = build_custom_model(d_model, nhead, num_layers).to(device)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--- Testing Config: d_model={d_model}, nhead={nhead}, layers={num_layers} | Params: {param_count} ---")
        
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
                if acc >= 0.99:
                    print("Reached 99% accuracy! Minimal config found.")
                    break

if __name__ == '__main__':
    train_model(50)
