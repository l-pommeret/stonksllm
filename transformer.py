import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class FastTokenTransformer(nn.Module):
    def __init__(
        self,
        n_tokens=52,        # Taille du vocabulaire des tokens
        d_model=64,         # Réduit pour des données plus simples
        nhead=4,            # Réduit pour éviter le surapprentissage
        num_layers=2,       # Réduit pour la rapidité
        context_length=32,  # Réduit pour correspondre aux données de trading
        dropout=0.1
    ):
        super().__init__()
        
        self.context_length = context_length
        
        # Embedding des tokens
        self.token_embedding = nn.Embedding(n_tokens, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(context_length, d_model))
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Couche de sortie
        self.fc = nn.Linear(d_model, n_tokens)
        
    def forward(self, x):
        # x shape: (batch, context_length)
        
        # Embedding + position
        x = self.token_embedding(x)
        x = x + self.pos_embedding[:x.size(1), :]
        
        # Transformer
        x = self.transformer(x)
        
        # Prédiction sur le dernier token
        x = x[:, -1, :]
        x = self.fc(x)
        
        return F.log_softmax(x, dim=-1)

class PriceDataset(Dataset):
    def __init__(self, tokens, context_length=32):
        self.data = torch.tensor(tokens, dtype=torch.long)
        self.context_length = context_length
        
        # Vérification de la taille minimale
        if len(self.data) <= context_length:
            raise ValueError(f"Dataset trop petit ({len(self.data)} tokens) pour le context_length ({context_length})")
    
    def __len__(self):
        return max(0, len(self.data) - self.context_length)
        
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.context_length]
        y = self.data[idx + self.context_length]
        return x, y

class TradingPredictor:
    def __init__(self, model, tokenizer, context_length=32, device="cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.device = device
        self.context = []
    
    def update_and_predict(self, new_token):
        """Met à jour le contexte et prédit la distribution du prochain token"""
        self.context.append(new_token)
        if len(self.context) > self.context_length:
            self.context = self.context[-self.context_length:]
        
        # Retourne une distribution uniforme si pas assez de contexte
        if len(self.context) < self.context_length:
            return torch.ones(self.tokenizer.vocab_size) / self.tokenizer.vocab_size
        
        # Prédiction
        with torch.no_grad():
            x = torch.tensor([self.context], dtype=torch.long).to(self.device)
            pred = self.model(x)
            return torch.exp(pred[0].cpu())  # Conversion log_softmax -> probabilités
    
    def get_trading_signal(self, pred_distribution, threshold=0.1):
        """Génère un signal de trading basé sur la distribution prédite"""
        # Trouver le token le plus probable
        max_prob_token = torch.argmax(pred_distribution).item()
        max_prob = pred_distribution[max_prob_token].item()
        
        # Si la probabilité n'est pas assez forte, reste neutre
        if max_prob < threshold:
            return 0
            
        # Cas spéciaux pour les tokens min/max
        if max_prob_token == self.tokenizer.BELOW_MIN_TOKEN:
            return -1
        elif max_prob_token == self.tokenizer.ABOVE_MAX_TOKEN:
            return 1
            
        # Pour les autres tokens, on regarde dans quel bucket ils tombent
        bucket_start = self.tokenizer.buckets[max_prob_token - 1]  # -1 car les indices commencent à 1
        
        # Décision basée sur la valeur du bucket
        if bucket_start > 0.001:  # Seuil pour position longue
            return 1
        elif bucket_start < -0.001:  # Seuil pour position courte
            return -1
        return 0  # Position neutre

def train_model(train_tokens, test_tokens, context_length=32, batch_size=16, n_epochs=50, device="cpu"):
    # Création des datasets
    try:
        train_dataset = PriceDataset(train_tokens, context_length)
        test_dataset = PriceDataset(test_tokens, context_length)
    except ValueError as e:
        print(f"Erreur lors de la création des datasets: {e}")
        return None
    
    # DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Modèle
    model = FastTokenTransformer(context_length=context_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    best_loss = float('inf')
    
    for epoch in range(n_epochs):
        # Entraînement
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = F.nll_loss(output, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_dataloader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                val_loss += F.nll_loss(output, batch_y).item()
                
                pred = output.argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)
        
        train_loss /= len(train_dataloader)
        val_loss /= len(test_dataloader)
        accuracy = correct / total if total > 0 else 0
        
        # Mise à jour du scheduler
        scheduler.step(val_loss)
        
        # Sauvegarde du meilleur modèle
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, 'best_model.pt')
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print("-" * 30)
    
    return model

if __name__ == "__main__":
    # Chargement des données
    print("Chargement des données...")
    data = np.load('dataset.npy', allow_pickle=True).item()
    tokens = data['tokens']
    
    # Vérification de la taille des données
    print(f"Nombre total de tokens: {len(tokens)}")
    
    # Paramètres adaptés aux données de trading
    context_length = 32  # Réduit pour correspondre à la quantité de données
    
    # Division train/test (80/20)
    train_size = int(0.8 * len(tokens))
    train_tokens = tokens[:train_size]
    test_tokens = tokens[train_size:]
    
    print(f"Tokens d'entraînement: {len(train_tokens)}")
    print(f"Tokens de test: {len(test_tokens)}")
    
    # Entraînement
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation de: {device}")
    
    model = train_model(train_tokens, test_tokens, 
                       context_length=context_length,
                       device=device)
    
    if model is not None:
        print("Entraînement terminé avec succès!")
    else:
        print("Erreur pendant l'entraînement")