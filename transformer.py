import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from dataset import tokenizer

class FastTokenTransformer(nn.Module):
    def __init__(
        self,
        n_tokens=203,        # Taille du vocabulaire de notre tokenizer
        d_model=64,         # Dimension du modèle (petite pour la rapidité)
        nhead=4,            # Nombre de têtes d'attention
        num_layers=2,       # Nombre de couches (minimal)
        context_length=64,  # Fenêtre d'observation (16 secondes)
        dropout=0.1
    ):
        super().__init__()
        
        self.context_length = context_length
        
        # Embedding des tokens
        self.token_embedding = nn.Embedding(n_tokens, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(context_length, d_model))
        
        # Encoder léger
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,  # Réduit pour la rapidité
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Prédiction de la distribution des prochains tokens
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, n_tokens)
        
    def forward(self, x):
        # x shape: (batch, context_length)
        
        # Embedding + position
        x = self.token_embedding(x)
        x = x + self.pos_embedding[:x.size(1), :]
        
        # Transformer
        x = self.transformer(x)
        
        # On ne prend que le dernier token pour la prédiction
        x = x[:, -1, :]
        
        # MLP final
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Distribution de probabilité sur les tokens
        return F.softmax(x, dim=-1)

class PriceDataset(Dataset):
    def __init__(self, tokens, context_length=64):
        self.data = torch.tensor(tokens, dtype=torch.long)
        self.context_length = context_length
        
    def __len__(self):
        return len(self.data) - self.context_length - 1
        
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.context_length]
        y = self.data[idx + self.context_length]
        return x, y

def evaluate_model(model, dataloader, device):
    """Évalue le modèle sur un jeu de données"""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            y_pred = model(batch_x)
            loss = F.cross_entropy(y_pred, batch_y)
            total_loss += loss.item()
            
            # Calcul de l'exactitude
            pred_tokens = torch.argmax(y_pred, dim=1)
            correct_predictions += (pred_tokens == batch_y).sum().item()
            total_predictions += batch_y.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy

def train_model(train_tokens, test_tokens, context_length=64, batch_size=64, n_epochs=50, device="cpu"):
    # Préparation des données
    train_dataset = PriceDataset(train_tokens, context_length)
    test_dataset = PriceDataset(test_tokens, context_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Modèle
    model = FastTokenTransformer(context_length=context_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # Entraînement
    best_loss = float('inf')
    best_test_accuracy = 0
    
    for epoch in range(n_epochs):
        # Mode entraînement
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            y_pred = model(batch_x)
            loss = F.cross_entropy(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calcul des métriques d'entraînement
        avg_train_loss = total_loss / len(train_dataloader)
        
        # Évaluation sur le jeu de test
        test_loss, test_accuracy = evaluate_model(model, test_dataloader, device)
        
        # Mise à jour du scheduler
        scheduler.step(test_loss)
        
        # Sauvegarde du meilleur modèle basé sur la précision de test
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy': test_accuracy,
                'test_loss': test_loss,
            }, 'best_model.pt')
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            print("-" * 50)
    
    return model

class TradingPredictor:
    def __init__(self, model, tokenizer, context_length=64, device="cpu"):
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
        
        # Si pas assez de contexte, retourne une distribution uniforme
        if len(self.context) < self.context_length:
            return torch.ones(203) / 203
        
        # Prédiction
        with torch.no_grad():
            x = torch.tensor([self.context], dtype=torch.long).to(self.device)
            pred = self.model(x)
            return pred[0].cpu()
    
    def get_trading_signal(self, pred_distribution, neutral_range=(-2.0, 2.0)):
        """
        Signal basé sur la concentration de la distribution prédite
        neutral_range: tuple (min_pct, max_pct) définissant la zone neutre en pourcentage
        """
        # Convertir les indices de tokens en pourcentages
        # Token 100 = 0%, chaque token = 0.05%
        def token_to_pct(token_idx):
            return (token_idx - 100) * 0.05

        # Trouver le token avec la plus haute probabilité
        max_prob_token = torch.argmax(pred_distribution).item()
        predicted_pct = token_to_pct(max_prob_token)

        # Décision basée sur où se concentre la distribution
        if neutral_range[0] <= predicted_pct <= neutral_range[1]:
            return 0  # Neutre
        elif predicted_pct > neutral_range[1]:
            return 1  # Long
        else:
            return -1  # Short

# Exemple d'utilisation
if __name__ == "__main__":
    # Chargement des données
    data = np.load('market_data_20241108_141411.npy', allow_pickle=True).item()
    tokens = data['tokens']
    
    # Séparation des données en train/test (90/10)
    train_size = int(0.9 * len(tokens))
    train_tokens = tokens[:train_size]
    test_tokens = tokens[train_size:]
    
    # Entraînement avec validation sur le jeu de test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_model(train_tokens, test_tokens, device=device)
    
    # Chargement du meilleur modèle
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Meilleure précision sur le test: {checkpoint['test_accuracy']:.4f}")
    
    # Création du prédicteur avec le meilleur modèle
    predictor = TradingPredictor(model, tokenizer, device=device)
    
    # Test sur les dernières données
    print("\nTest de prédiction sur les dernières données:")
    for token in test_tokens[-35:-1]:
        distribution = predictor.update_and_predict(token)
        signal = predictor.get_trading_signal(distribution)
        print(f"Token actuel: {token} ({tokenizer.decode(token)})")
        print(f"Signal: {'Long' if signal == 1 else 'Short' if signal == -1 else 'Neutre'}")