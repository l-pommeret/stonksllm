import numpy as np
from typing import List, Tuple, Union

class PriceChangeTokenizer:
    def __init__(
        self,
        bucket_size: float = 0.002,
        min_pct: float = -0.5,
        max_pct: float = 0.5
    ):
        """
        Initialise le tokenizer avec des buckets de taille fixe
        
        Args:
            bucket_size: Taille d'un bucket en pourcentage 
            min_pct: Pourcentage minimum avant token spécial 
            max_pct: Pourcentage maximum avant token spécial 
        """
        self.bucket_size = bucket_size
        self.min_pct = min_pct
        self.max_pct = max_pct
        
        # Créer les buckets
        self.buckets = np.arange(
            min_pct,
            max_pct + bucket_size,
            bucket_size
        )
        
        # Tokens spéciaux
        self.BELOW_MIN_TOKEN = 0  # Pour les valeurs < min_pct
        self.ABOVE_MAX_TOKEN = len(self.buckets) + 1  # Pour les valeurs > max_pct
        self.vocab_size = len(self.buckets) + 2  # +2 pour les tokens spéciaux
        
        # Créer le mapping inverse pour le decodage
        self.token_to_pct = {
            self.BELOW_MIN_TOKEN: f"<{min_pct}%",
            self.ABOVE_MAX_TOKEN: f">{max_pct}%"
        }
        for i, bucket_start in enumerate(self.buckets[:-1], 1):
            bucket_end = bucket_start + bucket_size
            self.token_to_pct[i] = f"{bucket_start:.2f}% à {bucket_end:.2f}%"
    
    def encode(self, pct_change: float) -> int:
        """Convertit un pourcentage en token"""
        if pct_change < self.min_pct:
            return self.BELOW_MIN_TOKEN
        elif pct_change > self.max_pct:
            return self.ABOVE_MAX_TOKEN
        else:
            # Trouver le bucket approprié
            bucket_idx = np.digitize(pct_change, self.buckets) - 1
            return bucket_idx + 1  # +1 car 0 est réservé pour BELOW_MIN
    
    def encode_sequence(self, pct_changes: List[float]) -> List[int]:
        """Convertit une séquence de pourcentages en tokens"""
        return [self.encode(pct) for pct in pct_changes]
    
    def decode(self, token: int) -> str:
        """Convertit un token en sa représentation en pourcentage"""
        return self.token_to_pct[token]
    
    def decode_sequence(self, tokens: List[int]) -> List[str]:
        """Convertit une séquence de tokens en pourcentages"""
        return [self.decode(token) for token in tokens]

# Exemple d'utilisation
if __name__ == "__main__":
    tokenizer = PriceChangeTokenizer()
    
    # Test avec quelques valeurs
    test_values = [-6.0, -2.5, -0.03, 0.12, 3.7, 6.0]
    tokens = tokenizer.encode_sequence(test_values)
    decoded = tokenizer.decode_sequence(tokens)
    
    print(f"Taille du vocabulaire: {tokenizer.vocab_size}")
    for val, token, dec in zip(test_values, tokens, decoded):
        print(f"Valeur: {val:>6.2f}% → Token: {token:>3d} → Décodé: {dec}")