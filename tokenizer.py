import numpy as np
from typing import List, Dict, Union

class PriceChangeTokenizer:
    def __init__(
        self,
        bucket_size: float = 0.002,
        min_pct: float = -0.05,
        max_pct: float = 0.05
    ):
        """
        Initialise le tokenizer avec des buckets de taille fixe.
        
        Args:
            bucket_size: Taille d'un bucket en pourcentage (défaut: 0.2%)
            min_pct: Pourcentage minimum avant token spécial (défaut: -50%)
            max_pct: Pourcentage maximum avant token spécial (défaut: 50%)
        """
        # Validation des paramètres
        if bucket_size <= 0:
            raise ValueError("bucket_size doit être positif")
        if min_pct >= max_pct:
            raise ValueError("min_pct doit être inférieur à max_pct")
            
        self.bucket_size = bucket_size
        self.min_pct = min_pct
        self.max_pct = max_pct
        
        # Création des buckets
        self.buckets = np.arange(min_pct, max_pct + bucket_size, bucket_size)
        
        # Tokens spéciaux
        self.BELOW_MIN_TOKEN = 0
        self.ABOVE_MAX_TOKEN = len(self.buckets) + 1
        self.vocab_size = len(self.buckets) + 2
        
        # Création du mapping inverse pour le décodage
        self._create_token_mapping()
        
        # Validation finale
        self._validate_setup()
    
    def _create_token_mapping(self) -> None:
        """Crée le mapping token -> représentation en pourcentage"""
        self.token_to_pct = {
            self.BELOW_MIN_TOKEN: f"<{self.min_pct:.2f}%",
            self.ABOVE_MAX_TOKEN: f">{self.max_pct:.2f}%"
        }
        
        for i, bucket_start in enumerate(self.buckets[:-1], 1):
            bucket_end = bucket_start + self.bucket_size
            self.token_to_pct[i] = f"{bucket_start:.2f}% à {bucket_end:.2f}%"
    
    def _validate_setup(self) -> None:
        """Vérifie la cohérence du setup"""
        # Vérification de la couverture des buckets
        expected_buckets = int((self.max_pct - self.min_pct) / self.bucket_size) + 1
        actual_buckets = len(self.buckets)
        if expected_buckets != actual_buckets:
            raise ValueError(f"Incohérence dans le nombre de buckets: {actual_buckets} vs {expected_buckets}")
        
        # Vérification de la taille du vocabulaire
        if self.vocab_size != len(self.buckets) + 2:
            raise ValueError("Incohérence dans la taille du vocabulaire")
    
    def encode(self, pct_change: float) -> int:
        """
        Convertit un pourcentage en token.
        
        Args:
            pct_change: Variation en pourcentage à encoder
            
        Returns:
            int: Token correspondant
        """
        if not isinstance(pct_change, (int, float)):
            raise TypeError("pct_change doit être un nombre")
            
        if np.isnan(pct_change):
            raise ValueError("pct_change ne peut pas être NaN")
            
        if pct_change < self.min_pct:
            return self.BELOW_MIN_TOKEN
        elif pct_change > self.max_pct:
            return self.ABOVE_MAX_TOKEN
            
        bucket_idx = np.digitize(pct_change, self.buckets) - 1
        token = bucket_idx + 1  # +1 car 0 est réservé pour BELOW_MIN
        
        # Vérification de sécurité
        if not 0 <= token < self.vocab_size:
            raise ValueError(f"Token {token} invalide pour pct_change={pct_change}")
            
        return token
    
    def encode_sequence(self, pct_changes: List[float]) -> List[int]:
        """
        Convertit une séquence de pourcentages en tokens.
        
        Args:
            pct_changes: Liste des variations en pourcentage
            
        Returns:
            List[int]: Liste des tokens correspondants
        """
        return [self.encode(pct) for pct in pct_changes]
    
    def decode(self, token: int) -> str:
        """
        Convertit un token en sa représentation en pourcentage.
        
        Args:
            token: Token à décoder
            
        Returns:
            str: Représentation en pourcentage
        """
        if not isinstance(token, (int, np.integer)):
            raise TypeError("token doit être un entier")
            
        if token not in self.token_to_pct:
            raise ValueError(f"Token invalide: {token}")
            
        return self.token_to_pct[token]
    
    def decode_sequence(self, tokens: List[int]) -> List[str]:
        """
        Convertit une séquence de tokens en pourcentages.
        
        Args:
            tokens: Liste des tokens à décoder
            
        Returns:
            List[str]: Liste des représentations en pourcentage
        """
        return [self.decode(token) for token in tokens]
    
    def get_vocab_info(self) -> Dict:
        """
        Retourne les informations sur le vocabulaire.
        
        Returns:
            Dict: Informations sur le vocabulaire
        """
        return {
            "vocab_size": self.vocab_size,
            "min_pct": self.min_pct,
            "max_pct": self.max_pct,
            "bucket_size": self.bucket_size,
            "n_buckets": len(self.buckets),
            "special_tokens": {
                "BELOW_MIN": self.BELOW_MIN_TOKEN,
                "ABOVE_MAX": self.ABOVE_MAX_TOKEN
            }
        }

def test_tokenizer():
    """Fonction de test du tokenizer"""
    tokenizer = PriceChangeTokenizer()
    test_values = [-6.0, -2.5, -0.03, 0.12, 3.7, 6.0]
    
    print("Test du tokenizer:")
    print(f"Configuration: {tokenizer.get_vocab_info()}\n")
    
    tokens = tokenizer.encode_sequence(test_values)
    decoded = tokenizer.decode_sequence(tokens)
    
    print("Tests de tokenization:")
    for val, token, dec in zip(test_values, tokens, decoded):
        print(f"Valeur: {val:>6.2f}% → Token: {token:>3d} → Décodé: {dec}")

if __name__ == "__main__":
    test_tokenizer()