import ccxt.async_support as ccxt
import asyncio
import time
import torch
import numpy as np
from datetime import datetime
from tokenizer import PriceChangeTokenizer
from transformer import FastTokenTransformer, TradingPredictor

class MarketPredictor:
    def __init__(self, model_path='best_model.pt'):
        # Chargement du checkpoint
        self.checkpoint = torch.load(model_path)
        
        # Récupération des paramètres du modèle sauvegardé
        state_dict = self.checkpoint['model_state_dict']
        self.vocab_size = state_dict['token_embedding.weight'].shape[0]
        self.d_model = state_dict['token_embedding.weight'].shape[1]
        self.context_length = state_dict['pos_embedding'].shape[0]  # Utilise la taille du pos_embedding sauvegardé
        
        # Initialisation avec les mêmes paramètres que lors de l'entraînement
        self.model = FastTokenTransformer(
            n_tokens=self.vocab_size,
            d_model=self.d_model,
            nhead=4,  # Même valeur que dans l'entraînement
            num_layers=2,  # Même valeur que dans l'entraînement
            context_length=self.context_length,  # Utilise la même taille de contexte
            dropout=0.1
        )
        
        # Chargement des poids
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        # Configuration du device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        print(f"Utilisation de: {self.device}")
        
        # Initialisation du tokenizer et du predictor
        self.tokenizer = PriceChangeTokenizer(bucket_size=0.002, min_pct=-0.5, max_pct=0.5)
        self.predictor = TradingPredictor(
            self.model, 
            self.tokenizer, 
            context_length=self.context_length,  # Utilise la même taille de contexte
            device=self.device
        )
        
        # Initialisation de l'exchange
        self.exchange = ccxt.bitget()
        
        print(f"\nInitialisation complète:")
        print(f"Taille du vocabulaire: {self.vocab_size}")
        print(f"Dimension du modèle: {self.d_model}")
        print(f"Taille du contexte: {self.context_length}")
        print(f"Taille bucket: {self.tokenizer.bucket_size}%")
        print(f"Range tokens: {self.tokenizer.min_pct}% à {self.tokenizer.max_pct}%")
    
    def calculate_confidence(self, distribution):
        """Calcule le niveau de confiance de la prédiction"""
        # Calcul basé sur l'entropie normalisée
        entropy = -torch.sum(distribution * torch.log(distribution + 1e-9))
        max_entropy = np.log(len(distribution))
        normalized_confidence = 1 - (entropy / max_entropy)
        return normalized_confidence.item()
    
    def evaluate_prediction(self, distribution, actual_change, tolerance=None):
        """Évalue la précision de la prédiction"""
        if tolerance is None:
            tolerance = self.tokenizer.bucket_size
        
        # Token avec la plus haute probabilité
        predicted_token = torch.argmax(distribution).item()
        
        # Conversion en pourcentage
        if predicted_token == 0:
            predicted_pct = self.tokenizer.min_pct
        elif predicted_token == self.vocab_size - 1:
            predicted_pct = self.tokenizer.max_pct
        else:
            predicted_pct = self.tokenizer.buckets[predicted_token - 1]
        
        # Calcul des métriques
        error = abs(predicted_pct - actual_change)
        exact_match = error <= tolerance
        direction_match = (predicted_pct * actual_change) > 0
        
        return {
            'predicted_pct': predicted_pct,
            'actual_pct': actual_change,
            'error': error,
            'exact_match': exact_match,
            'direction_match': direction_match
        }
    
    async def run_inference(self, duration_seconds=300, confidence_threshold=0.5):
        """Lance le processus d'inférence en temps réel"""
        print(f"\nDémarrage de l'inférence sur {duration_seconds} secondes")
        print(f"Seuil de confiance: {confidence_threshold}")
        
        await self.exchange.load_markets()
        
        try:
            start_time = time.time()
            last_price = None
            predictions = []
            context_full = False
            
            while time.time() - start_time < duration_seconds:
                try:
                    # Récupération du prix actuel
                    ticker = await self.exchange.fetch_ticker('BTC/USDT')
                    current_price = ticker['last']
                    
                    # Calcul de la variation si nous avons un prix précédent
                    if last_price is not None:
                        # Calcul de la variation en pourcentage
                        pct_change = (current_price - last_price) / last_price * 100
                        
                        # Tokenisation de la variation
                        current_token = self.tokenizer.encode(pct_change)
                        
                        # Prédiction
                        inference_start = time.perf_counter()
                        distribution = self.predictor.update_and_predict(current_token)
                        inference_time = (time.perf_counter() - inference_start) * 1000
                        
                        # Vérification du contexte
                        if not context_full and len(self.predictor.context) >= self.predictor.context_length:
                            context_full = True
                            print("\nContexte rempli, début des prédictions valides")
                        
                        # Calcul de la confiance et du signal
                        confidence = self.calculate_confidence(distribution) if context_full else 0.0
                        signal = (self.predictor.get_trading_signal(distribution) 
                                if confidence > confidence_threshold and context_full 
                                else 0)
                        
                        # Évaluation de la dernière prédiction
                        accuracy = None
                        if len(predictions) > 0 and context_full:
                            last_pred = predictions[-1]
                            accuracy = self.evaluate_prediction(
                                last_pred['distribution'],
                                pct_change
                            )
                        
                        # Enregistrement de la prédiction
                        predictions.append({
                            'timestamp': datetime.now(),
                            'price': current_price,
                            'pct_change': pct_change,
                            'token': current_token,
                            'distribution': distribution,
                            'confidence': confidence,
                            'signal': signal,
                            'inference_time': inference_time,
                            'context_full': context_full
                        })
                        
                        # Affichage en temps réel
                        status = "READY" if context_full else "FILLING"
                        accuracy_str = ""
                        if accuracy:
                            accuracy_str = (f"| Erreur: {accuracy['error']:.4f}% "
                                          f"| Direction: {'✓' if accuracy['direction_match'] else '✗'}")
                        
                        print(f"\rPrix: {current_price:.2f} | "
                              f"Var: {pct_change:.4f}% | "
                              f"Token: {current_token} ({self.tokenizer.decode(current_token)}) | "
                              f"Conf: {confidence:.2%} | "
                              f"Signal: {'🔼' if signal == 1 else '🔽' if signal == -1 else '➡️'} "
                              f"{accuracy_str} | "
                              f"Status: {status} | "
                              f"Inférence: {inference_time:.2f}ms", end='')
                    
                    last_price = current_price
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    print(f"\nErreur pendant le tick: {str(e)}")
                    await asyncio.sleep(1)
                    continue
            
            # Statistiques finales
            print("\n\nRésultats finaux:")
            
            # Filtrer les prédictions valides
            valid_predictions = [p for p in predictions if p['context_full']]
            
            if valid_predictions:
                print("\nStatistiques des variations:")
                pct_changes = [p['pct_change'] for p in valid_predictions]
                print(f"Nombre de ticks: {len(valid_predictions)}")
                print(f"Variation moyenne: {np.mean(pct_changes):.4f}%")
                print(f"Variation médiane: {np.median(pct_changes):.4f}%")
                print(f"Écart-type: {np.std(pct_changes):.4f}%")
                print(f"Min: {min(pct_changes):.4f}%")
                print(f"Max: {max(pct_changes):.4f}%")
                
                print("\nDistribution des tokens:")
                token_counts = {}
                for p in valid_predictions:
                    token = p['token']
                    token_counts[token] = token_counts.get(token, 0) + 1
                
                for token, count in sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                    print(f"Token {token} ({self.tokenizer.decode(token)}): "
                          f"{count} fois ({count/len(valid_predictions)*100:.1f}%)")
                
                print("\nPerformance du modèle:")
                confidences = [p['confidence'] for p in valid_predictions]
                inference_times = [p['inference_time'] for p in valid_predictions]
                signals = [p['signal'] for p in valid_predictions]
                
                print(f"Confiance moyenne: {np.mean(confidences):.2%}")
                print(f"Temps d'inférence moyen: {np.mean(inference_times):.2f}ms")
                print(f"Signaux générés: {len([s for s in signals if s != 0])} / {len(valid_predictions)}")
                
                # Sauvegarde des résultats
                results = {
                    'predictions': valid_predictions,
                    'config': {
                        'confidence_threshold': confidence_threshold,
                        'bucket_size': self.tokenizer.bucket_size,
                        'min_pct': self.tokenizer.min_pct,
                        'max_pct': self.tokenizer.max_pct,
                        'vocab_size': self.vocab_size
                    },
                    'stats': {
                        'pct_changes': pct_changes,
                        'confidences': confidences,
                        'inference_times': inference_times,
                        'token_distribution': token_counts
                    }
                }
                
                timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                np.save(f'inference_results_{timestamp_str}.npy', results)
                print(f"\nRésultats sauvegardés dans inference_results_{timestamp_str}.npy")
            
        except Exception as e:
            print(f"\nErreur principale: {str(e)}")
        finally:
            await self.exchange.close()

async def main():
    predictor = MarketPredictor('best_model.pt')
    await predictor.run_inference(
        duration_seconds=300,  # 5 minutes
        confidence_threshold=0.5
    )

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())