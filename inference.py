import ccxt.async_support as ccxt
import asyncio
import time
import torch
import numpy as np
from datetime import datetime
from dataset import tokenizer
from transformer import FastTokenTransformer, TradingPredictor

# Chargement du modèle
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FastTokenTransformer().to(device)
model.load_state_dict(torch.load('best_model.pt'))
model.eval()  # Mode inférence
predictor = TradingPredictor(model, tokenizer, device=device)
exchange = ccxt.bitget()

def calculate_confidence(distribution):
    """
    Calcule le niveau de confiance basé sur l'écart avec une distribution uniforme.
    """
    uniform_dist = torch.ones_like(distribution) / len(distribution)
    kl_div = torch.sum(distribution * torch.log(distribution / uniform_dist))
    confidence = 1 - torch.exp(-kl_div)
    return confidence.item()

def evaluate_prediction_accuracy(predicted_distribution, next_pct_change, tolerance=0.05):
    """
    Évalue l'exactitude de la prédiction par rapport à la variation réelle
    
    Args:
        predicted_distribution: Distribution de probabilité sur les tokens
        next_pct_change: Variation réelle observée
        tolerance: Marge d'erreur acceptée en pourcentage absolu
    
    Returns:
        dict: Métriques d'accuracy
    """
    # Token le plus probable
    max_prob_token = torch.argmax(predicted_distribution).item()
    predicted_pct = (max_prob_token - 100) * 0.05  # Conversion token -> pourcentage
    
    # Top-3 tokens les plus probables
    top3_tokens = torch.topk(predicted_distribution, 3).indices.tolist()
    top3_pcts = [(t - 100) * 0.05 for t in top3_tokens]
    
    # Calcul des différentes métriques
    exact_match = abs(predicted_pct - next_pct_change) <= tolerance
    direction_match = (predicted_pct * next_pct_change) > 0  # Même signe
    top3_match = any(abs(p - next_pct_change) <= tolerance for p in top3_pcts)
    
    return {
        'exact_match': exact_match,
        'direction_match': direction_match,
        'top3_match': top3_match,
        'predicted_pct': predicted_pct,
        'actual_pct': next_pct_change,
        'error': abs(predicted_pct - next_pct_change)
    }

async def inference_speed_test(duration_seconds=60, confidence_threshold=0.7):
    await exchange.load_markets()
    try:
        start_time = time.time()
        inference_times = []
        predictions = []
        accuracy_metrics = []
        last_price = None
        
        print(f"Démarrage test d'inférence sur {duration_seconds} secondes...")
        print(f"Device: {device}")
        print(f"Seuil de confiance: {confidence_threshold}")

        while time.time() - start_time < duration_seconds:
            tick_start = time.perf_counter()
            
            # Récupération du prix
            ticker = await exchange.fetch_ticker('BTC/USDT')
            current_price = ticker['last']
            
            # Calcul de la variation depuis le dernier prix
            if last_price is not None:
                actual_pct_change = (current_price - last_price) / last_price * 100
                
                # Si nous avons une prédiction précédente, évaluons son accuracy
                if predictions:
                    last_pred = predictions[-1]
                    accuracy = evaluate_prediction_accuracy(last_pred['distribution'], actual_pct_change)
                    accuracy_metrics.append(accuracy)
            
            last_price = current_price
            
            # Calcul du token actuel
            current_pct_change = (current_price - ticker['open']) / ticker['open'] * 100
            token = tokenizer.encode(current_pct_change)
            
            # Inférence avec mesure du temps
            inference_start = time.perf_counter()
            distribution = predictor.update_and_predict(token)
            confidence = calculate_confidence(distribution)
            
            if confidence > confidence_threshold:
                signal = predictor.get_trading_signal(distribution)
            else:
                signal = 0
                
            inference_time = (time.perf_counter() - inference_start) * 1000
            total_tick_time = (time.perf_counter() - tick_start) * 1000
            
            inference_times.append(inference_time)
            predictions.append({
                'price': current_price,
                'token': token,
                'signal': signal,
                'confidence': confidence,
                'distribution': distribution
            })
            
            # Affichage temps réel avec métriques d'accuracy si disponibles
            accuracy_str = ""
            if accuracy_metrics:
                last_accuracy = accuracy_metrics[-1]
                accuracy_str = f"| Erreur: {last_accuracy['error']:.3f}% "
                accuracy_str += f"| Direction: {'✓' if last_accuracy['direction_match'] else '✗'} "
            
            print(f"\rPrix: {current_price:.2f} USDT | "
                  f"Token: {token} ({tokenizer.decode(token)}) | "
                  f"Confiance: {confidence:.2%} | "
                  f"Signal: {'🔼' if signal == 1 else '🔽' if signal == -1 else '➡️'} "
                  f"{accuracy_str}"
                  f"| Inférence: {inference_time:.2f}ms", end='')
                  
            await asyncio.sleep(0.5)
            
    except Exception as e:
        print(f"\nErreur: {e}")
    finally:
        await exchange.close()
        
    # Statistiques finales
    if accuracy_metrics:
        print("\n\nStatistiques de prédiction:")
        print(f"Nombre total de prédictions évaluées: {len(accuracy_metrics)}")
        print(f"Accuracy exacte (±0.05%): {np.mean([m['exact_match'] for m in accuracy_metrics]):.2%}")
        print(f"Accuracy direction: {np.mean([m['direction_match'] for m in accuracy_metrics]):.2%}")
        print(f"Accuracy top-3: {np.mean([m['top3_match'] for m in accuracy_metrics]):.2%}")
        print(f"Erreur moyenne: {np.mean([m['error'] for m in accuracy_metrics]):.3f}%")
        print(f"Erreur médiane: {np.median([m['error'] for m in accuracy_metrics]):.3f}%")
    
    print("\nStatistiques d'inférence:")
    print(f"Temps moyen d'inférence: {np.mean(inference_times):.2f}ms")
    print(f"Signaux générés: {len([p for p in predictions if p['signal'] != 0])} / {len(predictions)}")
    print(f"Confiance moyenne: {np.mean([p['confidence'] for p in predictions]):.2%}")

# Lancer le test
loop = asyncio.get_event_loop()
loop.run_until_complete(inference_speed_test(60, confidence_threshold=0.7))