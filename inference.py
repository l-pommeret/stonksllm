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
    Plus la valeur est proche de 1, plus la distribution est éloignée de l'uniforme.
    """
    # Distribution uniforme de même taille
    uniform_dist = torch.ones_like(distribution) / len(distribution)
    
    # Calcul de la divergence KL
    kl_div = torch.sum(distribution * torch.log(distribution / uniform_dist))
    
    # Normaliser pour avoir une valeur entre 0 et 1
    confidence = 1 - torch.exp(-kl_div)
    return confidence.item()

async def inference_speed_test(duration_seconds=60, confidence_threshold=0.7):
    await exchange.load_markets()
    try:
        start_time = time.time()
        inference_times = []
        predictions = []
        print(f"Démarrage test d'inférence sur {duration_seconds} secondes...")
        print(f"Device: {device}")
        print(f"Seuil de confiance: {confidence_threshold}")

        while time.time() - start_time < duration_seconds:
            tick_start = time.perf_counter()
            
            # Récupération du prix
            ticker = await exchange.fetch_ticker('BTC/USDT')
            price = ticker['last']
            pct_change = (price - ticker['open']) / ticker['open'] * 100
            token = tokenizer.encode(pct_change)
            
            # Inférence avec mesure du temps
            inference_start = time.perf_counter()
            distribution = predictor.update_and_predict(token)
            confidence = calculate_confidence(distribution)
            
            # Signal uniquement si la confiance dépasse le seuil
            if confidence > confidence_threshold:
                signal = predictor.get_trading_signal(distribution)
            else:
                signal = 0  # Pas de signal si pas assez confiant
                
            inference_time = (time.perf_counter() - inference_start) * 1000  # en ms
            total_tick_time = (time.perf_counter() - tick_start) * 1000  # en ms
            
            inference_times.append(inference_time)
            predictions.append((price, token, signal, confidence))
            
            print(f"\rPrix: {price:.2f} USDT | "
                  f"Token: {token} ({tokenizer.decode(token)}) | "
                  f"Confiance: {confidence:.2%} | "
                  f"Signal: {'🔼' if signal == 1 else '🔽' if signal == -1 else '➡️'} | "
                  f"Temps inférence: {inference_time:.2f}ms | "
                  f"Temps total tick: {total_tick_time:.2f}ms", end='')
                  
            await asyncio.sleep(0.5)
            
    except Exception as e:
        print(f"\nErreur: {e}")
    finally:
        await exchange.close()
        
    # Statistiques finales
    print("\n\nStatistiques d'inférence:")
    print(f"Nombre total de prédictions: {len(inference_times)}")
    print(f"Temps moyen d'inférence: {np.mean(inference_times):.2f}ms")
    print(f"Temps médian d'inférence: {np.median(inference_times):.2f}ms")
    print(f"Temps min d'inférence: {np.min(inference_times):.2f}ms")
    print(f"Temps max d'inférence: {np.max(inference_times):.2f}ms")
    print(f"Écart-type temps d'inférence: {np.std(inference_times):.2f}ms")
    
    # Statistiques sur la confiance
    confidences = [p[3] for p in predictions]
    print("\nStatistiques de confiance:")
    print(f"Confiance moyenne: {np.mean(confidences):.2%}")
    print(f"Confiance médiane: {np.median(confidences):.2%}")
    print(f"Signaux générés: {len([p for p in predictions if p[2] != 0])} / {len(predictions)}")

# Lancer le test
loop = asyncio.get_event_loop()
loop.run_until_complete(inference_speed_test(60, confidence_threshold=0.7))  # Test sur 60 secondes