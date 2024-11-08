import ccxt.async_support as ccxt
import asyncio
import time
import torch
import numpy as np
from datetime import datetime

# Chargement du modèle
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FastTokenTransformer().to(device)
model.load_state_dict(torch.load('best_model.pt'))
model.eval()  # Mode inférence

predictor = TradingPredictor(model, tokenizer, device=device)
exchange = ccxt.bitget()

async def inference_speed_test(duration_seconds=60):
    await exchange.load_markets()
    
    try:
        start_time = time.time()
        inference_times = []
        predictions = []
        
        print(f"Démarrage test d'inférence sur {duration_seconds} secondes...")
        print(f"Device: {device}")
        
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
            signal = predictor.get_trading_signal(distribution)
            inference_time = (time.perf_counter() - inference_start) * 1000  # en ms
            
            total_tick_time = (time.perf_counter() - tick_start) * 1000  # en ms
            
            inference_times.append(inference_time)
            predictions.append((price, token, signal))
            
            print(f"\rPrix: {price:.2f} USDT | "
                  f"Token: {token} ({tokenizer.decode(token)}) | "
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

# Lancer le test
loop = asyncio.get_event_loop()
loop.run_until_complete(inference_speed_test(60))  # Test sur 60 secondes