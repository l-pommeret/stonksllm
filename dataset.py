from collections import deque
import ccxt.async_support as ccxt
import asyncio
import nest_asyncio
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from tokenizer import PriceChangeTokenizer

# Permet d'imbriquer les event loops dans Jupyter
nest_asyncio.apply()

# Initialisation avec les nouveaux paramètres
tokenizer = PriceChangeTokenizer(bucket_size=0.002, min_pct=-0.5, max_pct=0.5)
context_window = deque(maxlen=2400)  # 20 minutes (1 tick/0.5s)
raw_prices = deque(maxlen=2400)
timestamps = deque(maxlen=2400)

exchange = ccxt.bitget()

async def collect_data_timed(duration_minutes=20):
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    tick_count = 0
    last_price = None
    
    # Pour les stats
    all_pct_changes = []
    
    try:
        exchange = ccxt.bitget()
        await exchange.load_markets()
        
        while time.time() < end_time:
            try:
                ticker = await exchange.fetch_ticker('BTC/USDT')  # Changé pour BTC/USDT
                current_time = time.time()
                price = ticker['last']
                
                if last_price is not None:
                    pct_change = (price - last_price) / last_price * 100
                    token = tokenizer.encode(pct_change)
                    
                    timestamps.append(current_time)
                    raw_prices.append(price)
                    context_window.append(token)
                    all_pct_changes.append(pct_change)
                
                last_price = price
                tick_count += 1
                elapsed = current_time - start_time
                remaining = end_time - current_time
                
                if last_price is not None:
                    print(f"\rPrix: {price:.2f}, Var: {pct_change:.4f}%, "
                          f"Token: {token} ({tokenizer.decode(token)}), "
                          f"Buffer: {len(context_window)}, "
                          f"Temps restant: {remaining:.1f}s", end='')
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"\nErreur pendant tick: {e}")
                await asyncio.sleep(1)
                
    finally:
        await exchange.close()
        
        data = {
            'timestamps': list(timestamps),
            'prices': list(raw_prices),
            'tokens': list(context_window),
            'pct_changes': all_pct_changes
        }
        
        # Statistiques détaillées
        print("\n\nStatistiques sur les variations tick-to-tick:")
        print(f"Minimum: {min(all_pct_changes):.4f}%")
        print(f"Maximum: {max(all_pct_changes):.4f}%")
        print(f"Moyenne: {np.mean(all_pct_changes):.4f}%")
        print(f"Médiane: {np.median(all_pct_changes):.4f}%")
        print(f"Écart-type: {np.std(all_pct_changes):.4f}%")
        
        # Distribution des tokens
        unique_tokens, counts = np.unique(list(context_window), return_counts=True)
        print("\nDistribution des tokens:")
        for token, count in sorted(zip(unique_tokens, counts), key=lambda x: x[1], reverse=True)[:10]:
            print(f"Token {token} ({tokenizer.decode(token)}): {count} occurrences ({count/len(context_window)*100:.1f}%)")
        
        # Exemple de séquence
        print("\nExemple de séquence (10 derniers ticks):")
        for i in range(-10, 0):
            t = list(context_window)[i]
            p = list(raw_prices)[i]
            v = list(all_pct_changes)[i]
            print(f"Prix: {p:.2f} USDT | Variation: {v:.4f}% | Token: {t} ({tokenizer.decode(t)})")
        
        # Sauvegarde
        np.save(f'dataset.npy', data)
        
        print(f"\nCollecte terminée à {datetime.now()}")
        print(f"Nombre total de ticks: {tick_count}")
        if tick_count > 0:
            print(f"Moyenne ticks/sec: {tick_count/elapsed:.2f}")
        print(f"Données sauvegardées dans dataset.npy")
        
        return data

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    data = loop.run_until_complete(collect_data_timed(1))