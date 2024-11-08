from collections import deque
import ccxt.async_support as ccxt  # Notez le changement ici
import asyncio
import nest_asyncio
import time
import numpy as np
from datetime import datetime

from tokenizer import PriceChangeTokenizer

# Permet d'imbriquer les event loops dans Jupyter
nest_asyncio.apply()

# Initialisation
tokenizer = PriceChangeTokenizer()
context_window = deque(maxlen=1200)  # 10 minutes
raw_prices = deque(maxlen=1200)
timestamps = deque(maxlen=1200)

# Initialiser l'exchange avec la version async
exchange = ccxt.bitget()

async def collect_data_timed(duration_minutes=10):
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    tick_count = 0
    
    print(f"Démarrage collecte à {datetime.now()}")
    print(f"Durée prévue: {duration_minutes} minutes")
    
    try:
        await exchange.load_markets()  # Important: charger les marchés d'abord
        
        while time.time() < end_time:
            try:
                ticker = await exchange.fetch_ticker('BTC/USDT')
                current_time = time.time()
                
                # Stocker les données
                price = ticker['last']
                pct_change = (price - ticker['open']) / ticker['open'] * 100  # Calcul manuel du %
                token = tokenizer.encode(pct_change)
                
                timestamps.append(current_time)
                raw_prices.append(price)
                context_window.append(token)
                
                # Statistiques
                tick_count += 1
                elapsed = current_time - start_time
                remaining = end_time - current_time
                ticks_per_sec = tick_count / elapsed
                
                print(f"\rPrix: {price:.2f}, Var: {pct_change:.3f}%, "
                      f"Token: {token}, Buffer: {len(context_window)}, "
                      f"Temps restant: {remaining:.1f}s", end='')
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"\nErreur pendant tick: {e}")
                await asyncio.sleep(1)
    
    except Exception as e:
        print(f"\nErreur principale: {e}")
    
    finally:
        await exchange.close()  # Important: fermer proprement l'exchange
        
        # Sauvegarde finale
        data = {
            'timestamps': list(timestamps),
            'prices': list(raw_prices),
            'tokens': list(context_window)
        }
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        np.save(f'market_data_{timestamp_str}.npy', data)
        
        print(f"\n\nCollecte terminée à {datetime.now()}")
        print(f"Nombre total de ticks: {tick_count}")
        print(f"Moyenne ticks/sec: {tick_count/elapsed:.2f}" if tick_count > 0 else "Pas de données collectées")
        print(f"Données sauvegardées dans market_data_{timestamp_str}.npy")
        
        return data

# Lancer la collecte
loop = asyncio.get_event_loop()
data = loop.run_until_complete(collect_data_timed(10))

# Afficher les statistiques si on a des données
if len(data['tokens']) > 0:
    tokens_array = np.array(data['tokens'])
    print("\nStatistiques des tokens:")
    print(f"Min: {min(tokens_array)}")
    print(f"Max: {max(tokens_array)}")
    print(f"Moyenne: {np.mean(tokens_array):.2f}")
    print(f"Écart-type: {np.std(tokens_array):.2f}")
    
    unique_tokens, counts = np.unique(tokens_array, return_counts=True)
    print("\nTop 5 tokens les plus fréquents:")
    for token, count in sorted(zip(unique_tokens, counts), key=lambda x: x[1], reverse=True)[:5]:
        print(f"Token {token} ({tokenizer.decode(token)}): {count} occurrences")