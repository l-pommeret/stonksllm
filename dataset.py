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

class FuturesDataCollector:
    def __init__(self, 
                 symbol='BTC/USDT:USDT',  # Format pour les futures perpetuels sur Bitget
                 bucket_size=0.002,
                 window_size=2400):
        self.symbol = symbol
        self.bucket_size = bucket_size
        self.window_size = window_size
        self.context_window = deque(maxlen=window_size)
        self.raw_prices = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.funding_rates = deque(maxlen=window_size)
        
        # Initialisation de l'exchange en mode futures
        self.exchange = ccxt.bitget({
            'options': {
                'defaultType': 'swap',  # Pour les futures perpetuels
            }
        })
        
    def encode_price_change(self, pct_change):
        """Encode la variation de prix en token"""
        n_buckets = int(1.0 / self.bucket_size)
        bucket_idx = int((pct_change + 0.5) / self.bucket_size)
        return max(0, min(bucket_idx, n_buckets - 1))
        
    async def collect_data_timed(self, duration_minutes=20):
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        tick_count = 0
        last_price = None
        all_pct_changes = []
        
        try:
            await self.exchange.load_markets()
            
            while time.time() < end_time:
                try:
                    # Récupération du prix et du funding rate
                    ticker = await self.exchange.fetch_ticker(self.symbol)
                    funding_info = await self.exchange.fetch_funding_rate(self.symbol)
                    
                    current_time = time.time()
                    price = ticker['last']
                    funding_rate = funding_info['fundingRate'] if 'fundingRate' in funding_info else 0
                    
                    if last_price is not None:
                        pct_change = (price - last_price) / last_price * 100
                        token = self.encode_price_change(pct_change)
                        
                        self.timestamps.append(current_time)
                        self.raw_prices.append(price)
                        self.context_window.append(token)
                        self.funding_rates.append(funding_rate)
                        all_pct_changes.append(pct_change)
                    
                    last_price = price
                    tick_count += 1
                    elapsed = current_time - start_time
                    remaining = end_time - current_time
                    
                    # Affichage en temps réel
                    print(f"\rPrix Futures: {price:.2f}, "
                          f"Variation: {pct_change:.4f}% si défini, "
                          f"Token: {token if last_price else 'N/A'}, "
                          f"Funding Rate: {funding_rate:.4f}%, "
                          f"Buffer: {len(self.context_window)}, "
                          f"Temps restant: {remaining:.1f}s", end='')
                    
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    print(f"\nErreur pendant tick: {e}")
                    await asyncio.sleep(1)
                    
        finally:
            await self.exchange.close()
            
            # Sauvegarde des données
            data = {
                'timestamps': list(self.timestamps),
                'prices': list(self.raw_prices),
                'tokens': list(self.context_window),
                'funding_rates': list(self.funding_rates),
                'pct_changes': all_pct_changes
            }
            
            # Statistiques
            print("\n\nStatistiques sur les variations:")
            if all_pct_changes:
                print(f"Minimum: {min(all_pct_changes):.4f}%")
                print(f"Maximum: {max(all_pct_changes):.4f}%")
                print(f"Moyenne: {np.mean(all_pct_changes):.4f}%")
                print(f"Médiane: {np.median(all_pct_changes):.4f}%")
                print(f"Écart-type: {np.std(all_pct_changes):.4f}%")
            
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            np.save(f'futures_dataset_{timestamp_str}.npy', data)
            print(f"\nDonnées sauvegardées dans futures_dataset_{timestamp_str}.npy")

if __name__ == "__main__":
    collector = FuturesDataCollector()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(collector.collect_data_timed(20))  