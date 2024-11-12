import ccxt.async_support as ccxt
import asyncio
import time
import torch
import numpy as np
from datetime import datetime
from dataset import tokenizer
from transformer import FastTokenTransformer, TradingPredictor

# Configuration de l'API Bitget
BITGET_CONFIG = {
    'apiKey': 'bg_88c...',     # Complétez votre clé API
    'secret': '3fba3...',      # Complétez votre clé secrète
    'password': '',            # Ajoutez votre mot de passe API
    'options': {
        'defaultType': 'swap',    # Mode futures/swap
        'defaultSubType': 'linear' # Futures USDT-M
    },
    'enableRateLimit': True
}

class BitgetFuturesTrader:
    def __init__(
        self,
        predictor,
        symbol='BTCUSDT',
        leverage=5,
        position_size=0.1,  # Taille de position en % du capital
        testnet=True
    ):
        # Configuration de l'exchange avec les credentials
        self.exchange = ccxt.bitget(BITGET_CONFIG)
        
        if testnet:
            self.exchange.set_sandbox_mode(True)
            print("Mode testnet activé")
            
        self.predictor = predictor
        self.symbol = symbol
        self.leverage = leverage
        self.position_size = position_size
        self.current_position = 0
        
    async def initialize(self):
        """Configure le compte et les paramètres initiaux"""
        await self.exchange.load_markets()
        
        try:
            # Test de connexion API
            balance = await self.exchange.fetch_balance()
            print("Connexion API réussie!")
        except Exception as e:
            print(f"Erreur de connexion API: {e}")
            return False
        
        # Configuration du levier
        try:
            await self.exchange.set_leverage(self.leverage, self.symbol)
            print(f"Levier configuré à {self.leverage}x")
        except Exception as e:
            print(f"Erreur lors de la configuration du levier: {e}")
            return False
            
        # Récupération du solde initial
        self.initial_balance = float(balance['USDT']['free'])
        print(f"Balance initiale: {self.initial_balance} USDT")
        return True
    
    async def get_position_info(self):
        """Récupère les informations sur la position actuelle"""
        try:
            positions = await self.exchange.fetch_positions([self.symbol])
            if positions:
                position = positions[0]
                return {
                    'size': float(position['contracts']),
                    'side': position['side'],
                    'entry_price': float(position['entryPrice']),
                    'leverage': float(position['leverage']),
                    'unrealized_pnl': float(position['unrealizedPnl']),
                    'margin': float(position['initialMargin'])
                }
        except Exception as e:
            print(f"Erreur lors de la récupération de la position: {e}")
        return None
    
    async def execute_trade(self, signal, current_price):
        """
        Exécute un trade basé sur le signal
        signal: 1 (long), -1 (short), 0 (neutre)
        """
        try:
            position = await self.get_position_info()
            balance = await self.exchange.fetch_balance()
            available_balance = float(balance['USDT']['free'])
            
            # Calcul de la taille de la position
            contract_value = current_price  # Valeur d'un contrat en USDT
            position_value = available_balance * self.position_size
            contracts = position_value / contract_value * self.leverage
            
            # Arrondi à la précision du marché
            market = self.exchange.market(self.symbol)
            contracts = self.exchange.amount_to_precision(self.symbol, contracts)
            
            if signal != 0:
                # Fermeture position existante si direction opposée
                if position and ((signal > 0 and position['side'] == 'short') or
                               (signal < 0 and position['side'] == 'long')):
                    close_order = await self.exchange.create_market_order(
                        self.symbol,
                        'buy' if position['side'] == 'short' else 'sell',
                        position['size'],
                        {'reduceOnly': True}
                    )
                    print(f"Position fermée: {close_order}")
                    
                # Ouverture nouvelle position
                side = 'buy' if signal > 0 else 'sell'
                order = await self.exchange.create_market_order(
                    self.symbol,
                    side,
                    contracts,
                    {'leverage': self.leverage}
                )
                print(f"Ordre exécuté: {order}")
                
                # Mise à jour position actuelle
                self.current_position = contracts if signal > 0 else -contracts
                
        except Exception as e:
            print(f"Erreur lors de l'exécution du trade: {e}")
    
    async def run(self, duration_seconds=3600, confidence_threshold=0.7):
        """Lance le trader sur la durée spécifiée"""
        if not await self.initialize():
            print("Échec de l'initialisation, arrêt du trader")
            return
            
        start_time = time.time()
        trades_executed = 0
        
        print(f"\nDémarrage du trader futures pour {duration_seconds} secondes...")
        print(f"Symbol: {self.symbol}")
        print(f"Levier: {self.leverage}x")
        print(f"Taille de position: {self.position_size * 100}% du capital")
        
        try:
            while time.time() - start_time < duration_seconds:
                # Récupération du prix et calcul du token
                ticker = await self.exchange.fetch_ticker(self.symbol)
                current_price = ticker['last']
                current_pct_change = (current_price - ticker['open']) / ticker['open'] * 100
                
                # Prédiction et signal
                token = tokenizer.encode(current_pct_change)
                distribution = self.predictor.update_and_predict(token)
                confidence = calculate_confidence(distribution)
                
                signal = self.predictor.get_trading_signal(distribution) if confidence > confidence_threshold else 0
                
                # Exécution si signal valide
                if signal != 0:
                    await self.execute_trade(signal, current_price)
                    trades_executed += 1
                
                # Affichage status
                position = await self.get_position_info()
                balance = await self.exchange.fetch_balance()
                current_balance = float(balance['USDT']['total'])
                
                pnl_total = ((current_balance - self.initial_balance) / self.initial_balance) * 100
                
                print(f"\rPrix: {current_price:.2f} | "
                      f"Signal: {'🔼' if signal == 1 else '🔽' if signal == -1 else '➡️'} | "
                      f"Position: {'Long' if position and position['side'] == 'long' else 'Short' if position and position['side'] == 'short' else 'None'} | "
                      f"PnL: {pnl_total:.2f}% | "
                      f"Balance: {current_balance:.2f} USDT | "
                      f"Trades: {trades_executed}", end='')
                
                await asyncio.sleep(1)
                
        except Exception as e:
            print(f"\nErreur pendant l'exécution: {e}")
        finally:
            # Affichage des résultats finaux
            balance = await self.exchange.fetch_balance()
            final_balance = float(balance['USDT']['total'])
            total_return = ((final_balance - self.initial_balance) / self.initial_balance) * 100
            
            print("\n\nRésultats du trading:")
            print(f"Balance finale: {final_balance:.2f} USDT")
            print(f"Return total: {total_return:.2f}%")
            print(f"Nombre de trades: {trades_executed}")
            
            await self.exchange.close()

# Chargement du modèle
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FastTokenTransformer().to(device)
model.load_state_dict(torch.load('best_model.pt'))
model.eval()
predictor = TradingPredictor(model, tokenizer, device=device)

# Fonction principale
async def main():
    trader = BitgetFuturesTrader(
        predictor=predictor,
        leverage=5,           # Levier 5x
        position_size=0.1,    # 10% du capital par trade
        testnet=True         # Utilisation du testnet
    )
    await trader.run(duration_seconds=3600)  # Test sur 1 heure

# Lancement
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())