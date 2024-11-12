import ccxt.async_support as ccxt
import asyncio
import time
import torch
import numpy as np
from datetime import datetime
from tokenizer import PriceChangeTokenizer
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

def calculate_confidence(distribution):
    """
    Calcule le niveau de confiance basé sur l'écart avec une distribution uniforme.
    Retourne une valeur entre 0 et 1, où 1 indique une confiance maximale.
    """
    uniform_dist = torch.ones_like(distribution) / len(distribution)
    kl_div = torch.sum(distribution * torch.log(distribution / uniform_dist))
    confidence = 1 - torch.exp(-kl_div)
    return confidence.item()

class BitgetFuturesTrader:
    def __init__(
        self,
        predictor,
        tokenizer,
        symbol='BTCUSDT',
        leverage=5,
        position_size=0.1,  # Taille de position en % du capital
        testnet=True,
        neutral_range=(-0.1, 0.1)  # Zone neutre adaptée à la volatilité BTC
    ):
        # Configuration de l'exchange avec les credentials
        self.exchange = ccxt.bitget(BITGET_CONFIG)
        
        if testnet:
            self.exchange.set_sandbox_mode(True)
            print("Mode testnet activé")
            
        self.predictor = predictor
        self.tokenizer = tokenizer
        self.symbol = symbol
        self.leverage = leverage
        self.position_size = position_size
        self.current_position = 0
        self.neutral_range = neutral_range
        
        # Métriques de trading
        self.trades_history = []
        self.predictions_accuracy = []
        
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
        
        print("\nParamètres du tokenizer:")
        print(f"Taille bucket: {self.tokenizer.bucket_size}%")
        print(f"Range: {self.tokenizer.min_pct}% à {self.tokenizer.max_pct}%")
        print(f"Nombre de tokens: {self.tokenizer.vocab_size}")
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
    
    def evaluate_last_prediction(self, predicted_pct, actual_pct):
        """Évalue la qualité de la dernière prédiction"""
        tolerance = self.tokenizer.bucket_size
        exact_match = abs(predicted_pct - actual_pct) <= tolerance
        direction_match = (predicted_pct * actual_pct) > 0
        
        self.predictions_accuracy.append({
            'timestamp': datetime.now(),
            'predicted_pct': predicted_pct,
            'actual_pct': actual_pct,
            'exact_match': exact_match,
            'direction_match': direction_match,
            'error': abs(predicted_pct - actual_pct)
        })
    
    async def execute_trade(self, signal, current_price, confidence):
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
            
            # Ajustement de la taille en fonction de la confiance
            contracts *= min(1.0, confidence * 1.5)  # Max 150% de la taille standard à confiance maximale
            
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
                    print(f"\nPosition fermée: {close_order}")
                    
                # Ouverture nouvelle position
                side = 'buy' if signal > 0 else 'sell'
                order = await self.exchange.create_market_order(
                    self.symbol,
                    side,
                    contracts,
                    {'leverage': self.leverage}
                )
                
                self.trades_history.append({
                    'timestamp': datetime.now(),
                    'side': side,
                    'price': current_price,
                    'size': contracts,
                    'confidence': confidence,
                    'order': order
                })
                
                print(f"\nOrdre exécuté: {order}")
                
                # Mise à jour position actuelle
                self.current_position = contracts if signal > 0 else -contracts
                
        except Exception as e:
            print(f"\nErreur lors de l'exécution du trade: {e}")
    
    async def run(self, duration_seconds=3600, confidence_threshold=0.7):
        """Lance le trader sur la durée spécifiée"""
        if not await self.initialize():
            print("Échec de l'initialisation, arrêt du trader")
            return
            
        start_time = time.time()
        last_price = None
        
        print(f"\nDémarrage du trader futures pour {duration_seconds} secondes...")
        print(f"Symbol: {self.symbol}")
        print(f"Levier: {self.leverage}x")
        print(f"Taille de position: {self.position_size * 100}% du capital")
        print(f"Seuil de confiance: {confidence_threshold}")
        print(f"Zone neutre: {self.neutral_range[0]}% à {self.neutral_range[1]}%")
        
        try:
            while time.time() - start_time < duration_seconds:
                # Récupération du prix et calcul des variations
                ticker = await self.exchange.fetch_ticker(self.symbol)
                current_price = ticker['last']
                
                if last_price:
                    actual_pct_change = (current_price - last_price) / last_price * 100
                    if self.predictions_accuracy:
                        last_prediction = self.predictions_accuracy[-1]
                        self.evaluate_last_prediction(last_prediction['predicted_pct'], actual_pct_change)
                
                # Calcul du token actuel
                current_pct_change = (current_price - ticker['open']) / ticker['open'] * 100
                token = self.tokenizer.encode(current_pct_change)
                
                # Prédiction et signal
                distribution = self.predictor.update_and_predict(token)
                confidence = calculate_confidence(distribution)
                
                if confidence > confidence_threshold:
                    signal = self.predictor.get_trading_signal(distribution, self.neutral_range)
                    if signal != 0:
                        await self.execute_trade(signal, current_price, confidence)
                else:
                    signal = 0
                
                # Affichage status
                position = await self.get_position_info()
                balance = await self.exchange.fetch_balance()
                current_balance = float(balance['USDT']['total'])
                pnl_total = ((current_balance - self.initial_balance) / self.initial_balance) * 100
                
                # Calcul des métriques de prédiction
                accuracy_str = ""
                if len(self.predictions_accuracy) > 0:
                    last_10_acc = self.predictions_accuracy[-10:]
                    direction_acc = np.mean([p['direction_match'] for p in last_10_acc])
                    exact_acc = np.mean([p['exact_match'] for p in last_10_acc])
                    accuracy_str = f"| Dir Acc (10): {direction_acc:.1%} | Exact (10): {exact_acc:.1%}"
                
                print(f"\rPrix: {current_price:.2f} | "
                      f"Token: {token} ({self.tokenizer.decode(token)}) | "
                      f"Conf: {confidence:.2%} | "
                      f"Signal: {'🔼' if signal == 1 else '🔽' if signal == -1 else '➡️'} | "
                      f"Position: {'Long' if position and position['side'] == 'long' else 'Short' if position and position['side'] == 'short' else 'None'} | "
                      f"PnL: {pnl_total:.2f}% | "
                      f"Balance: {current_balance:.2f} USDT | "
                      f"Trades: {len(self.trades_history)} "
                      f"{accuracy_str}", end='')
                
                last_price = current_price
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
            print(f"Nombre de trades: {len(self.trades_history)}")
            
            if self.predictions_accuracy:
                print("\nMétriques de prédiction:")
                print(f"Accuracy direction: {np.mean([p['direction_match'] for p in self.predictions_accuracy]):.2%}")
                print(f"Accuracy exacte (±{self.tokenizer.bucket_size}%): {np.mean([p['exact_match'] for p in self.predictions_accuracy]):.2%}")
                print(f"Erreur moyenne: {np.mean([p['error'] for p in self.predictions_accuracy]):.4f}%")
                print(f"Erreur médiane: {np.median([p['error'] for p in self.predictions_accuracy]):.4f}%")
            
            # Sauvegarde des résultats
            results = {
                'config': {
                    'symbol': self.symbol,
                    'leverage': self.leverage,
                    'position_size': self.position_size,
                    'confidence_threshold': confidence_threshold,
                    'neutral_range': self.neutral_range,
                    'tokenizer_config': {
                        'bucket_size': self.tokenizer.bucket_size,
                        'min_pct': self.tokenizer.min_pct,
                        'max_pct': self.tokenizer.max_pct,
                        'vocab_size': self.tokenizer.vocab_size
                    }
                },
                'initial_balance': self.initial_balance,
                'final_balance': final_balance,
                'total_return': total_return,
                'trades': self.trades_history,
                'predictions': self.predictions_accuracy
            }
            
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            np.save(f'trading_results_{timestamp_str}.npy', results)
            print(f"\nRésultats sauvegardés dans trading_results_{timestamp_str}.npy")
            
            await self.exchange.close()

# Création des instances
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = PriceChangeTokenizer(bucket_size=0.002, min_pct=-0.5, max_pct=0.5)
model = FastTokenTransformer(n_tokens=tokenizer.vocab_size).to(device)
model.load_state_dict(torch.load('best_model.pt'))
model.eval()
predictor = TradingPredictor(model, tokenizer, device=device)

# Fonction principale
async def main():
    trader = BitgetFuturesTrader(
        predictor=predictor,
        tokenizer=tokenizer,
        leverage=5,           # Levier 5x
        position_size=0.1,    # 10% du capital par trade
        testnet=True,        # Utilisation du testnet
        neutral_range=(-0.1, 0.1)  # Zone neutre adaptée à BTC
    )
    await trader.run(duration_seconds=3600, confidence_threshold=0.7)

# Lancement
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())