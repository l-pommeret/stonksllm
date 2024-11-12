import ccxt.async_support as ccxt
import asyncio
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
from inference import MarketPredictor
from config import API_CREDENTIALS, TRADING_PARAMS, MODEL_PARAMS, EXCHANGE_PARAMS


@dataclass
class Position:
    entry_price: float
    size: float
    side: str  # 'long' or 'short'
    pnl: float = 0.0
    
class PortfolioSimulator:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        initial_balance: float = 10000,  # SUSDT
        leverage: int = 5,
        position_size_pct: float = 0.1,  # 10% du portfolio par trade
        take_profit_pct: float = 0.5,    # 0.5% de take profit
        stop_loss_pct: float = 0.3,      # 0.3% de stop loss
    ):
        # Configuration du client Bitget demo
        self.exchange = ccxt.bitget({
            'apiKey': api_key,
            'secret': api_secret,
            'password': passphrase,
            'options': {
                'defaultType': 'swap',
                'defaultSubType': 'linear',
                'marginMode': 'isolated',
            }
        })
        
        # Configuration du portfolio
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.position_size_pct = position_size_pct
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        
        # État du portfolio
        self.current_position: Position = None
        self.trades_history: List[Dict] = []
        self.last_signal = 0
        
        # Statistiques
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Configuration du symbol
        self.symbol = "SBTCSUSDT"
        self.base_currency = "SBTC"
        self.quote_currency = "SUSDT"
        
        print(f"\nInitialisation du simulateur de portfolio:")
        print(f"Balance initiale: {self.initial_balance} SUSDT")
        print(f"Levier: {self.leverage}x")
        print(f"Taille position: {self.position_size_pct*100}% du portfolio")
        print(f"Take profit: {self.take_profit_pct}%")
        print(f"Stop loss: {self.stop_loss_pct}%")
    
    async def initialize(self):
        """Initialize exchange connection and load markets"""
        await self.exchange.load_markets()
        await self.set_leverage()
    
    async def set_leverage(self):
        """Configure le levier pour le trading"""
        try:
            await self.exchange.set_leverage(self.leverage, self.symbol)
            print(f"Levier configuré à {self.leverage}x pour {self.symbol}")
        except Exception as e:
            print(f"Erreur lors de la configuration du levier: {str(e)}")
    
    async def place_order(self, side: str, size: float, price: float, 
                         stop_loss: float = None, take_profit: float = None) -> Dict:
        """Place un ordre sur le marché avec SL/TP"""
        try:
            # Ordre principal
            order = await self.exchange.create_order(
                symbol=self.symbol,
                type='limit',
                side=side,
                amount=size,
                price=price,
                params={
                    'productType': 'susdt-futures',
                    'marginMode': 'isolated',
                    'marginCoin': 'SUSDT',
                    'tradeSide': 'open',
                    'presetStopLossPrice': stop_loss if stop_loss else None,
                    'presetStopSurplusPrice': take_profit if take_profit else None
                }
            )
            return order
        except Exception as e:
            print(f"Erreur lors du placement de l'ordre: {str(e)}")
            return None
    
    async def close_position(self):
        """Ferme la position actuelle"""
        if not self.current_position:
            return
        
        try:
            close_side = 'sell' if self.current_position.side == 'long' else 'buy'
            order = await self.exchange.create_order(
                symbol=self.symbol,
                type='market',
                side=close_side,
                amount=self.current_position.size,
                params={
                    'productType': 'susdt-futures',
                    'marginMode': 'isolated',
                    'marginCoin': 'SUSDT',
                    'tradeSide': 'close'
                }
            )
            return order
        except Exception as e:
            print(f"Erreur lors de la fermeture de la position: {str(e)}")
            return None
    
    def calculate_position_size(self, price: float) -> float:
        """Calcule la taille de la position en fonction du portfolio"""
        position_value = self.balance * self.position_size_pct
        contract_value = position_value * self.leverage
        return contract_value / price
    
    def update_stats(self, pnl: float):
        """Met à jour les statistiques de trading"""
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
    
    async def process_signal(self, signal: int, price: float, confidence: float):
        """Traite un signal de trading"""
        if signal == 0 or confidence < 0.6:  # Ignore les signaux faibles
            return
        
        # Éviter les signaux redondants
        if signal == self.last_signal:
            return
        
        self.last_signal = signal
        
        try:
            # Fermer la position existante si direction opposée
            if self.current_position:
                if (signal > 0 and self.current_position.side == 'short') or \
                   (signal < 0 and self.current_position.side == 'long'):
                    await self.close_position()
                    self.current_position = None
            
            # Ouvrir nouvelle position
            if not self.current_position:
                size = self.calculate_position_size(price)
                side = 'buy' if signal > 0 else 'sell'
                
                # Calculer SL/TP
                stop_loss = price * (1 - self.stop_loss_pct/100) if signal > 0 else \
                           price * (1 + self.stop_loss_pct/100)
                take_profit = price * (1 + self.take_profit_pct/100) if signal > 0 else \
                            price * (1 - self.take_profit_pct/100)
                
                order = await self.place_order(
                    side=side,
                    size=size,
                    price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                if order:
                    self.current_position = Position(
                        entry_price=price,
                        size=size,
                        side='long' if signal > 0 else 'short'
                    )
                    print(f"\nNouvelle position: {side.upper()} {size:.4f} @ {price:.2f}")
                    print(f"SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
        
        except Exception as e:
            print(f"Erreur lors du traitement du signal: {str(e)}")
    
    async def run_simulation(self, duration_seconds: int = 3600):
        """Lance la simulation de trading"""
        await self.initialize()
        
        # Initialiser le predictor
        predictor = MarketPredictor('best_model.pt')
        
        print(f"\nDémarrage de la simulation sur {duration_seconds} secondes")
        start_time = datetime.now()
        
        try:
            await predictor.run_inference(
                duration_seconds=duration_seconds,
                confidence_threshold=0.6,
                callback=self.process_signal
            )
            
        except Exception as e:
            print(f"Erreur pendant la simulation: {str(e)}")
        
        finally:
            # Fermer la position finale si elle existe
            if self.current_position:
                await self.close_position()
            
            # Afficher les résultats
            print("\nRésultats de la simulation:")
            print(f"Durée: {(datetime.now() - start_time).seconds} secondes")
            print(f"Balance finale: {self.balance:.2f} SUSDT")
            print(f"P&L: {(self.balance - self.initial_balance):.2f} SUSDT")
            print(f"Return: {((self.balance/self.initial_balance - 1) * 100):.2f}%")
            print(f"Nombre de trades: {self.total_trades}")
            if self.total_trades > 0:
                win_rate = (self.winning_trades / self.total_trades) * 100
                print(f"Win rate: {win_rate:.1f}%")
            
            # Sauvegarder les résultats
            results = {
                'initial_balance': self.initial_balance,
                'final_balance': self.balance,
                'pnl': self.balance - self.initial_balance,
                'return_pct': (self.balance/self.initial_balance - 1) * 100,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': win_rate if self.total_trades > 0 else 0,
                'trades_history': self.trades_history,
                'config': {
                    'leverage': self.leverage,
                    'position_size_pct': self.position_size_pct,
                    'take_profit_pct': self.take_profit_pct,
                    'stop_loss_pct': self.stop_loss_pct
                }
            }
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            with open(f'simulation_results_{timestamp}.json', 'w') as f:
                json.dump(results, f)
            print(f"\nRésultats sauvegardés dans simulation_results_{timestamp}.json")
            
            await self.exchange.close()

async def main():
    simulator = PortfolioSimulator(
        api_key=API_CREDENTIALS['api_key'],
        api_secret=API_CREDENTIALS['api_secret'],
        passphrase=API_CREDENTIALS['passphrase'],
        initial_balance=TRADING_PARAMS['initial_balance'],
        leverage=TRADING_PARAMS['leverage'],
        position_size_pct=TRADING_PARAMS['position_size_pct'],
        take_profit_pct=TRADING_PARAMS['take_profit_pct'],
        stop_loss_pct=TRADING_PARAMS['stop_loss_pct']
    )
    
    await simulator.run_simulation(duration_seconds=MODEL_PARAMS['simulation_duration'])

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())