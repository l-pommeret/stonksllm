import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
import asyncio
import time
from inference import MarketPredictor

@dataclass
class Trade:
    timestamp: datetime
    side: str  # 'BUY' or 'SELL'
    price: float
    quantity: float
    fees: float
    pnl: Optional[float] = None  # Pour les positions fermées
    
class Position:
    def __init__(self):
        self.quantity: float = 0
        self.average_entry: float = 0
        self.unrealized_pnl: float = 0
        self.realized_pnl: float = 0
        
    def update(self, current_price: float):
        if self.quantity != 0:
            self.unrealized_pnl = (current_price - self.average_entry) * self.quantity
            
class FuturesPortfolioSimulator:
    def __init__(
        self,
        initial_capital: float = 10000.0,
        maker_fee: float = 0.0002,  # 0.02% pour les futures
        taker_fee: float = 0.0005,  # 0.05% pour les futures
        leverage: float = 5,         # Effet de levier x5 par défaut
        risk_per_trade: float = 0.02,
        max_position_size: float = 0.8,  # 80% du capital avec levier
        stop_loss_pct: float = 0.01,     # Plus serré avec le levier
        take_profit_pct: float = 0.02,   # Plus serré avec le levier
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        self.position = Position()
        self.trades = []
        self.equity_curve = []
        
        self.peak_capital = initial_capital
        self.max_drawdown = 0.0
        self.win_count = 0
        self.loss_count = 0
        
    def calculate_position_size(self, price: float) -> float:
        """Calcule la taille de position avec levier"""
        # Calcul basé sur le capital effectif (avec levier)
        effective_capital = self.current_capital * self.leverage
        max_quantity = (effective_capital * self.max_position_size) / price
        
        # Ajustement pour le stop loss avec levier
        risk_amount = self.current_capital * self.risk_per_trade
        price_to_stop = price * self.stop_loss_pct
        risk_based_quantity = (risk_amount * self.leverage) / price_to_stop
        
        return min(max_quantity, risk_based_quantity)
    
    def execute_trade(self, timestamp: datetime, side: str, price: float, confidence: float) -> bool:
        """Exécute un trade futures"""
        if side == 'BUY' and self.position.quantity > 0:
            return False
        if side == 'SELL' and self.position.quantity < 0:
            return False
        
        # Calcul de la quantité avec levier
        base_quantity = self.calculate_position_size(price)
        quantity = base_quantity * confidence
        
        # Calcul des frais (sur la position avec levier)
        position_value = quantity * price
        fees = position_value * self.taker_fee
        
        # Vérification de la marge requise
        margin_required = position_value / self.leverage
        if margin_required + fees > self.current_capital:
            return False
        
        # Exécution du trade
        if side == 'BUY':
            self.position.quantity += quantity
            self.position.average_entry = price if self.position.quantity == quantity else (
                (self.position.average_entry * (self.position.quantity - quantity) + price * quantity)
                / self.position.quantity
            )
        else:
            self.position.quantity -= quantity
        
        # Mise à jour du capital (uniquement les frais sont déduites)
        self.current_capital -= fees
        
        # Enregistrement
        trade = Trade(
            timestamp=timestamp,
            side=side,
            price=price,
            quantity=quantity,
            fees=fees
        )
        self.trades.append(trade)
        self._update_metrics()
        return True
    
    def check_exit_conditions(self, current_price: float, funding_rate: float = 0) -> Optional[str]:
        """Vérifie les conditions de sortie incluant le taux de funding"""
        if self.position.quantity == 0:
            return None
        
        self.position.update(current_price)
        
        # Calcul du PnL en pourcentage incluant le funding
        pnl_pct = (current_price - self.position.average_entry) / self.position.average_entry
        # Ajout de l'impact du funding rate (positif pour les longs si négatif et vice versa)
        if self.position.quantity > 0:
            pnl_pct -= funding_rate  # Pour les positions longues
        else:
            pnl_pct += funding_rate  # Pour les positions courtes
        
        # Amplification des seuils par le levier
        effective_stop_loss = self.stop_loss_pct / self.leverage
        effective_take_profit = self.take_profit_pct / self.leverage
        
        if self.position.quantity > 0:  # Position longue
            if pnl_pct <= -effective_stop_loss:
                return 'STOP_LOSS'
            elif pnl_pct >= effective_take_profit:
                return 'TAKE_PROFIT'
        else:  # Position courte
            if pnl_pct >= effective_stop_loss:
                return 'STOP_LOSS'
            elif pnl_pct <= -effective_take_profit:
                return 'TAKE_PROFIT'
        
        return None
    
    def close_position(self, timestamp: datetime, price: float, reason: str) -> bool:
        """Ferme la position actuelle"""
        if self.position.quantity == 0:
            return False
            
        side = 'SELL' if self.position.quantity > 0 else 'BUY'
        quantity = abs(self.position.quantity)
        fees = quantity * price * self.taker_fee
        
        # Calcul du PnL
        pnl = (price - self.position.average_entry) * quantity * (1 if side == 'SELL' else -1)
        pnl -= fees
        
        # Mise à jour du capital et des statistiques
        self.current_capital += (quantity * price) - fees
        if pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
        
        # Enregistrement du trade de clôture
        trade = Trade(
            timestamp=timestamp,
            side=side,
            price=price,
            quantity=quantity,
            fees=fees,
            pnl=pnl
        )
        self.trades.append(trade)
        
        # Réinitialisation de la position
        self.position = Position()
        
        # Mise à jour des métriques
        self._update_metrics()
        return True
    
    def get_performance_metrics(self) -> Dict:
        """Calcule et retourne les métriques de performance du portefeuille"""
        total_trades = self.win_count + self.loss_count
        win_rate = self.win_count / total_trades if total_trades > 0 else 0
        
        # Calcul des profits/pertes
        realized_pnl = sum(trade.pnl for trade in self.trades if trade.pnl is not None)
        total_fees = sum(trade.fees for trade in self.trades)
        
        # Calcul du ROI
        final_equity = self.equity_curve[-1]['equity'] if self.equity_curve else self.current_capital
        roi = (final_equity - self.initial_capital) / self.initial_capital
        
        # Calcul du Profit Factor
        gross_profits = sum(trade.pnl for trade in self.trades if trade.pnl and trade.pnl > 0)
        gross_losses = abs(sum(trade.pnl for trade in self.trades if trade.pnl and trade.pnl < 0))
        profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'realized_pnl': realized_pnl,
            'total_fees': total_fees,
            'roi': roi,
            'max_drawdown': self.max_drawdown,
            'profit_factor': profit_factor,
            'final_capital': final_equity,
            'equity_curve': pd.DataFrame(self.equity_curve)
        }
    
async def run_portfolio_simulation(predictor, duration_seconds=300, confidence_threshold=0.5):
    """
    Exécute la simulation du portefeuille en temps réel avec le prédicteur
    """
    portfolio = PortfolioSimulator(
        initial_capital=10000.0,
        maker_fee=0.001,
        taker_fee=0.002,
        risk_per_trade=0.02,
        max_position_size=0.5,
        stop_loss_pct=0.02,
        take_profit_pct=0.04
    )
    
    print(f"\nDémarrage de la simulation sur {duration_seconds} secondes")
    print(f"Capital initial: {portfolio.initial_capital:,.2f} USDT")
    print(f"Seuil de confiance: {confidence_threshold:.1%}")
    
    await predictor.exchange.load_markets()
    
    try:
        start_time = time.time()
        last_price = None
        context_full = False
        
        while time.time() - start_time < duration_seconds:
            try:
                # Récupération du prix actuel
                ticker = await predictor.exchange.fetch_ticker('BTC/USDT')
                current_price = ticker['last']
                current_time = datetime.now()
                
                if last_price is not None:
                    # Calcul de la variation
                    pct_change = (current_price - last_price) / last_price * 100
                    current_token = predictor.tokenizer.encode(pct_change)
                    
                    # Prédiction et calcul de la confiance
                    distribution = predictor.predictor.update_and_predict(current_token)
                    confidence = predictor.calculate_confidence(distribution) if context_full else 0.0
                    
                    # Vérification du remplissage du contexte
                    if not context_full and len(predictor.predictor.context) >= predictor.predictor.context_length:
                        context_full = True
                        print("\nContexte rempli, début du trading")
                    
                    # Gestion des positions existantes
                    if portfolio.position.quantity != 0:
                        exit_signal = portfolio.check_exit_conditions(current_price)
                        if exit_signal:
                            portfolio.close_position(current_time, current_price, exit_signal)
                            print(f"\n🔄 Position fermée - Raison: {exit_signal}")
                    
                    # Signaux de trading
                    if context_full and confidence > confidence_threshold:
                        signal = predictor.predictor.get_trading_signal(distribution)
                        
                        if signal != 0 and portfolio.position.quantity == 0:
                            side = 'BUY' if signal == 1 else 'SELL'
                            success = portfolio.execute_trade(
                                current_time, 
                                side, 
                                current_price, 
                                confidence
                            )
                            if success:
                                print(f"\n📈 Trade exécuté - {side} à {current_price:.2f} USDT")
                    
                    # Affichage en temps réel
                    metrics = portfolio.get_performance_metrics()
                    status = "READY" if context_full else "FILLING"
                    
                    print(f"\rPrix: {current_price:.2f} | "
                          f"Capital: {metrics['final_capital']:.2f} | "
                          f"ROI: {metrics['roi']:.2%} | "
                          f"Position: {portfolio.position.quantity:.4f} | "
                          f"PnL: {portfolio.position.unrealized_pnl:.2f} | "
                          f"Conf: {confidence:.2%} | "
                          f"Status: {status}", end='')
                
                last_price = current_price
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"\nErreur pendant le tick: {str(e)}")
                await asyncio.sleep(1)
                continue
        
        # Fermeture de la position finale si elle existe
        if portfolio.position.quantity != 0:
            portfolio.close_position(datetime.now(), current_price, "END_OF_SIMULATION")
        
        # Affichage des résultats finaux
        print("\n\n=== Résultats de la simulation ===")
        metrics = portfolio.get_performance_metrics()
        
        print(f"\nPerformance du portefeuille:")
        print(f"Capital final: {metrics['final_capital']:,.2f} USDT")
        print(f"ROI: {metrics['roi']:.2%}")
        print(f"Drawdown maximum: {metrics['max_drawdown']:.2%}")
        print(f"Nombre total de trades: {metrics['total_trades']}")
        print(f"Win rate: {metrics['win_rate']:.2%}")
        print(f"Profit factor: {metrics['profit_factor']:.2f}")
        print(f"Total des frais: {metrics['total_fees']:.2f} USDT")
        
        # Sauvegarde des résultats
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        results = {
            'metrics': metrics,
            'config': {
                'initial_capital': portfolio.initial_capital,
                'risk_per_trade': portfolio.risk_per_trade,
                'stop_loss_pct': portfolio.stop_loss_pct,
                'take_profit_pct': portfolio.take_profit_pct,
                'confidence_threshold': confidence_threshold
            }
        }
        
        np.save(f'simulation_results_{timestamp_str}.npy', results)
        print(f"\nRésultats sauvegardés dans simulation_results_{timestamp_str}.npy")
        
        # Plot de la courbe de capital si possible
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            plt.plot(metrics['equity_curve'].timestamp, metrics['equity_curve'].equity)
            plt.title('Évolution du capital')
            plt.xlabel('Date')
            plt.ylabel('Capital (USDT)')
            plt.grid(True)
            plt.savefig(f'equity_curve_{timestamp_str}.png')
            plt.close()
            print(f"Courbe de capital sauvegardée dans equity_curve_{timestamp_str}.png")
        except Exception as e:
            print(f"Impossible de générer le graphique: {str(e)}")
    
    except Exception as e:
        print(f"\nErreur principale: {str(e)}")
    finally:
        await predictor.exchange.close()

async def main():
    # Initialisation du prédicteur
    predictor = MarketPredictor('best_model.pt')
    
    # Lancement de la simulation
    await run_portfolio_simulation(
        predictor,
        duration_seconds=300,  # 5 minutes
        confidence_threshold=0.5  # 50% de confiance minimum
    )

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())