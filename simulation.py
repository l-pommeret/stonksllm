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
        self.open_time: Optional[datetime] = None
        
    def update(self, current_price: float):
        """Met à jour le PnL non réalisé de la position"""
        try:
            if self.quantity != 0 and current_price > 0 and self.average_entry > 0:
                self.unrealized_pnl = (current_price - self.average_entry) * self.quantity
            else:
                self.unrealized_pnl = 0
        except Exception as e:
            print(f"Erreur dans Position.update: {str(e)}")
            self.unrealized_pnl = 0
            
class PortfolioSimulator:
    def __init__(
        self,
        initial_capital: float = 10000.0,
        maker_fee: float = 0.001,  # 0.1%
        taker_fee: float = 0.002,  # 0.2%
        risk_per_trade: float = 0.02,  # 2% du capital
        max_position_size: float = 0.5,  # 50% du capital
        stop_loss_pct: float = 0.02,  # 2%
        take_profit_pct: float = 0.04,  # 4%
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # État du portefeuille
        self.position = Position()
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        
        # Métriques de performance
        self.peak_capital = initial_capital
        self.max_drawdown = 0.0
        self.win_count = 0
        self.loss_count = 0
    
    def _update_metrics(self):
        """Met à jour les métriques de performance du portefeuille"""
        total_equity = self.current_capital
        if self.position.quantity != 0:
            total_equity += self.position.unrealized_pnl
        
        self.peak_capital = max(self.peak_capital, total_equity)
        current_drawdown = (self.peak_capital - total_equity) / self.peak_capital
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        self.equity_curve.append({
            'timestamp': self.trades[-1].timestamp if self.trades else datetime.now(),
            'equity': total_equity,
            'drawdown': current_drawdown
        })
    
    def check_exit_conditions(self, current_price: float) -> Optional[str]:
        """Vérifie les conditions de sortie"""
        if self.position.quantity == 0 or current_price <= 0 or self.position.average_entry <= 0:
            return None
            
        self.position.update(current_price)
        
        try:
            pnl_pct = ((current_price - self.position.average_entry) / self.position.average_entry 
                      if self.position.average_entry != 0 else 0)
            
            if self.position.quantity > 0:  # Position longue
                if pnl_pct <= -self.stop_loss_pct:
                    return 'STOP_LOSS'
                elif pnl_pct >= self.take_profit_pct:
                    return 'TAKE_PROFIT'
            else:  # Position courte
                if pnl_pct >= self.stop_loss_pct:
                    return 'STOP_LOSS'
                elif pnl_pct <= -self.take_profit_pct:
                    return 'TAKE_PROFIT'
        except Exception as e:
            print(f"Erreur dans check_exit_conditions: {str(e)}")
            return None
        
        return None

    def execute_trade(self, timestamp: datetime, side: str, price: float, confidence: float) -> bool:
        """Exécute un trade"""
        try:
            if price <= 0 or confidence <= 0:
                return False
                
            if side == 'BUY' and self.position.quantity > 0:
                return False
            if side == 'SELL' and self.position.quantity < 0:
                return False
                
            base_quantity = self.calculate_position_size(price)
            if base_quantity <= 0:
                return False
                
            quantity = base_quantity * min(confidence, 1.0)
            fees = abs(quantity * price * self.taker_fee)
            total_cost = (quantity * price) + fees
            
            if total_cost > self.current_capital or total_cost <= 0:
                return False
                
            if side == 'BUY':
                self.position.quantity += quantity
                if abs(self.position.quantity) < 1e-8:
                    self.position.quantity = 0
                    return False
                    
                if self.position.quantity == quantity:
                    self.position.average_entry = price
                else:
                    try:
                        self.position.average_entry = (
                            (self.position.average_entry * (self.position.quantity - quantity) + price * quantity)
                            / self.position.quantity
                        )
                    except ZeroDivisionError:
                        return False
            else:
                self.position.quantity -= quantity
                
            self.current_capital -= total_cost
            
            # Enregistrement du timestamp d'ouverture
            self.position.open_time = timestamp
            
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
            
        except Exception as e:
            print(f"Erreur dans execute_trade: {str(e)}")
            return False
        
    def close_position(self, timestamp: datetime, price: float, reason: str) -> bool:
        """Ferme la position actuelle"""
        if self.position.quantity == 0:
            return False
            
        side = 'SELL' if self.position.quantity > 0 else 'BUY'
        quantity = abs(self.position.quantity)
        fees = quantity * price * self.taker_fee
        
        pnl = (price - self.position.average_entry) * quantity * (1 if side == 'SELL' else -1)
        pnl -= fees
        
        self.current_capital += (quantity * price) - fees
        if pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
        
        trade = Trade(
            timestamp=timestamp,
            side=side,
            price=price,
            quantity=quantity,
            fees=fees,
            pnl=pnl
        )
        self.trades.append(trade)
        
        self.position = Position()
        self._update_metrics()
        return True
    
    def get_performance_metrics(self) -> Dict:
        """Calcule et retourne les métriques de performance"""
        total_trades = self.win_count + self.loss_count
        win_rate = self.win_count / total_trades if total_trades > 0 else 0
        
        realized_pnl = sum(trade.pnl for trade in self.trades if trade.pnl is not None)
        total_fees = sum(trade.fees for trade in self.trades)
        
        final_equity = self.equity_curve[-1]['equity'] if self.equity_curve else self.current_capital
        roi = (final_equity - self.initial_capital) / self.initial_capital if self.initial_capital != 0 else 0
        
        gross_profits = sum(trade.pnl for trade in self.trades if trade.pnl and trade.pnl > 0) or 0
        gross_losses = abs(sum(trade.pnl for trade in self.trades if trade.pnl and trade.pnl < 0)) or 1e-9
        profit_factor = gross_profits / gross_losses
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'realized_pnl': realized_pnl,
            'total_fees': total_fees,
            'roi': roi,
            'max_drawdown': self.max_drawdown,
            'profit_factor': profit_factor,
            'final_capital': final_equity,
            'equity_curve': pd.DataFrame(self.equity_curve) if self.equity_curve else pd.DataFrame()
        }

    def calculate_position_size(self, price: float) -> float:
        """Calcule la taille de position optimale"""
        if price <= 0:
            return 0
            
        max_quantity = (self.current_capital * self.max_position_size) / price
        
        if self.stop_loss_pct <= 0:
            risk_based_quantity = 0
        else:
            risk_based_quantity = (self.current_capital * self.risk_per_trade) / (price * self.stop_loss_pct)
        
        return min(max_quantity, risk_based_quantity)

async def run_portfolio_simulation(predictor, duration_seconds=300, confidence_threshold=0.75, min_expected_move=0.3):
    """Exécute la simulation du portefeuille en temps réel"""
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
    print(f"Variation minimale attendue: {min_expected_move:.1f}%")
    
    await predictor.exchange.load_markets()
    
    try:
        start_time = time.time()
        last_price = None
        context_full = False
        last_trade_time = None
        min_time_between_trades = 5
        
        while time.time() - start_time < duration_seconds:
            try:
                ticker = await predictor.exchange.fetch_ticker('BTC/USDT')
                current_price = ticker['last']
                current_time = datetime.now()
                
                if last_price is not None:
                    pct_change = (current_price - last_price) / last_price * 100
                    current_token = predictor.tokenizer.encode(pct_change)
                    
                    distribution = predictor.predictor.update_and_predict(current_token)
                    confidence = predictor.calculate_confidence(distribution) if context_full else 0.0
                    expected_move = predictor.get_expected_move(distribution) if context_full else 0.0
                    
                    if not context_full and len(predictor.predictor.context) >= predictor.predictor.context_length:
                        context_full = True
                        print("\nContexte rempli, début du trading")
                    
                    # Gestion des positions existantes
                    if portfolio.position.quantity != 0:
                        if portfolio.position.open_time:
                            position_age = (current_time - portfolio.position.open_time).total_seconds()
                            if position_age >= 1:
                                portfolio.close_position(current_time, current_price, "TIME_LIMIT")
                                print(f"\n🕒 Position fermée - Raison: TIME_LIMIT après {position_age:.1f}s")
                                continue
                        
                        exit_signal = portfolio.check_exit_conditions(current_price)
                        if exit_signal:
                            portfolio.close_position(current_time, current_price, exit_signal)
                            print(f"\n🔄 Position fermée - Raison: {exit_signal}")
                    
                    # Signaux de trading
                    if context_full and confidence > confidence_threshold:
                        signal = 1 if expected_move > 0 else -1 if expected_move < 0 else 0
                        
                        can_trade = True
                        if last_trade_time:
                            time_since_last_trade = (current_time - last_trade_time).total_seconds()
                            if time_since_last_trade < min_time_between_trades:
                                can_trade = False
                        
                        if abs(expected_move) >= min_expected_move and signal != 0 and portfolio.position.quantity == 0 and can_trade:
                            side = 'BUY' if signal == 1 else 'SELL'
                            success = portfolio.execute_trade(
                                current_time, 
                                side, 
                                current_price, 
                                confidence
                            )
                            if success:
                                last_trade_time = current_time
                                print(f"\n📈 Trade exécuté - {side} à {current_price:.2f} USDT "
                                      f"(Variation attendue: {expected_move:+.2f}%, Conf: {confidence:.1%})")
                    
                    # Affichage en temps réel
                    metrics = portfolio.get_performance_metrics()
                    status = "READY" if context_full else "FILLING"
                    
                    position_age = ""
                    if portfolio.position.quantity != 0 and portfolio.position.open_time:
                        age = (current_time - portfolio.position.open_time).total_seconds()
                        position_age = f"Age: {age:.1f}s | "
                    
                    expected_move_str = f"Var. attendue: {expected_move:+.2f}% | " if context_full else ""
                    
                    print(f"\rPrix: {current_price:.2f} | "
                          f"Capital: {metrics['final_capital']:.2f} | "
                          f"ROI: {metrics['roi']:.2%} | "
                          f"Position: {portfolio.position.quantity:.4f} | "
                          f"{position_age}"
                          f"{expected_move_str}"
                          f"PnL: {portfolio.position.unrealized_pnl:.2f} | "
                          f"Conf: {confidence:.2%} | "
                          f"Status: {status}", end='')
                
                last_price = current_price
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"\nErreur pendant le tick: {str(e)}")
                await asyncio.sleep(0.1)
                continue
        
        # Fermeture de la position finale
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
                'confidence_threshold': confidence_threshold,
                'min_expected_move': min_expected_move
            }
        }
        
        np.save(f'simulation_results_{timestamp_str}.npy', results)
        print(f"\nRésultats sauvegardés dans simulation_results_{timestamp_str}.npy")
        
        # Plot de la courbe de capital
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

def get_expected_move(self, distribution):
    """
    Calcule la variation attendue en pourcentage basée sur la distribution prédite
    """
    bucket_size = 0.002  # 0.002%
    num_buckets = len(distribution)
    bucket_values = np.linspace(-0.5, 0.5, num_buckets)
    
    # Calcul de la moyenne pondérée
    expected_move = np.sum(bucket_values * distribution)
    return expected_move  # retourne la variation attendue en pourcentage

async def main():
    """Fonction principale"""
    # Initialisation du prédicteur
    predictor = MarketPredictor('best_model.pt')
    
    # Ajout de la méthode get_expected_move à l'instance
    predictor.get_expected_move = get_expected_move.__get__(predictor)
    
    # Lancement de la simulation avec les nouveaux paramètres
    await run_portfolio_simulation(
        predictor,
        duration_seconds=300,  # 5 minutes
        confidence_threshold=0.75,  # 75% de confiance minimum
        min_expected_move=0.3  # 0.3% de variation minimale attendue
    )

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())