import predictor
import alpaca

predictor = predictor.Predictor()
buys = predictor.getPredictions()
maxPosition = 400000/len(buys)
trader = alpaca.AlpacaTrader()

for stock in buys:
    price = trader.getPrice(stock)
    qty = maxPosition//price
    trader.makeTrade(stock, qty)