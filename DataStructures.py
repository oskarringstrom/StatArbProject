import pandas as pd

from TradingLogic import getPosition, executeTrade

from statsmodels.regression.linear_model import OLS

class SecurityPair():
    """ Datastructure for storing necessary data and values for each tradealble
    asset pair.
    
    Main attributes:
        tickers  /tuple of strings for the ticker-pair
        pos      /string representing the current position in the pair (long/flat/short)
    
    Main functions:
        getPositionsAndExecuteTrades 
                 / Exectutes the trading strategy and open/closes positions in the 
                   spread accordingly
    """
    def __init__(self,algo,tickerTuple):
        
        self.tickers = tickerTuple
        
        self.beta = algo.BetaParams[self.tickers]
        
        self.pos = 'f'; self.posYday = self.pos
        
        # Built-in data structure for storing historic price data
        self.priceWindows = (
            RollingWindow[float](algo.universalLookbackPeriod),
            RollingWindow[float](algo.universalLookbackPeriod)
        )
        
        # Fill arrays upon creation
        for i, window in enumerate(self.priceWindows):
            
            # Request historical price data
            history = algo.History(
                self.tickers[i], 
                algo.universalLookbackPeriod, 
                algo.resolution
            )
            
            # Store price data
            for h in history:
                self.priceWindows[i].Add(h.Close)
                
    def getPositionsAndExecuteTrades(self,algo):
        
        zscore = self.getZScore() # Generate zscore to evaluate
        
        self.pos = getPosition(zscore, self.pos) # Get spread position according to strat
        
        if self.pos == self.posYday: return
        
        executeTrade(
            algo,
            self.pos,
            self.tickers,
            (x[0] for x in self.priceWindows)
        )
        
        self.posYday = self.pos
    
    def getZScore(self):
        """ Function for generating a zscore for the current price spread. 
        Normalizes the spread with historical volatility and mean.
        """
        # Get price series
        A = pd.Series([x for x in self.priceWindows[0]])
        B = pd.Series([x for x in self.priceWindows[1]])
        
        # Get params for the spread
        spread = A - (self.beta*B); mean = spread.mean(); std = spread.std()
        
        zscore = (spread - mean) / std # Generate Zscores
        zs = float(zscore.values[0]) # Select todays
        
        return zs
        
    def update(self,data):
        """ Updates price arrays with most recent price data
        """
        for i, window in enumerate(self.priceWindows):
            window.Add(data[self.tickers[i]].Close)
