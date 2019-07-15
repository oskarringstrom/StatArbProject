import numpy as np
import pandas as pd

from datetime import timedelta

from DataStructures import SecurityPair
from UniverseSelection import getTickersToTrade, addStaticUniverse

class OptimizedNadionCircuit(QCAlgorithm):

    def Initialize(self):
        """ Built-in function. Initializes the algorithm.
        """
        self.SetStartDate(2018, 1, 26)  # Set start date for backtest
        self.SetEndDate(2019, 7, 1)  # Set end date for backtest
        self.SetCash(100000)  # Set strategy equity

        self.tradingDaysPerYear = 253 # Approx. number of trading days per year
        self.universalLookbackPeriod = self.tradingDaysPerYear # Used to make sure all arrays etc. have the same dim
        
        self.resolution = Resolution.Daily # Feed the algo daily price updates
        
        addStaticUniverse(self)
        self.UpdateUniverse()
        
    def OnData(self,data):
        """ Built-in function. Gets called every time new data is availible.
        
        Args:
            data  /object contating all availible data for the universe, keyed by ticker
        """
        
        # Plot the liquitity every day
        self.Plot('Liquidity', 'l', float(self.Portfolio.MarginRemaining / self.Portfolio.TotalPortfolioValue))
        
        if len(data.Keys) < self.lenOfStaticUniverse / 2: return # Rudimentary way to avoid trading on weekends

        self.IntraDayUniverseCheck(data) # Short intra-day check of the pairs to trade
        
        # Update and trade each pair
        for pair in self.TradeablePairs:

            pair.update(data)
        
            pair.getPositionsAndExecuteTrades(self)
            
            self.Plot('Z-Score', str(pair.tickers), pair.getZScore())
            
            if pair.pos == 'l': x = 1; 
            if pair.pos == 'f': x = 0; 
            if pair.pos == 's': x = -1; 
            self.Plot('Position', str(pair.tickers), x)

    def IntraDayUniverseCheck(self,data):
        """ Performs a check that all pairs that will be traded can infact be traded,
        i.e. if data is availibe. Deals with events such as delistings etc.
        
        Args:
            data  /object contating all availible data for the universe, keyed by ticker
        """
        pairsToRemove = []
        
        # Check that data is availible for both assets in each pair
        for i, pair in enumerate(self.TradeablePairs):
            
            ta, tb = pair.tickers; keys = [k.Value for k in data.Keys]
            
            if ta not in keys or tb not in keys: # Keys - array with all tickers that are tradeable
            
                pairsToRemove.append((ta,tb))
                self.Log('Removing pair: '+str((ta,tb)))
        
        # Update the tradeable pairs and remove all non tradeable ones
        self.TradeablePairs = [pair for pair in self.TradeablePairs if pair.tickers not in pairsToRemove]
                
    def UpdateUniverse(self):
        """ Main function for running the analysis in Universe Selection module.
        The universe of pairs is currently recalculated on a yearly basis.
        """
        self.Liquidate() # Liquidate all current positions in the old pairs

        self.TickersToTrade, self.BetaParams = getTickersToTrade(self) # Run pairs selection logic (Universe Selection module)
       
        self.Log('New universe:'+str(self.TickersToTrade))
        
        self.TradeablePairs = [SecurityPair(self,pair) for pair in self.TickersToTrade] # Create SecurityPars obj (see Data Structures module)
        
        # Schedule next universe calculation
        nextUpdate = self.Time.date() + timedelta(365)
        
        self.Schedule.On(
            self.DateRules.On(nextUpdate.year,nextUpdate.month,nextUpdate.day),
            self.TimeRules.At(8,0),
            self.UpdateUniverse
        )
