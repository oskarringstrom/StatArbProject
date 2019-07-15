def getPosition(zs,currentPos):
        """ Generates a position for the spread according to the strategy.
        Used both in the universe selection and live trading.
        
        Args:
            zs          /Zscore to evaluate
            currentPos  /current spread-position for the pair (long/flat/short)
        """
        newPos = currentPos
        
        # determine when to open position
        if zs > 2: newPos = 's'
        
        if zs < -2: newPos = 'l'
            
        # determine when to close position
        if zs < 0.75 and currentPos == 's': newPos = 'f'
        
        if zs > 0.5 and currentPos == 'l': newPos = 'f'
        
        return newPos
        
def executeTrade(algo,pos,assetTickers,assetPrices):
    """ Execute trades according to the position, with QC's builtin trading logic.
    
    Args: 
        algo         /instance of the QC algorithm object
        pos          /position to take in the spread
        assetTickers /tickers for the pair to trade
        asserPrices  /current prices for both assets
    """
    weightings = 1 / len(algo.TradeablePairs) # Take equal positions in all assets
    
    a, b = assetPrices; ta, tb = assetTickers
        
    # Adjust position with QC's trading logic
    if pos == 'f':
        algo.SetHoldings(ta,0) 
        algo.SetHoldings(tb,0)

    if pos == 's':
        if a >= b:
            # long a, short b
            algo.SetHoldings(ta,weightings) 
            algo.SetHoldings(tb,-weightings)
        else:
            # short secA, long secB
            algo.SetHoldings(ta,-weightings) 
            algo.SetHoldings(tb,weightings)
                
    if pos == 'l':
        if a >= b:
            # short a, long b
            algo.SetHoldings(ta,-weightings) 
            algo.SetHoldings(tb,weightings)
        else:
            # long secA, short secB
            algo.SetHoldings(ta,weightings) 
            algo.SetHoldings(tb,-weightings)
