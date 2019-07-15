import numpy as np
import pandas as pd

from TradingLogic import getPosition
from StatisticalAnalysis import coint, adfuller

from statsmodels.regression.linear_model import OLS

def addStaticUniverse(algo):
    """ Adds a static universe will be considered for the selection analysis.
       
        Args:
            algo  /an instance of the QuantConnect algorithm
    """
    
    # S&P 500
    tickers = ['SPY','MSFT','AAPL','AMZN','FB','JNJ','JPM','GOOG','GOOGL','XOM','V','PG','BAC','CSCO','DIS','MA','PFE','UNH','T','CVX','VZ','HD','MRK','INTC','KO','WFC','PEP','CMCSA','BA','C','MCD','NFLX','WMT','ABT','ORCL','ADBE','PYPL','MDT','HON','UNP','IBM','PM','ACN','CRM','TMO','ABBV','COST','AVGO','LIN','AMGN','TXN','NKE','UTX','SBUX','LLY','NEE','MMM','MO','AMT','NVDA','GE','DHR','LMT','QCOM','AXP','GILD','BKNG','USB','MDLZ','LOW','BMY','ADP','CME','ANTM','CAT','CVS','UPS','CHTR','CB','INTU','COP','CELG','GS','BDX','TJX','CL','DUK','PNC','CSX','SYK','D','CI','ISRG','BSX','SPGI','SO','MS','CCI','DD','NSC','BLK','NOC','ZTS','RTN','SCHW','ECL','MMC','SPG','ILMN','PLD','EOG','SLB','ICE','APD','PGR','EXC','DE','GM','GD','AIG','MET','KMB','ITW','AON','TGT','BIIB','WM','VRTX','AEP','COF','EQIX','WBA','PRU','AFL','AMAT','KMI','EMR','AGN','FIS','TRV','BK','MU','ADI','EL','FDX','DOW','SHW','EW','F','ROP','BBT','MAR','BAX','SRE','PSA','ROST','PSX','CTSH','ADSK','OXY','APC','DG','JCI','FISV','ATVI','HCA','DAL','HUM','ETN','SYY','ALL','YUM','WMB','EBAY','RHT','MPC','VLO','STZ','MCO','WELL','TEL','GIS','ORLY','HPQ','IR','XEL','PEG','LRCX','AMD','AVB','EA','NEM','EQR','STI','PAYX','XLNX','ED','AZO','APH','VFC','HLT','TWTR','PPG','OKE','MSI','WEC','ALXN','MNST','DFS','TROW','MCK','LYB','GPN','SBAC','LUV','ZBH','DLTR','GLW','WLTW','PCAR','DLR','PXD','TSN','ES','VRSK','REGN','CERN','CMI','DTE','MTB','HRS','FLT','TDG','FTV','IDXX','VTR','ALGN','CNC','BLL','ADM','A','IQV','INFO','VRSN','PPL','O','PH','SYF','STT','FE','TSS','CCL','RCL','SWK','AWK','FITB','BXP','MSCI','AMP','MCHP','LLL','CLX','MTD','CXO','CTAS','APTV','EIX','HIG','AME','HPE','NTRS','ESS','KR','ROK','MKC','CTVA','CHD','HAL','ETR','ULTA','HSY','SNPS','KHC','KLAC','UAL','FAST','CDNS','RSG','AEE','WY','IP','VMC','CMG','OMC','KEY','RMD','ARE','LH','CBS','FRC','CFG','MXIM','AJG','ANSS','CMS','NUE','COO','EFX','CBRE','KEYS','NTAP','DHI','CINF','EVRG','LEN','BR','FCX','BBY','IFF','FANG','WAT','CPRT','HCP','GPC','L','DRI','HES','CNP','XYL','KMX','TFX','RF','WCG','PFG','EXPE','IT','HBAN','MLM','CAG','DOV','SJM','EXR','MGM','K','HST','INCY','DXC','ABC','MAA','DGX','CE','ANET','CAH','LNC','HOLX','SWKS','AKAM','TSCO','EXPD','UDR','TTWO','XRAY','AAL','GWW','ABMD','CBOE','HAS','SYMC','CTXS','KSU','ATO','BHGE','FOXA','VAR','VNO','NCLH','TXT','LNT','MAS','ETFC','DVN','STX','REG','MRO','WYNN','CMA','DRE','AAP','SIVB','HRL','CHRW','RJF','NDAQ','WDC','AES','APA','FTNT','FMC','PNW','HSIC','NI','COG','UHS','JKHY','PKI','RE','NBL','TAP','URI','BEN','VIAB','EMN','TIF','ALLE','ARNC','NRG','FRT','FTI','CTL','AVY','CF','GRMN','BF.B','JEC','DISCK','MHK','JNPR','WRK','HII','TMK','SNA','WAB','LW','MYL','IRM','WU','PKG','TPR','FFIV','PHM','ZION','LKQ','DISH','BWA','IVZ','WHR','KSS','IPG','QRVO','NOV','AIV','NLSN','CPB','MOS','KIM','ALB','JBHT','SLG','FBHS','ALK','UNM','SEE','FLIR','XRX','PVH','RHI','M','HFC','ADS','FLS','RL','AOS','DVA','COTY','HBI','PBCT','PNR','HOG','XEC','NKTR','NWL','ROL','PRGO','HRB','FOX','HP','PWR','AIZ','CPRI','BMS','LB','LEG','FL','AMG','JEF','IPGP','UAA','TRIP','NWSA','DISCA','UA','GPS','MAC','JWN','NWS']
    algo.lenOfStaticUniverse = len(tickers)
    for ticker in tickers:
        algo.AddEquity(ticker, algo.resolution).FeeModel = ConstantFeeModel(0)

def getTickersToTrade(algo):
    """ Main function for identifying asset-pairs to trade. The function has two
    main filters, that gradually reduces the number of assets considered. First,
    an Augmented Dickey Fuller test is used to identify time-stationary assets.
    Then, all combinations of the stationary assets are tested for cointegration,
    acting as the second filter. The cointegration test used is a version of the
    2 step Engle-Granger. A simple backtest is then simulated for the identified
    pairs, and the best performing ones are selected to be traded.
    
    Args:
       algo       /an instance of the QuantConnect algorithm
       
    Returns:
        universe  /a list of tuples containing the asset-pairs to be traded
    
    """
    beta_params = {} # Used to save the beta parameters from the OLS regression in the adfuller test
    
    # Add all assets to the algorithm and collect their historical price data
    securities = []
    for equity in algo.Securities.Keys:
        
        if equity == 'SPY': continue # SPY used as benchmark
    
        h = algo.History(equity, algo.universalLookbackPeriod, algo.resolution)
        
        if len(h) == algo.universalLookbackPeriod:
            h = h['close']
            securities.append(h)
            
    stationarySecurities = list(filter(HasUnitRoot,securities)) # First filter func, adfuller test
    
    cointegratedPairs,beta_params = GetSecurityPairs(stationarySecurities,beta_params) # Second filter func, Engle-Granger test
    
    # Run a simple backtest of the trading strat for each pair
    pairResults = []
    for i, pair in enumerate(cointegratedPairs):
        
        secA = pair[0]; tickerA = secA.index[0][0] 
        secB = pair[1]; tickerB = secB.index[0][0] 
        
        beta = beta_params[tickerA,tickerB]
        
        spread = secA.values - (beta*secB.values)
        mean = spread.mean(); std = spread.std()
        zscore = (spread - mean) / std
        
        res = simpleBacktest(
            secA,
            secB,
            zscore,
            algo.universalLookbackPeriod
        )
        
        pairResults.append((res,(tickerA,tickerB)))
        
    pairResults.sort(key = lambda x: x[0],reverse = True) # Sort pairs by backtest result
    
    sortedPairs = [x[1] for x in pairResults if x[0] > 0] # Select all pairs with positive result
    
    universe = getUniqueUniverse(sortedPairs) # Filter out all non-unique pairs
    universe = [universe[0:10],beta_params] # Select the top 10 best performing pairs
    
    return universe # Return universe

def HasUnitRoot(series):
    """ Augmented Dickey Fuller test for a timeseries. See Statistical Analysis
    module for more info.
    
    Args:
        series      /a pandas Series object representing the price timeseries of a security
        
    Returns:
        True/False  /bool for if the series is stationary or not
    
    """
    X = series.values # Get price values

    result = adfuller(X) # Execute adfuller test
    pvalue = result[1]
    
    if pvalue <= 0.05: return True # Return True for statistically significant results
    else: return False

def GetSecurityPairs(securityList,beta_params):
    """ Tests all possible asset combinations for cointegration, using the Engle-
    Granger 2 step test. For more info see Statistical Analysis module.
    
    Args:
        securityList  /a list of pandas Series representing price timeseries
        
    Returns:
        pairs        /a list of tuples of all cointegrated pandas Series
        beta_params  /dictionary of beta parameters from OLS regression, keyed by
                      the ticker tuple
    
    """
    pairs = []
    beta_params = {}
    # Interate through all possible combinations
    for x1 in securityList:
        for x2 in securityList:
            
            ticker1 = x1.index[0][0]
            ticker2 = x2.index[0][0]
            
            if ticker1 != ticker2:
                
                result = coint(x1,x2) # Test for cointegration
                pvalue = result[1]
                beta_params[(ticker1,ticker2)] = result[2][0] # Save beta 
                
                if pvalue <= 0.05: # Keep all statistically significant results
                    pairs.append((x1,x2))
                    
    return pairs,beta_params

def simpleBacktest(secA,secB,zscore,length):
    """ Function for running a simplified backtest of the trading strategy.
    
    Args:
        secA      /pandas Series representing price timeseries for first asset
        secB      /pandas Series representing price timeseries for second asset
        zscore    /pandas Series representing zscore timeseries for the pair
        length    /number of iterations (days) to backtest for
        
    Returns:
        res[-1]   /the absolute return for the backtest
    """
    res = np.ones(length) # Empty vector to store results in
    pos = 'f'; lastPos = pos 
    for t in range(length):
        
        zs = float(zscore[t]) # Current zscore
        
        # Today's and yesterdays close-prices 
        a = secA[t]; ay = secA[t-1] 
        b = secB[t]; by = secB[t-1]
        
        s = float(a - b) # Value of todats spread

        if t == 0: continue  

        pos = getPosition(zs,pos) # Get current position for the pair

        res[t] = res[t-1] # default flat position

        # open short position
        if pos == 's':
            res[t] -= (a-b)-(ay-by)

        # open long position
        if pos == 'l':
            res[t] += (a-b)-(ay-by)
    
    return res[-1]
    
def getUniqueUniverse(pairs):
    """ Removers all non-unique ticker pairs from a list.
    
    Args:
        pairs     /a list of tuples containing tickers
        
    Returns:
        universe  /a list of tuples containing the asset-pairs to be traded
    """
    tickersInUniverse = []
    universe = []
    for pair in pairs:
        ta, tb = pair
        if ta not in tickersInUniverse and tb not in tickersInUniverse:
            universe.append((ta,tb)); tickersInUniverse.append(ta), tickersInUniverse.append(tb)
    
    return universe
