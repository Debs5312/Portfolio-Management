import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.optimize import minimize
import ipywidgets as ipw
from IPython.display import display

def drawdown(data):
    """
        Function takes a series of wealth
        and Computes and returns a dataframe
        Dataframe Contains:-
        1.Wealth index
        2.Previous Peaks
        3.percentage Drawdowns
        
    """
    
    wealth_index = 1000*(1+data).cumprod() 
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "Drawdowns": drawdowns
    })

def ret_dframe_drawdown(path: str, pct_company: int):
    """
    DataFrame For drawdown 
    
    """
    max_draw = pd.read_csv(path,
                       header = 0, 
                       index_col = 0, 
                       parse_dates=True, 
                       na_values = -99.99)
    
    columns = ["Lo "+f"{pct_company}","Hi "+f"{pct_company}"]
    data = max_draw[columns]
    data.columns = ["SmallCap","LargeCap"]
    data = data/100
    data.index = pd.to_datetime(data.index, format="%Y%m")
    data.index = data.index.to_period("M")
    return data

def ret_dframe_devNorm(path: str):
    data = pd.read_csv(path,
                       header = 0, 
                       index_col = 0, 
                       parse_dates=True, 
                       na_values = -99.99)
    
    data = data/100
    data.index = data.index.to_period("M")
    return data

def get_ind_file(filetype):
    """
    Load and format the Ken French 30 Industry Portfolios files
    """
    known_types = ["returns", "nfirms", "size"]
    if filetype not in known_types:
        sep = ','
        raise ValueError(f'filetype must be one of:{sep.join(known_types)}')
    if filetype is "returns":
        name = "vw_rets"
        divisor = 100
    elif filetype is "nfirms":
        name = "nfirms"
        divisor = 1
    elif filetype is "size":
        name = "size"
        divisor = 1
    ind = pd.read_csv(f"data/ind30_m_{name}.csv", header=0, index_col=0)/divisor
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_returns():
    """
    Load and format the Ken French 30 Industry Portfolios Value Weighted Monthly Returns
    """
    return get_ind_file("returns")

def get_ind_nfirms():
    """
    Load and format the Ken French 30 Industry Portfolios Average number of Firms
    """
    return get_ind_file("nfirms")

def get_ind_size():
    """
    Load and format the Ken French 30 Industry Portfolios Average size (market cap)
    """
    return get_ind_file("size")

                         
def get_total_market_index_returns():
    """
    Load the 30 industry portfolio data and derive the returns of a capweighted total market index
    """
    ind_nfirms = get_ind_nfirms()
    ind_size = get_ind_size()
    ind_return = get_ind_returns()
    ind_mktcap = ind_nfirms * ind_size
    total_mktcap = ind_mktcap.sum(axis=1)
    ind_capweight = ind_mktcap.divide(total_mktcap, axis="rows")
    total_market_return = (ind_capweight * ind_return).sum(axis="columns")
    return total_market_return

def industry_dframe():
    """
    Return Industry return Dataframe
    df = pd.read_csv("data/ind30_m_vw_rets.csv",
                header=0, index_col=0, parse_dates=True)
    df = df/100
    df.index = pd.to_datetime(df.index, format="%Y%m")
    df.index = df.index.to_period("M")
    df.columns = df.columns.str.strip()
    return df
    
    """
    
    df = pd.read_csv("data/ind30_m_vw_rets.csv",
                header=0, index_col=0, parse_dates=True)
    df = df/100
    df.index = pd.to_datetime(df.index, format="%Y%m")
    df.index = df.index.to_period("M")
    df.columns = df.columns.str.strip()
    return df

def industry_dframe_MarketSize():
    """
    df = pd.read_csv("data/ind30_m_size.csv",
                header=0, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, format="%Y%m")
    df.index = df.index.to_period("M")
    df.columns = df.columns.str.strip()
    return df
    
    """
    
    df = pd.read_csv("data/ind30_m_size.csv",
                header=0, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, format="%Y%m")
    df.index = df.index.to_period("M")
    df.columns = df.columns.str.strip()
    return df

def industry_dframe_firmsNo():
    """
    df = pd.read_csv("data/ind30_m_nfirms.csv",
                header=0, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, format="%Y%m")
    df.index = df.index.to_period("M")
    df.columns = df.columns.str.strip()
    return df
    
    """
    
    df = pd.read_csv("data/ind30_m_nfirms.csv",
                header=0, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, format="%Y%m")
    df.index = df.index.to_period("M")
    df.columns = df.columns.str.strip()
    return df

def semideviation(r):
    """
    This function measures semideviation aka negative semideviation for r
    where r is a pandas series or Dataframe.
    
    """
    is_negative = r < 0
    return r[is_negative].std(ddof=0)

def Skewness(r):
    """
    This is an alternative for Scipy.stats.skew()
    
    """
    dmeaned_r = r - r.mean()
    # use population Standard Deviation so we set ddof = 0
    sigma_r = r.std(ddof=0)
    exp = (dmeaned_r**3).mean()
    return exp/sigma_r**3

def Kurtosis(r):
    """
    This is an alternative for Scipy.stats.skew()
    
    """
    dmeaned_r = r - r.mean()
    # use population Standard Deviation so we set ddof = 0
    sigma_r = r.std(ddof=0)
    exp = (dmeaned_r**4).mean()
    return exp/sigma_r**4

def is_normal(r,level = 0.01):
    
    """
        We perform Jaque Bera test to determine whether a distribution is Normal
        or not and set a level of 1% for cutoff and return boolean value
        True or False
        
    """
    statistic,p_value = st.jarque_bera(r)
    return p_value > level

def plot_year(path: str, pct_company:int, start: int, end: int, size="LargeCap"):
    """
    Plotting yearwise Multiple Subplot to show the Drawdown For each and every case
    in a single plot
    
    """
    dataframe = drawdown(path, pct_company, size = size)
    diff = end - start
    if diff%2 == 0:
        nrow = diff//2
    else:
        nrow = diff//2 + 1
    
    fig, axes = plt.subplots(nrows=nrow, ncols=2)
    fig.set_figheight(diff*2)
    fig.set_figwidth(15)
    for i in range(nrow):
        for j in range(2):
            dataframe[["Wealth","Peaks"]][f"{start}"].plot(ax=axes[i,j])
            start = start+1
                
    plt.show()
    
def var_historic(r, level=5):
    """
    VaR Historic
    
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return np.percentile(r, level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")


def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
        
def var_gaussian(r, level=5, modified=False):
    """
    Returns the parametric gaussian Var of a series or DataFrame
    If modified is True then var_gaussian return modified var
    considering Cornish-Fisher Modification
    
    """
    #Compute the Z score assuming it was Gaussian
    z = st.norm.ppf(level/100)
    
    if modified:
        #compute Z value based on observed Skewness and Kurtosis
        s = Skewness(r)
        k = Kurtosis(r)
        z = (z + (z**2 - 1)*s/6 + (z**3 - 3*z)*(k-3)/24 + (2*z**3 - 5*z)*(s**2)/36)
        
    return -(r.mean()+z*r.std(ddof=0))

def periodic_returns(df, period: int):
    """
    Computes Monthly,Annualy,Quaterly Returns 
    
    """
    compound_growth = (1+df).prod()
    n_periods = df.shape[0]
    return compound_growth**(period/n_periods)-1

def periodic_volatility(df, period: int):
    """
    Computes Volatility of a priod(Annual: period(12), monthly: period(1), quaterly: period(4))
    
    """
    return df.std()*(period**0.5)

def sharp_ratio(df, period_per_year: int, risk_free_rate: float):
    '''
    computes Sharp ratio for a set of return
    
    '''
    #compute annual riskfree_rate to periodic value
    rfree_per_period = (1+risk_free_rate)**(1/period_per_year)-1
    excess_rate = df - rfree_per_period
    annual_excess_rate = periodic_returns(excess_rate, period_per_year)
    annualized_vol = periodic_volatility(df, period_per_year)
    return annual_excess_rate/annualized_vol
        
def portfolio_return(weights, returns):
    """
    weights -> returns
    
    """
    return weights.T @ returns

def portfolio_volatility(weights, cov):
    """
    weights -> vol
    
    """
    return (weights.T @ cov @ weights)**0.5

def plot_2Asset_Frontier(n_points: int, asset_ret, cov, style=".-"):
    """
    The function helps to plot 2 Asset Frontier For the below Given Arguments,
    
    Arguments:
        1.no of points aka no of weights we are trying to plot
        2.asset_ret is a dataframe made using Edhec_risk_kit library functions i.e(industry_dframe() and periodic returns())
                asset_ret is a dataframe for two columns.(*required) 
        3.cov is the covarience matrix calculated using df.cov()
                where df = erk.industry_dframe()[mention time bound(*optional)]
                
    """
    l = asset_ret.shape[0]
    
    if l != 2:
        raise ValueError(" Expected length of input dataframe consists of Two columns as it is a 2 Asset Frontier plotting")
    
    weights = [np.array([w,1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, asset_ret) for w in weights]
    vols = [portfolio_volatility(w, cov) for w in weights]
    data = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })
    data.plot.line(x=data.columns[1], y=data.columns[0], style=style)
    plt.show()
  
    
def minimize_vol(target_ret, er, cov):
    """
    target return -> weight vector
    minimize voltility function for n weights
    
    """
    n = er.shape[0]
    init_guess =  np.repeat(1/n,n)
    bounds = ((0.0, 1.0),)*n
    return_is_target = {
        "type": "eq",
        "args": (er,),
        "fun": lambda weights, er: target_ret - portfolio_return(weights,er)
    }
    weights_sum_to_1 = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights)-1
    }
    results = minimize(portfolio_volatility,
                       init_guess,
                       args = (cov,),
                       method = "SLSQP",
                       options = {"disp": False},
                       constraints = (return_is_target,weights_sum_to_1),
                       bounds = bounds
                      )
    return results.x

def optimal_weights(n_points, er, cov):
    """
    This Functions generates optimal weights for all possible target return
    
    """
    target_ret = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_ret]
    return weights

def msr(risk_free_rate, er, cov):
    """
    msr: Maximum Sharp Ratio for N asset frontier
    Input:
        1.risk_free_rate
        2.er-> Expected Return(ind = erk.industry_dframe(),er = erk.periodic_returns(ind["1996":"2000"],12),er = er[assets])
        3.cov ->Covarience of all pickedup assets(cov = ind["1996":"2000"].cov(),cov = cov.loc[assets,assets])
    Output:
        weights that maximize sharp ratio
    
    """
    n = er.shape[0]
    init_guess =  np.repeat(1/n,n)
    bounds = ((0.0, 1.0),)*n
    weights_sum_to_1 = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights)-1
    }
    def negative_sharp_ratio(weights,risk_free_rate,er,cov):
        """
        Calculates Negative Sharp ratio For Assets
        Input:
            1.weights
            2.riskfree_rate
            3.cov
        Output:
            negtive sharp ratio
            
        """
        r = portfolio_return(weights,er)
        vol = portfolio_volatility(weights, cov)
        return -(r-risk_free_rate)/vol
    
    results = minimize(negative_sharp_ratio,
                       init_guess,
                       args = (risk_free_rate,er,cov,),
                       method = "SLSQP",
                       options = {"disp": False},
                       constraints = (weights_sum_to_1),
                       bounds = bounds
                      )
    return results.x

def GMV(cov):
    """
    return Global Minimum Varience
    
    """
    n = cov.shape[0]
    return msr(0,np.repeat(1/n,n),cov)
    
def plot_NAsset_Frontier(n_points: int, asset_ret, cov, style=".-",show_CML = False, risk_free_rate = 0, show_ew = False, show_GMV=False):
    """
    The function helps to plot N Asset Frontier For the below Given Arguments,
    
    Arguments:
        1.no of points aka no of weights we are trying to plot
        2.asset_ret is a dataframe made using Edhec_risk_kit library functions i.e(industry_dframe() and periodic returns())
                asset_ret is a dataframe for N columns.(*required) 
        3.cov is the covarience matrix calculated using df.cov().loc[columns,columns]
                where df = erk.industry_dframe()[mention time bound(*optional)]
        Optional:
            1.style of line plot.(*default=".-")
            2.show_CMl: to plot a Frontier curve with maximum Sharp ratio line (*default=False)
            3.risk_free_rate = For calculating Sharp ratio(*default=0)
            4.show_ew = Display eqully weighted portfolio 
    """
    
    weights = optimal_weights(n_points, asset_ret, cov)
    rets = [portfolio_return(w, asset_ret) for w in weights]
    vols = [portfolio_volatility(w, cov) for w in weights]
    data = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })
    ax = data.plot.line(x=data.columns[1], y=data.columns[0], style=style)
    
    if show_ew:
        n = asset_ret.shape[0]
        w = np.repeat(1/n,n)
        ret = portfolio_return(w, asset_ret)
        vol = portfolio_volatility(w, cov)
        ax.plot([vol],[ret], color="goldenrod", marker="o", markersize=5)
    
    if show_GMV:
        w_gmv = GMV(cov)
        r_gmv = portfolio_return(w_gmv, asset_ret)
        vol_gmv = portfolio_volatility(w_gmv, cov)
        ax.plot([vol_gmv],[r_gmv], color="midnightblue", marker="o", markersize=10)
        
    if show_CML:
        ax.set_xlim(left = 0)
        w_msr = msr(risk_free_rate, asset_ret, cov)
        r_msr = portfolio_return(w_msr, asset_ret)
        vol_msr = portfolio_volatility(w_msr, cov)
        #Add CML(Capital Market Limit)
        cml_x = [0, vol_msr]
        cml_y = [risk_free_rate,r_msr]
        ax.plot(cml_x,cml_y,color="green",marker="o",linestyle="dashed",markersize=12,linewidth=2)
    return ax

def CPPI(risky_return, safe_return = None, multiplier=3, start=1000, floor=0.8, risk_free_rate = 0.03, drawdown=None):
    """
    The Function takes a set of returns for Risky asset as an input and computes CPPI and 
    returns a Dictionary Containing: Asset value History, Risky weight history, Risk Budget History
    ##     result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risky Budget": cusion_history,
        "Risky Allocation": risky_weight_history,
        "multiplier": m,
        "start": start,
        "floor": floor,
        "Risky return": risky_return,
        "Safe return": safe_return
    }
    
    """
    dates = risky_return.index
    n_steps = len(dates)
    account_value = start
    #floor = 0.8 # 0.8 means 80% of  account value will be considered as a floor value
    floor_value = account_value * floor
    m = multiplier # m-->multiplier for CPPI
    peak = start
    
    if isinstance(risky_return, pd.Series):
        risky_return = pd.DataFrame(risky_return, columns=["R"])
    
    if safe_return is None:
        safe_return = pd.DataFrame().reindex_like(risky_return)
        safe_return[:] = risk_free_rate/12
        
    # History for looking at the changes at each step
    account_history = pd.DataFrame().reindex_like(risky_return)
    cusion_history = pd.DataFrame().reindex_like(risky_return)
    risky_weight_history = pd.DataFrame().reindex_like(risky_return)
    # CPPI Algo
    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1 - drawdown)
        cusion = (account_value-floor_value)/account_value
        risky_weight = m * cusion
        risky_weight = np.minimum(risky_weight,1)
        risky_weight = np.maximum(risky_weight,0)
        safe_weight = 1 - risky_weight
        risky_alloc = risky_weight * account_value
        safe_alloc = safe_weight * account_value
        # Update the account value For each time step
        account_value = risky_alloc * (1 + risky_return.iloc[step]) + safe_alloc * (1 + safe_return.iloc[step])
        # Save the value to a dataframe To look into it in future
        cusion_history.iloc[step] = cusion
        account_history.iloc[step] = account_value
        risky_weight_history.iloc[step] = risky_weight
    
    risky_wealth = start*(1+risky_return).cumprod()  
    result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risky Budget": cusion_history,
        "Risky Allocation": risky_weight_history,
        "multiplier": m,
        "start": start,
        "floor": floor,
        "Risky return": risky_return,
        "Safe return": safe_return
    }
    
    return result

def summary_stats(r, risk_free_rate = 0.03):
    """
    returns a dataframe for all the aggregated Summary Stuff For the return of the columns of r
    
    """
    ann_ret = r.aggregate(periodic_returns, period=12)
    ann_vol = r.aggregate(periodic_volatility, period=12)
    ann_sr = r.aggregate(sharp_ratio, risk_free_rate=risk_free_rate, period_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdowns.min())
    skew = r.aggregate(Skewness)
    kurt = r.aggregate(Kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_var5 = r.aggregate(var_historic)
    return pd.DataFrame({
        "Annualized Return": ann_ret,
        "Annualized Volatility": ann_vol,
        "Sharp Ratio": ann_sr,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher Var(5%)": cf_var5,
        "Historic CVar(5%)": hist_var5,
        "Max Drawdown": dd
    })

def gbm(n_years = 10, n_scenarios = 1000, mu=0.07, sigma=0.15, steps_per_year = 12,  s_0=100.0, prices=True):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return ret_val

def show_cppi(n_scenarios=50, mu=0.07, sigma=0.15, m=3, floor=0.8, risk_free_rate=0.03, y_max=100):
    """
    Plot the results of a Monte Carlo Simulation of CPPI
    
    """
    start = 100
    risky_r = gbm(n_scenarios=n_scenarios, mu=mu, sigma = sigma, s_0 = start, prices=False)
    risky_ret = pd.DataFrame(risky_r)
    btr = CPPI(risky_ret, risk_free_rate=risk_free_rate, multiplier=m, start=start, floor=floor)
    wealth = btr["Wealth"]
    
    # Calculate terminal wealth Stats
    terminal_wealth = wealth.iloc[-1]
    y_max = wealth.values.max()*y_max/100
    
    tw_mean = terminal_wealth.mean()
    tw_median = terminal_wealth.median()
    tw_max = terminal_wealth.max()
    tw_min = terminal_wealth.min()
    tw_diff = tw_max - tw_min
    faliure_mask = np.less(terminal_wealth, start*floor)
    n_faliurs = faliure_mask.sum()
    p_fail = n_faliurs/n_scenarios
    
    e_shortfall = np.dot(terminal_wealth-start*floor, faliure_mask)/n_faliurs if n_faliurs > 0 else 0.0
    
    # plot a frame For showing histogram for the arrangements ofthe number of component returns in a range
    fig, (wealth_ax, hist_ax) = plt.subplots(nrows=1, ncols=2, sharey=True, gridspec_kw={'width_ratios':[3,2]},figsize=(24,9))
    plt.subplots_adjust(wspace=0.0)
    
    wealth.plot(ax=wealth_ax, legend=False, alpha=0.3, color="indianred", figsize=(24,9))
    wealth_ax.axhline(y = start, ls=":", color="black",linewidth=3)
    wealth_ax.axhline(y = start*floor, ls="--", color="red")
    wealth_ax.set_ylim(top=y_max)
    
    terminal_wealth.plot.hist(ax=hist_ax, bins=50, ec="w", fc="indianred", orientation="horizontal")
    hist_ax.axhline(y = start, ls=":", color="black",linewidth=3)
    hist_ax.annotate(f"Mean: ${int(tw_mean)}", xy=(.7, .9),xycoords='axes fraction', fontsize=24)
    hist_ax.annotate(f"Median: ${int(tw_median)}", xy=(.7, .85),xycoords='axes fraction', fontsize=24)
    hist_ax.annotate(f"Difference: ${int(tw_diff)}", xy=(.7, .75),xycoords='axes fraction', fontsize=24)
    hist_ax.annotate(f"Worst case: ${int(tw_min)}", xy=(.7, .65),xycoords='axes fraction', fontsize=24)
    if (floor > 0.01):
        hist_ax.axhline(y=start*floor, ls="--", color="red", linewidth=3)
        hist_ax.annotate(f"Violations: {n_faliurs} ({p_fail*100:2.2f}%)\nE(shortfall)=${e_shortfall:2.2f}", 
                         xy=(.7, .55), xycoords='axes fraction', fontsize=20)
        
def dynamic_CPPI_plot():
    cppi_controls = ipw.interactive(show_cppi,
                                    n_scenarios = ipw.IntSlider(min=1, max=1000, step=5, value=50),
                                    mu = (0, +0.2, 0.01),
                                    sigma = (0, 0.3, 0.05),
                                    floor = (0, 2., 0.1),
                                    m = (1, 5, 0.5),
                                    risk_free_rate = (0, 0.05, 0.01),
                                    y_max = ipw.IntSlider(min=0, max=100, step=1, value=100,
                                                          description="Zoom Y Axis")
                                   )
    display(cppi_controls)

def discount(t, r):
    """
    Compute the price of a pure discount bond that pays a dollar at time period t
    and r is the per-period interest rate
    returns a |t| x |r| Series or DataFrame
    r can be a float, Series or DataFrame
    returns a DataFrame indexed by t
    """
    discounts = pd.DataFrame([(r+1)**-i for i in t])
    discounts.index = t
    return discounts

def pv(flows, r):
    """
    Compute the present value of a sequence of cash flows given by the time (as an index) and amounts
    r can be a scalar, or a Series or DataFrame with the number of rows matching the num of rows in flows
    """
    dates = flows.index
    discounts = discount(dates, r)
    return discounts.multiply(flows, axis='rows').sum()


def inst_to_ann(r):
    """
    Convert an instantaneous interest rate to an annual interest rate
    """
    return np.expm1(r)

def ann_to_inst(r):
    """
    Convert an instantaneous interest rate to an annual interest rate
    """
    return np.log1p(r)

def cir(n_years = 10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
    """
    Generate random interest rate evolution over time using the CIR model
    b and r_0 are assumed to be the annualized rates, not the short rate
    and the returned values are the annualized rates as well
    """
    if r_0 is None: r_0 = b 
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    num_steps = int(n_years*steps_per_year) + 1 # because n_years might be a float
    
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0

    ## For Price Generation
    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    ####

    def price(ttm, r):
        _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years, r_0)
    ####
    
    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        # generate prices at time t as well ...
        prices[step] = price(n_years-step*dt, rates[step])

    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    ### for prices
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    ###
    return rates, prices

def show_cir_prices(r_0=0.03, a=0.5, b=0.03, sigma=0.05, n_scenarios=5):
    cir(r_0=r_0, a=a, b=b, sigma=sigma, n_scenarios=n_scenarios)[1].plot(legend=False, figsize=(12,5))

    controls = ipw.interactive(show_cir_prices,
                               r_0 = (0, .15, .01),
                               a = (0, 1, .1),
                               b = (0, .15, .01),
                               sigma= (0, .1, .01),
                               n_scenarios = (1, 100))
    display(controls)
    
def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    """
    Returns the series of cash flows generated by a bond,
    indexed by the payment/coupon number
    """
    n_coupons = round(maturity*coupons_per_year)
    coupon_amt = principal*coupon_rate/coupons_per_year
    coupon_times = np.arange(1, n_coupons+1)
    cash_flows = pd.Series(data=coupon_amt, index=coupon_times)
    cash_flows.iloc[-1] += principal # add the principal to the last payment
    return cash_flows

def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    """
    Computes the price of a bond that pays regular coupons until maturity
    at which time the principal and the final coupon is returned
    This is not designed to be efficient, rather,
    it is to illustrate the underlying principle behind bond pricing!
    If discount_rate is a DataFrame, then this is assumed to be the rate on each coupon date
    and the bond value is computed over time.
    i.e. The index of the discount_rate DataFrame is assumed to be the coupon number
    """
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price(maturity-t/coupons_per_year, principal, coupon_rate, coupons_per_year,
                                      discount_rate.loc[t])
        return prices
    else: # base case ... single time period
        if maturity <= 0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return pv(cash_flows, discount_rate/coupons_per_year)
    
def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    """
    Computes the total return of a Bond based on monthly bond prices and coupon payments
    Assumes that dividends (coupons) are paid out at the end of the period (e.g. end of 3 months for quarterly div)
    and that dividends are reinvested in the bond
    """
    coupons = pd.DataFrame(data = 0, index=monthly_prices.index, columns=monthly_prices.columns)
    t_max = monthly_prices.index.max()
    pay_date = np.linspace(12/coupons_per_year, t_max, int(coupons_per_year*t_max/12), dtype=int)
    coupons.iloc[pay_date] = principal*coupon_rate/coupons_per_year
    total_returns = (monthly_prices + coupons)/monthly_prices.shift()-1
    return total_returns.dropna()


def macaulay_duration(flows, discount_rate):
    """
    Computes the Macaulay Duration of a sequence of cash flows, given a per-period discount rate
    """
    discounted_flows = discount(flows.index, discount_rate)*pd.DataFrame(flows)
    weights = discounted_flows/discounted_flows.sum()
    return np.average(flows.index, weights=weights.iloc[:,0])

def match_durations(cf_t, cf_s, cf_l, discount_rate):
    """
    Returns the weight W in cf_s that, along with (1-W) in cf_l will have an effective
    duration that matches cf_t
    """
    d_t = macaulay_duration(cf_t, discount_rate)
    d_s = macaulay_duration(cf_s, discount_rate)
    d_l = macaulay_duration(cf_l, discount_rate)
    return (d_l - d_t)/(d_l - d_s)

def funding_ratio(assets, liabilities, r):
    """
    Computes the funding ratio of a series of liabilities, based on an interest rate and current value of assets
    """
    return pv(assets, r)/pv(liabilities, r)

def bt_mix(r1, r2, allocator, **kwargs):
    """
    Runs a back test (simulation) of allocating between a two sets of returns
    r1 and r2 are T x N DataFrames or returns where T is the time step index and N is the number of scenarios.
    allocator is a function that takes two sets of returns and allocator specific parameters, and produces
    an allocation to the first portfolio (the rest of the money is invested in the GHP) as a T x 1 DataFrame
    Returns a T x N DataFrame of the resulting N portfolio scenarios
    """
    if not r1.shape == r2.shape:
        raise ValueError("r1 and r2 should have the same shape")
    weights = allocator(r1, r2, **kwargs)
    if not weights.shape == r1.shape:
        raise ValueError("Allocator returned weights with a different shape than the returns")
    r_mix = weights*r1 + (1-weights)*r2
    return r_mix


def fixedmix_allocator(r1, r2, w1, **kwargs):
    """
    Produces a time series over T steps of allocations between the PSP and GHP across N scenarios
    PSP and GHP are T x N DataFrames that represent the returns of the PSP and GHP such that:
     each column is a scenario
     each row is the price for a timestep
    Returns an T x N DataFrame of PSP Weights
    """
    return pd.DataFrame(data = w1, index=r1.index, columns=r1.columns)

def terminal_values(rets):
    """
    Computes the terminal values from a set of returns supplied as a T x N DataFrame
    Return a Series of length N indexed by the columns of rets
    """
    return (rets+1).prod()

def terminal_stats(rets, floor = 0.8, cap=np.inf, name="Stats"):
    """
    Produce Summary Statistics on the terminal values per invested dollar
    across a range of N scenarios
    rets is a T x N DataFrame of returns, where T is the time-step (we assume rets is sorted by time)
    Returns a 1 column DataFrame of Summary Stats indexed by the stat name 
    """
    terminal_wealth = (rets+1).prod()
    breach = terminal_wealth < floor
    reach = terminal_wealth >= cap
    p_breach = breach.mean() if breach.sum() > 0 else np.nan
    p_reach = breach.mean() if reach.sum() > 0 else np.nan
    e_short = (floor-terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan
    e_surplus = (cap-terminal_wealth[reach]).mean() if reach.sum() > 0 else np.nan
    sum_stats = pd.DataFrame.from_dict({
        "mean": terminal_wealth.mean(),
        "std" : terminal_wealth.std(),
        "p_breach": p_breach,
        "e_short":e_short,
        "p_reach": p_reach,
        "e_surplus": e_surplus
    }, orient="index", columns=[name])
    return sum_stats

def glidepath_allocator(r1, r2, start_glide=1, end_glide=0.0):
    """
    Allocates weights to r1 starting at start_glide and ends at end_glide
    by gradually moving from start_glide to end_glide over time
    """
    n_points = r1.shape[0]
    n_col = r1.shape[1]
    path = pd.Series(data=np.linspace(start_glide, end_glide, num=n_points))
    paths = pd.concat([path]*n_col, axis=1)
    paths.index = r1.index
    paths.columns = r1.columns
    return paths

def floor_allocator(psp_r, ghp_r, floor, zc_prices, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    if zc_prices.shape != psp_r.shape:
        raise ValueError("PSP and ZC Prices must have the same shape")
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = floor*zc_prices.iloc[step] ## PV of Floor assuming today's rates and flat YC
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1) # same as applying min and max
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
    return w_history

def drawdown_allocator(psp_r, ghp_r, maxdd, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    ### For MaxDD
    peak_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        ### For MaxDD
        floor_value = (1-maxdd)*peak_value ### Floor is based on Prev Peak
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1) # same as applying min and max
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        ### For MaxDD
        peak_value = np.maximum(peak_value, account_value) ### For MaxDD
        w_history.iloc[step] = psp_w
    return w_history