import os
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

def get_stock_features(ticker, period="10y"):
    """下载历史时序数据，采用双重保险算法交叉合并财务估值指标，完美适配新上市拆分巨头与虚拟币信托"""
    # 1. 建立离线硬盘缓存
    cache_dir = "cache_data"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    today_str = datetime.now().strftime("%Y-%m-%d")
    cache_file = os.path.join(cache_dir, f"{ticker}_{period}.csv")

    df = None
    loaded_from_cache = False

    if os.path.exists(cache_file):
        file_modify_date_str = datetime.fromtimestamp(os.path.getmtime(cache_file)).strftime("%Y-%m-%d")
        if file_modify_date_str == today_str:
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if df is not None and not df.empty:
                    loaded_from_cache = True
                    print(f"📁 【硬盘缓存命中】成功直接读取 {ticker} 的今日数据。")
            except Exception as e:
                print(f"⚠️ 读取本地缓存失败: {e}")

    # 2. 时序数据下载
    if not loaded_from_cache:
        try:
            print(f"🌐 【网络下载】正在从雅虎财经下载 {ticker} 的时序数据...")
            df = yf.download(ticker, period=period, auto_adjust=True, prepost=False, threads=False, progress=False)
            if df is None or df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.to_csv(cache_file)
        except Exception as e:
            print(f"❌ 网络下载时序失败 {ticker}: {e}")
            return None

    # 3. 🚀【双重保险因子注入】：全面防御新上市拆分资产的市值断层
    valuation_metrics = {
        "市值": "-", "Trailing_PE": "-", "Forward_PE": "-", 
        "PEG_Ratio": "-", "市现率_PCF": "-", "自由现金流收益率": "-"
    }
    
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        f_info = ticker_obj.fast_info
        
        # 🚀【第一道保险】：先用快速快照提取市值
        m_cap = 0
        if f_info and "market_cap" in f_info and f_info["market_cap"] > 0:
            m_cap = f_info["market_cap"]
            
        # 🚀【第二道保险】：如果快速快照是空，或者新资产没记录，强行降级去读取完整的财务结构大表
        if m_cap == 0 and info and "marketCap" in info:
            m_cap = info["marketCap"]
            
        # 格式化输出市值
        if m_cap and m_cap >= 1e12:
            valuation_metrics["市值"] = f"${m_cap / 1e12:.2f}T"
        elif m_cap and m_cap >= 1e9:
            valuation_metrics["市值"] = f"${m_cap / 1e9:.2f}B"
        elif m_cap and m_cap > 0:
            valuation_metrics["市值"] = f"${m_cap / 1e6:.1f}M"
                
        # 抓取财务多重估值乘数
        if info:
            if info.get("trailingPE"): valuation_metrics["Trailing_PE"] = round(info["trailingPE"], 1)
            if info.get("forwardPE"): valuation_metrics["Forward_PE"] = round(info["forwardPE"], 1)
            if info.get("pegRatio"): valuation_metrics["PEG_Ratio"] = round(info["pegRatio"], 2)
            if info.get("priceToCashFlow"): valuation_metrics["市现率_PCF"] = round(info["priceToCashFlow"], 1)
            
            # 自由现金流收益率
            fcf = info.get("freeCashflow") or info.get("operatingCashflow", 0)
            if fcf and m_cap and m_cap > 0:
                fcf_yield = (fcf / m_cap) * 100
                if fcf_yield > 0:
                    valuation_metrics["自由现金流收益率"] = f"{fcf_yield:.1f}%"
    except Exception as e:
        print(f"💡 提示：{ticker} 属于指数/大宗商品，自动跳过企业基本面因子注入。")

    # 4. 数据科学特征工程计算
    try:
        df = df.ffill().bfill()
        if "Close" in df.columns: df["Price"] = df["Close"]
        elif "close" in df.columns: df["Price"] = df["close"]
        else: df["Price"] = df.iloc[:, 0]

        df["Return"] = df["Price"].pct_change()
        df["MA_50"] = df["Price"].rolling(window=min(50, len(df))).mean()
        df["MA_250"] = df["Price"].rolling(window=min(250, len(df))).mean()
        df["Rolling_Max_10"] = df["Price"].rolling(window=min(10, len(df))).max()
        df["Rolling_Min_10"] = df["Price"].rolling(window=min(10, len(df))).min()
        df["Amplitude_10"] = (df["Rolling_Max_10"] - df["Rolling_Min_10"]) / df["MA_50"]

        log_prices = np.log(df["Price"].values.astype(float))
        x = np.arange(len(log_prices))
        slope, intercept = np.polyfit(x, log_prices, 1)
        fitted_log_prices = slope * x + intercept
        df["Log_Trend_Price"] = np.exp(fitted_log_prices)
        
        res_std = np.std(log_prices - fitted_log_prices) or 0.0001
        df["Z_Score"] = (log_prices - fitted_log_prices) / res_std
        df = df.ffill().bfill()
        if len(df) > 250:
            df = df.iloc[250:].copy()

        df.attrs["valuation"] = valuation_metrics
        print(f"✅ [DEBUG] {ticker} 特征与双重基本面因子完美熔接。")
        return df

    except Exception as e:
        print(f"❌ 特征计算失败 {ticker}: {e}")
        return None
