import numpy as np
import pandas as pd
import yfinance as yf


def get_stock_features(ticker, period="10y"):
    """下载特定股票的历史数据，并生成基础的数据科学特征矩阵"""
    try:
        # 1. 下载数据（默认拉取10年，你也可以在app里改）
                # 显式关闭多线程和进度条，确保云端 Linux 环境 100% 稳定下载
        df = yf.download(ticker, period=period, group_by='column', auto_adjust=True, prepost=False, threads=False, progress=False)


        if df.empty:
            return None

        # 针对 yfinance 返回的 MultiIndex 进行扁平化处理
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 2. 基础价格特征
        df["Price"] = df["Close"]
        df["Return"] = df["Price"].pct_change()

        # 3. 基础时间序列指标（滚动均线与振幅）
        df["MA_50"] = df["Price"].rolling(window=50).mean()
        df["MA_250"] = df["Price"].rolling(window=250).mean()

        # 过去10天的最高价和最低价（用于你说的“过去X天有过Y behavior”的形态捕捉）
        df["Rolling_Max_10"] = df["Price"].rolling(window=10).max()
        df["Rolling_Min_10"] = df["Price"].rolling(window=10).min()

        # 滚动振幅（用来衡量是否“横盘”）
        df["Amplitude_10"] = (df["Rolling_Max_10"] - df["Rolling_Min_10"]) / df[
            "MA_50"
        ]

        # 4. 长期对数线性趋势与 Z-Score (实现你最初的翻倍通道想法)
        log_prices = np.log(df["Price"].values)
        x = np.arange(len(log_prices))

        # 线性回归拟合对数价格
        slope, intercept = np.polyfit(x, log_prices, 1)
        fitted_log_prices = slope * x + intercept

        # 计算残差的标准差和当前的 Z-Score偏离度
        residuals = log_prices - fitted_log_prices
        std_dev = np.std(residuals)

        df["Log_Trend_Price"] = np.exp(fitted_log_prices)
        df["Z_Score"] = (log_prices - fitted_log_prices) / std_dev

        # 只保留有完整计算数据的行
        df = df.dropna()
        return df

    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None
