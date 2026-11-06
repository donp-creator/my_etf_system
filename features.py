import os
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf


def get_stock_features(ticker, period="10y"):
    """下载特定股票的历史数据（优先读取本地离线CSV缓存），并生成数据科学特征矩阵"""
    # 1. 建立本地硬盘缓存目录
    cache_dir = "cache_data"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    today_str = datetime.now().strftime("%Y-%m-%d")
    cache_file = os.path.join(cache_dir, f"{ticker}_{period}.csv")

    df = None
    loaded_from_cache = False

    # 2. 策略：优先检查本地硬盘是否有今日最新的缓存文件
    if os.path.exists(cache_file):
        # 获取文件的最后修改日期
        file_modify_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        file_modify_date_str = file_modify_time.strftime("%Y-%m-%d")

        # 如果缓存文件是今天的，直接从硬盘读取，0秒加载，彻底免疫限流
        if file_modify_date_str == today_str:
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if not df.empty:
                    loaded_from_cache = True
                    print(f"📁 【硬盘缓存命中】成功从本地读取 {ticker} 的今日数据。")
            except Exception as e:
                print(f"读取本地缓存失败，转为网络下载: {e}")

    # 3. 如果本地没有或缓存过期，才启动网络下载（降级方案）
    if not loaded_from_cache:
        try:
            print(f"🌐 【网络下载】正在从雅虎财经下载 {ticker} 的最新数据...")
            # 显式关闭多线程和进度条，确保云端与本地稳定
            df = yf.download(
                ticker,
                period=period,
                group_by="column",
                auto_adjust=True,
                prepost=False,
                threads=False,
                progress=False,
            )

            if df.empty:
                print(f"❌ 雅虎财经未返回 {ticker} 的数据，可能触发了IP限流。")
                return None

            # 针对 yfinance 返回的 MultiIndex 进行扁平化处理
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 成功下载后，立刻在本地硬盘存一份备份，供今天后续高频刷新使用
            df.to_csv(cache_file)
            print(f"💾 【缓存更新】已将 {ticker} 的最新数据持久化保存到本地硬盘。")

        except Exception as e:
            print(f"网络下载 {ticker} 失败: {e}")
            return None

    # 4. 数据科学特征矩阵工程（基于清洗好的df计算指标）
    try:
        df["Price"] = df["Close"]
        df["Return"] = df["Price"].pct_change()

        # 基础时间序列指标
        df["MA_50"] = df["Price"].rolling(window=50).mean()
        df["MA_250"] = df["Price"].rolling(window=250).mean()

        # 过去10天的最高价和最低价
        df["Rolling_Max_10"] = df["Price"].rolling(window=10).max()
        df["Rolling_Min_10"] = df["Price"].rolling(window=10).min()

        # 滚动振幅
        df["Amplitude_10"] = (df["Rolling_Max_10"] - df["Rolling_Min_10"]) / df[
            "MA_50"
        ]

        # 长期对数线性趋势与 Z-Score
        log_prices = np.log(df["Price"].values)
        x = np.arange(len(log_prices))

        slope, intercept = np.polyfit(x, log_prices, 1)
        fitted_log_prices = slope * x + intercept

        df["Log_Trend_Price"] = np.exp(fitted_log_prices)
        df["Z_Score"] = (log_prices - fitted_log_prices) / np.std(
            log_prices - fitted_log_prices
        )

        df = df.dropna()
        return df

    except Exception as e:
        print(f"特征工程计算错误 {ticker}: {e}")
        return None
