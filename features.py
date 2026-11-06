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
        file_modify_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        file_modify_date_str = file_modify_time.strftime("%Y-%m-%d")

        if file_modify_date_str == today_str:
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if df is not None and not df.empty:
                    loaded_from_cache = True
                    print(f"📁 【硬盘缓存命中】成功从本地直接读取 {ticker} 的今日数据。")
            except Exception as e:
                print(f"⚠️ 读取本地缓存失败，转为网络下载: {e}")

    # 3. 如果本地没有或缓存过期，才启动网络下载（降级方案）
    if not loaded_from_cache:
        try:
            print(f"🌐 【网络下载】正在从雅虎财经下载 {ticker} 的最新数据...")
            df = yf.download(
                ticker,
                period=period,
                auto_adjust=True,
                prepost=False,
                threads=False,
                progress=False
            )

            if df is None or df.empty:
                return None

            # 深度降维扁平化
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.to_csv(cache_file)
            print(f"💾 【缓存更新】已将 {ticker} 的最新数据持久化保存到本地硬盘。")

        except Exception as e:
            print(f"❌ 网络下载 {ticker} 严重失败: {e}")
            return None

    # 4. 数据科学特征矩阵工程
    try:
        # 安全清洗：先用前向填充修复可能由于刚收盘产生的 NaN 缺失值
        df = df.ffill().bfill()

        if "Close" in df.columns:
            df["Price"] = df["Close"]
        elif "close" in df.columns:
            df["Price"] = df["close"]
        elif "Adj Close" in df.columns:
            df["Price"] = df["Adj Close"]
        else:
            df["Price"] = df.iloc[:, 0]

        df["Return"] = df["Price"].pct_change()

        # 基础时间序列指标
        df["MA_50"] = df["Price"].rolling(window=50).mean()
        df["MA_250"] = df["Price"].rolling(window=250).mean()

        # 过去10天的最高价和最低价
        df["Rolling_Max_10"] = df["Price"].rolling(window=10).max()
        df["Rolling_Min_10"] = df["Price"].rolling(window=10).min()

        df["Amplitude_10"] = (df["Rolling_Max_10"] - df["Rolling_Min_10"]) / df["MA_50"]

        # 长期对数线性趋势与 Z-Score
        log_prices = np.log(df["Price"].values.astype(float))
        x = np.arange(len(log_prices))

        slope, intercept = np.polyfit(x, log_prices, 1)
        fitted_log_prices = slope * x + intercept

        df["Log_Trend_Price"] = np.exp(fitted_log_prices)
        
        res_std = np.std(log_prices - fitted_log_prices)
        if res_std == 0:
            res_std = 0.0001
            
        df["Z_Score"] = (log_prices - fitted_log_prices) / res_std

        # 🚀 【关键修正】：只在最后提取有效行，绝对不盲目 dropna 毁掉全表
        # 250日均线需要250个起点数据，所以我们切片保留有均线数据的后半段
        df = df.iloc[250:].copy()
        
        print(f"⚙️ [DEBUG] {ticker} 价格特征对齐成功，当前最新真实价格为: {df['Price'].iloc[-1]:.2f}")
        print(f"✅ [DEBUG] {ticker} 全部特征矩阵计算大获成功！矩阵最终形状为: {df.shape}")
        return df

    except Exception as e:
        print(f"❌ [DEBUG 崩溃] 特征工程在计算 {ticker} 时遇到了逻辑错误: {e}")
        return None
