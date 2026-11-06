import os
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

def get_stock_features(ticker, period="10y"):
    """下载特定股票/ETF/虚拟币的历史数据（优先读取本地离线CSV缓存），并生成数据科学特征矩阵"""
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
        # 安全清洗：前向+后向填充修复可能存在的 NaN 缺失值
        df = df.ffill().bfill()

        # 兼容性多重保险列名对齐
        if "Close" in df.columns:
            df["Price"] = df["Close"]
        elif "close" in df.columns:
            df["Price"] = df["close"]
        elif "Adj Close" in df.columns:
            df["Price"] = df["Adj Close"]
        else:
            df["Price"] = df.iloc[:, 0]

        df["Return"] = df["Price"].pct_change()

        # 基础时间序列指标：如果数据量太少（比如新上市的币圈ETF），动态减小滚动窗口防止返回NaN
        total_rows = len(df)
        ma_window_short = min(50, max(5, int(total_rows / 4)))
        ma_window_long = min(250, max(10, int(total_rows / 2)))

        df["MA_50"] = df["Price"].rolling(window=ma_window_short).mean()
        df["MA_250"] = df["Price"].rolling(window=ma_window_long).mean()

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

        # 🚀 【自适应动态切片补丁】：完美兼容新老资产，绝不丢出空DataFrame
        # 只有在总行数远大于长期均线窗口时，才进行头部切片去噪；否则保留完整行数并前向填充
        df = df.ffill().bfill()
        if total_rows > ma_window_long:
            df = df.iloc[ma_window_long:].copy()
        
        print(f"⚙️ [DEBUG] {ticker} 价格对齐成功，当前最新真实价格为: {df['Price'].iloc[-1]:.2f}")
        print(f"✅ [DEBUG] {ticker} 全部特征矩阵计算大获成功！最终矩阵形状为: {df.shape}")
        return df

    except Exception as e:
        print(f"❌ [DEBUG 崩溃] 特征工程在计算 {ticker} 时遇到了逻辑错误: {e}")
        return None
