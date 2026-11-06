import os
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

def get_stock_features(ticker, period="10y"):
    """下载历史时序数据，采用三层降级防弹算法：优先计算分析师预期Double Beat，次优降级静态季报环比，末尾安全熔断"""
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

    # 3. 🚀【多因子三重保险注入层】
    valuation_metrics = {
        "市值": "-", "Trailing_PE": "-", "Forward_PE": "-", 
        "PEG_Ratio": "-", "市现率_PCF": "-", "自由现金流收益率": "-",
        "最新财报战绩": "-"
    }
    
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        f_info = ticker_obj.fast_info
        
        # 提取市值双重保险
        m_cap = 0
        if f_info and "market_cap" in f_info and f_info["market_cap"] > 0:
            m_cap = f_info["market_cap"]
        if m_cap == 0 and info and "marketCap" in info:
            m_cap = info["marketCap"]
            
        if m_cap and m_cap >= 1e12: valuation_metrics["市值"] = f"${m_cap / 1e12:.2f}T"
        elif m_cap and m_cap >= 1e9: valuation_metrics["市值"] = f"${m_cap / 1e9:.2f}B"
        elif m_cap and m_cap > 0: valuation_metrics["市值"] = f"${m_cap / 1e6:.1f}M"
                
        # 抓取基本面核心乘数
        if info:
            if info.get("trailingPE"): valuation_metrics["Trailing_PE"] = round(info["trailingPE"], 1)
            if info.get("forwardPE"): valuation_metrics["Forward_PE"] = round(info["forwardPE"], 1)
            if info.get("pegRatio"): valuation_metrics["PEG_Ratio"] = round(info["pegRatio"], 2)
            if info.get("priceToCashFlow"): valuation_metrics["市现率_PCF"] = round(info["priceToCashFlow"], 1)
            
            fcf = info.get("freeCashflow") or info.get("operatingCashflow", 0)
            if fcf and m_cap and m_cap > 0:
                fcf_yield = (fcf / m_cap) * 100
                if fcf_yield > 0: valuation_metrics["自由现金流收益率"] = f"{fcf_yield:.1f}%"

        # 🚀🚀🚀【三层降级防弹算法核心】🚀🚀🚀
        report_success = False
        
        # 🥇 【第一层保险】：优先穿透分析师预期矩阵计算 Double Beat
        try:
            h_earnings = ticker_obj.get_earnings_history(limit=4)
            if h_earnings is not None and len(h_earnings) > 0:
                latest_report = None
                for report in h_earnings:
                    if report.get("epsActual") is not None and report.get("epsEstimate") is not None:
                        latest_report = report
                        break
                
                if latest_report:
                    eps_act, eps_est = latest_report.get("epsActual"), latest_report.get("epsEstimate")
                    rev_act, rev_est = latest_report.get("revenueActual"), latest_report.get("revenueEstimate")
                    
                    eps_beat = eps_act > eps_est if (eps_act is not None and eps_est is not None) else False
                    rev_beat = rev_act > rev_est if (rev_act is not None and rev_est is not None) else False
                    
                    if eps_beat and rev_beat:
                        eps_surprise = ((eps_act - eps_est) / abs(eps_est)) * 100 if eps_est != 0 else 0
                        valuation_metrics["最新财报战绩"] = f"🔥 Double Beat (+{eps_surprise:.1f}%)"
                    elif (not eps_beat) and (not rev_beat):
                        valuation_metrics["最新财报战绩"] = "🚨 Double Miss"
                    elif eps_beat and (not rev_beat):
                        valuation_metrics["最新财报战绩"] = "✨ EPS Beat / 营收 Miss"
                    else:
                        valuation_metrics["最新财报战绩"] = "✨ 营收 Beat / EPS Miss"
                    report_success = True
        except Exception:
            pass

        # 🥈 【第二层保险】：如果第一层掉链子返回了空，自动无缝降级读取底层的季度利润大表算“环比”
        if not report_success:
            try:
                q_financials = ticker_obj.quarterly_financials
                if q_financials is not None and not q_financials.empty:
                    income_labels = ["Net Income", "Operating Income", "Gross Profit"]
                    target_row = None
                    for label in income_labels:
                        if label in q_financials.index:
                            target_row = q_financials.loc[label]
                            break
                    
                    if target_row is not None and len(target_row) >= 2:
                        latest_quarter_val = float(target_row.iloc[0])
                        prev_quarter_val = float(target_row.iloc[1])
                        
                        if latest_quarter_val > prev_quarter_val and latest_quarter_val > 0:
                            growth_pct = ((latest_quarter_val - prev_quarter_val) / abs(prev_quarter_val)) * 100
                            valuation_metrics["最新财报战绩"] = f"📈 利润环比大超 (+{growth_pct:.1f}%)"
                        elif latest_quarter_val < prev_quarter_val:
                            growth_pct = ((latest_quarter_val - prev_quarter_val) / abs(prev_quarter_val)) * 100
                            valuation_metrics["最新财报战绩"] = f"📉 利润环比下滑 ({growth_pct:.1f}%)"
                        else:
                            valuation_metrics["最新财报战绩"] = "⚖️ 环比基本持平"
                        report_success = True
            except Exception:
                pass

        # 🥉 【第三层】：如果前两层都拿不到数据（如 GLD 黄金信托等衍生品），最后安全兜底为 "-"
        if not report_success:
            valuation_metrics["最新财报战绩"] = "-"

    except Exception as e:
        print(f"💡 提示：{ticker} 属于衍生资产，自动跳过基本面因子注入。")

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
        return df

    except Exception as e:
        print(f"❌ 特征计算失败 {ticker}: {e}")
        return None
