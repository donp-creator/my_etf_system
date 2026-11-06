import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from features import get_stock_features
from rules import check_all_rules

# 1. 网页基础配置（自适应手机）
st.set_page_config(page_title="智能ETF决策系统", layout="wide")

st.title("📱 智能交易决策系统 (MVP)")
st.caption("长线趋势追踪 + 短线横盘/回调择时逻辑")

# 2. 侧边栏：手动配置全局股票池
st.sidebar.header("⚙️ 全局系统配置")
ticker_input = st.sidebar.text_input(
    "输入你要监控的股票/ETF (英文逗号隔开):", 
    "VOO, XLK, MOAT, GDX"
)

# 清洗输入的 Tickers
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

# 3. 核心数据调度中心
@st.cache_data(ttl=3600)
def run_system_scan(ticker_list):
    summary_data = []
    detailed_dfs = {}
    
    for ticker in ticker_list:
        try:
            df = get_stock_features(ticker, period="10y")
            if df is not None and not df.empty and len(df) > 0:
                action, reason = check_all_rules(df, ticker)
                detailed_dfs[ticker] = df
                
                # 安全地提取标量值，确保不受多层索引干扰
                price_val = float(df["Price"].iloc[-1])
                z_val = float(df["Z_Score"].iloc[-1])
                amp_val = float(df["Amplitude_10"].iloc[-1])
                
                summary_data.append({
                    "Ticker": str(ticker),
                    "当前价格": round(price_val, 2),
                    "长线Z-Score": round(z_val, 2),
                    "10日滚动振幅": f"{amp_val*100:.1f}%",
                    "建议行动": str(action),
                    "触发原因": str(reason)
                })
        except Exception as e:
            st.sidebar.warning(f"解析 {ticker} 失败: {e}")
            
    # 如果完全没有成功的数据，返回一个带正确列名的空DataFrame
    if not summary_data:
        return pd.DataFrame(columns=["Ticker", "当前价格", "长线Z-Score", "10日滚动振幅", "建议行动", "触发原因"]), {}
        
    return pd.DataFrame(summary_data), detailed_dfs

# 触发扫描
if tickers:
    with st.spinner("正在通过 yfinance 抓取数据并进行时间序列分析..."):
        summary_df, all_dfs = run_system_scan(tickers)
else:
    st.warning("请在左侧侧边栏输入至少一个有效的股票/ETF代码。")
    st.stop()

# 4. 【首屏核心】：高亮推荐卡片
st.subheader("🚨 今日强力推荐")

# 安全检查“建议行动”列是否存在
if "建议行动" in summary_df.columns and not summary_df.empty:
    triggered_actions = summary_df[summary_df["建议行动"].isin(["买入", "卖出"])]
    
    if not triggered_actions.empty:
        cols = st.columns(len(triggered_actions))
        for idx, row in enumerate(triggered_actions.itertuples()):
            with cols[idx]:
                color = "rgba(255, 75, 75, 0.1)" if row.建议行动 == "买入" else "rgba(9, 171, 92, 0.1)"
                border_color = "#FF4B4B" if row.建议行动 == "买入" else "#09AB5C"
                
                st.markdown(f"""
                    <div style="background-color: {color}; border: 2px solid {border_color}; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                        <h3 style="margin: 0; color: {border_color};">【{row.建议行动}】{row.Ticker}</h3>
                        <p style="margin: 5px 0 0 0; font-size: 14px; color: #31333F;"><b>核心原因:</b> {row.触发原因}</p>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("☀️ 今天所有监控的 ETF 都处于正常波动范围内，暂无强力买入/卖出推荐。")
else:
    st.info("☀️ 暂无监控数据。")

st.markdown("---")

# 5. 【中层】：全局股票池监控表
st.subheader("📊 全局资产实时监控总览")
st.dataframe(summary_df, use_container_width=True, hide_index=True)

# 6. 【底层】：穿透式详情图表
st.markdown("---")
st.subheader("📈 深度时间序列轨道探索")
selected_ticker = st.selectbox("选择一只 ETF 查看长期对数趋势轨道和近况:", tickers)

if selected_ticker in all_dfs:
    plot_df = all_dfs[selected_ticker].tail(500)
    
    fig = go.Figure()
    # 实际价格 - 安全转换为1维数组
    y_price = plot_df['Price'].values.flatten()
    y_trend = plot_df['Log_Trend_Price'].values.flatten()
    
    fig.add_trace(go.Scatter(x=plot_df.index, y=y_price, name='实际价格', line=dict(color='#1f77b4', width=2)))
    fig.add_trace(go.Scatter(x=plot_df.index, y=y_trend, name='长期对数回归中心线', line=dict(color='#ff7f0e', dash='dash')))
    
    fig.update_layout(
        title=f"{selected_ticker} 过去500天长线回归轨道",
        yaxis_type="log",
        xaxis_title="日期",
        yaxis_title="价格 (对数轴)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)
