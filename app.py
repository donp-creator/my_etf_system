import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from features import get_stock_features
from rules import check_all_rules

# 1. 网页基础配置
st.set_page_config(page_title="智能ETF决策系统", layout="wide")

st.title("📱 智能交易决策系统 (MVP)")
st.caption("长线趋势追踪 + 短线横盘/回调择时逻辑")

# 2. 侧边栏：配置全局股票池
st.sidebar.header("⚙️ 全局系统配置")
ticker_input = st.sidebar.text_input(
    "输入你要监控的股票/ETF (英文逗号隔开):", 
    "SPY, VOO, QQQ, QQQE, RSP, SOXX, MOAT, SCHD, VYM, NOBL, VIG, SPLV, COW, IWY, FXI, MCHI, ASHR, IEFA, EWJ, EWW, DIA, IWM, VGT, XLK, XLY, XLP, XLV, XLF, XLI, XLB, XLRE, XLU, XLE, GDX, GLD, VNQ, BIL, SHY, IEF, TLT, LQD, HYG, BND, BITO, FBTC, IBIT, NVDA, MSTR, AAPL, MSFT, GOOG, AMZN, SMH, UNH, APP, BRK.B, GEV, SPCX, TSLA, VRT, BE, HOOD, URA, UMAC, BTCL, NVDA, CRCL, NBIS, MU, LITE, CIFR, UUUU, ALAB, RKLB"
)

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
                # 🚀【核心重构】：让策略层彻底收回对 Score 的控制权，直接接收第三个返回值！
                action, reason, priority_score = check_all_rules(df, ticker)
                detailed_dfs[ticker] = df
                
                price_val = float(df["Price"].iloc[-1])
                z_val = float(df["Z_Score"].iloc[-1])
                amp_val = float(df["Amplitude_10"].iloc[-1])
                
                summary_data.append({
                    "Ticker": str(ticker),
                    "当前价格": round(price_val, 2),
                    "长线Z-Score": round(z_val, 2),
                    "10日滚动振幅": f"{amp_val*100:.1f}%",
                    "建议行动": str(action),
                    "触发原因": str(reason),
                    "Priority_Score": priority_score
                })
        except Exception as e:
            st.sidebar.warning(f"解析 {ticker} 失败: {e}")
            
    if not summary_data:
        return pd.DataFrame(columns=["Ticker", "当前价格", "长线Z-Score", "10日滚动振幅", "建议行动", "触发原因", "Priority_Score"]), {}
        
    return pd.DataFrame(summary_data), detailed_dfs

# 触发扫描
if tickers:
    with st.spinner("正在通过 yfinance 抓取数据并进行时间序列分析..."):
        summary_df, all_dfs = run_system_scan(tickers)
        # 🚀 大表一律按字母 A-Z 排列，满足强迫症
        summary_df_table = summary_df.sort_values(by="Ticker", ascending=True).reset_index(drop=True)
else:
    st.warning("请在左侧侧边栏输入至少一个有效的股票/ETF代码。")
    st.stop()

# 4. 【首屏核心】：自适应响应式网格卡片瀑布流（纯盲盒解耦排序）
st.subheader("🚨 今日强力推荐")

if "建议行动" in summary_df.columns and not summary_df.empty:
    triggered_actions = summary_df[summary_df["建议行动"].isin(["买入", "卖出"])].copy()
    
    if not triggered_actions.empty:
        # 🚀 盲盒排序：不管后台分数怎么变，前端雷打不动直接降序排列
        triggered_actions = triggered_actions.sort_values(by="Priority_Score", ascending=False).reset_index(drop=True)
        
        st.info(f"🔔 提示：当前股票池共爆出 {len(triggered_actions)} 个信号。已按【策略层推荐优先级】从高到低为您置顶排序！")
        
        container_html = '<div style="display: flex; flex-wrap: wrap; gap: 15px; width: 100%;">'
        records = triggered_actions.to_dict(orient="records")
        
        for idx, row_dict in enumerate(records):
            bg_color = "rgba(255, 75, 75, 0.05)" if row_dict["建议行动"] == "买入" else "rgba(9, 171, 92, 0.05)"
            border_color = "#FF4B4B" if row_dict["建议行动"] == "买入" else "#09AB5C"
            text_color = "#FF4B4B" if row_dict["建议行动"] == "买入" else "#09AB5C"
            emoji = "🔴" if row_dict["建议行动"] == "买入" else "🟢"
            
            # 优雅打上高亮排名的勋章
            rank_badge = f'<span style="background-color:{border_color}; color:white; padding:2px 6px; border-radius:4px; font-size:11px; margin-right:5px;">🔥 TOP {idx+1}</span>' if idx < 3 else ''
            
            card_html = f"""
                <div style="flex: 1 1 280px; max-width: 350px; background-color: {bg_color}; border: 1px solid {border_color}; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); margin-bottom: 5px;">
                    <h4 style="margin: 0 0 8px 0; color: {text_color}; font-size: 16px;">
                        {rank_badge}{emoji} 【{row_dict['建议行动']}】{row_dict['Ticker']}
                    </h4>
                    <p style="margin: 0; font-size: 13px; color: #31333F; line-height: 1.45;">
                        <b>当前价:</b> ${row_dict['当前价格']} <br>
                        <b>长线 Z-Score:</b> {row_dict['长线Z-Score']} | <b>10日振幅:</b> {row_dict['10日滚动振幅']}<br>
                        <span style="color: #555555;"><b>决策情报:</b> {row_dict['触发原因']}</span>
                    </p>
                </div>
            """
            container_html += card_html
            
        container_html += '</div>'
        st.html(container_html)
    else:
        st.info("☀️ 今天所有监控的 ETF 都处于正常波动范围内，暂无强力买入/卖出推荐。")
else:
    st.info("☀️ 暂无监控数据。")

st.markdown("---")

# 5. 【中层】：全局股票池数据监控大表
st.subheader("📊 全局资产实时监控总览")
table_display_df = summary_df_table.drop(columns=["Priority_Score"], errors="ignore")
st.dataframe(table_display_df, use_container_width=True, hide_index=True)

# 6. 【底层】：穿透式详情图表
st.markdown("---")
st.subheader("📈 深度时间序列轨道探索")
selected_ticker = st.selectbox("选择一只 ETF 查看长期对数趋势轨道和近况:", tickers)

if selected_ticker in all_dfs:
    plot_df = all_dfs[selected_ticker].tail(500)
    fig = go.Figure()
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
