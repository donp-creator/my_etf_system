import logging
import re
import streamlit as st
import pandas as pd
import yfinance as yf
from html import escape
import plotly.graph_objects as go
from datetime import datetime, timedelta
from features import get_stock_features
from rules import check_all_rules
from research_profiles import generate_profile

# 1. 网页基础配置
st.set_page_config(page_title="智能ETF决策系统", layout="wide")

st.title("📱 智能交易决策系统 (MVP)")
st.caption("长线趋势追踪 + 基本面多因子 + 宏观事件驱动决策日历看板")

# 2. 侧边栏配置
st.sidebar.header("⚙️ 全局系统配置")
ticker_input = st.sidebar.text_input(
    "输入你要监控的股票/ETF (英文逗号隔开):", 
    "SPY, VOO, QQQ, QQQE, RSP, SOXX, MOAT, SCHD, VYM, NOBL, VIG, SPLV, COW, IWY, FXI, MCHI, ASHR, IEFA, EWJ, EWW, DIA, IWM, VGT, XLK, XLY, XLP, XLV, XLF, XLI, XLB, XLRE, XLU, XLE, GDX, GLD, VNQ, SHY, IEF, TLT, LQD, HYG, BND, BITO, FBTC, IBIT, NVDA, MSTR, AAPL, MSFT, GOOG, AMZN, SMH, UNH, APP, BRK.B, GEV, SPCX, TSLA, VRT, BE, HOOD, URA, UMAC, BTCL, CRCL, NBIS, MU, LITE, CIFR, UUUU, ALAB, RKLB,  META, INTC"
)
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

# 3. 数据调度中心
@st.cache_data(ttl=3600)
def run_system_scan(ticker_list):
    summary_data = []
    detailed_dfs = {}
    
    for ticker in ticker_list:
        try:
            df = get_stock_features(ticker, period="10y")
            if df is not None and not df.empty and len(df) > 0:
                action, reason, priority_score = check_all_rules(df, ticker)
                detailed_dfs[ticker] = df
                
                v_m = df.attrs.get("valuation", {
                    "市值": "-", "Trailing_PE": "-", "Forward_PE": "-", 
                    "PEG_Ratio": "-", "市现率_PCF": "-", "自由现金流收益率": "-",
                    "最新财报战绩": "-"
                })
                
                summary_data.append({
                    "Ticker": str(ticker),
                    "当前价格": round(float(df["Price"].iloc[-1]), 2),
                    "市值": v_m["市值"],
                    "PE (滚动/前瞻)": f"{v_m['Trailing_PE']} / {v_m['Forward_PE']}",
                    "PEG比率": v_m["PEG_Ratio"],
                    "自由现金收益率": v_m["自由现金流收益率"],
                    "最新财报战绩": v_m["最新财报战绩"],
                    "长线Z-Score": round(float(df["Z_Score"].iloc[-1]), 2),
                    "10日滚动振幅": f"{float(df['Amplitude_10'].iloc[-1])*100:.1f}%",
                    "建议行动": str(action),
                    "触发原因": str(reason),
                    "Priority_Score": priority_score
                })
        except Exception as e:
            logging.getLogger(__name__).exception("扫描 %s 失败", ticker)
            st.sidebar.warning(f"解析 {ticker} 失败: {e}")
            
    if not summary_data:
        return pd.DataFrame(), {}
        
    return pd.DataFrame(summary_data), detailed_dfs

# 触发主扫描
if tickers:
    with st.spinner("正在抓取行情与企业财务数据..."):
        summary_df, all_dfs = run_system_scan(tickers)
        if not summary_df.empty:
            summary_df_table = summary_df.sort_values(by="Ticker", ascending=True).reset_index(drop=True)
        else:
            summary_df_table = summary_df
else:
    st.warning("请在左侧侧边栏输入至少一个有效的股票/ETF代码。")
    st.stop()

# 🚀🚀🚀 【全新重磅板块】：接下来30天大事件与重要日期监控日历 🚀🚀🚀
st.markdown("---")
st.subheader("📅 接下来30天：大事件与重要日期决策日历 (驱动追踪)")

# 构建固定宏观重磅经济事件矩阵 (锁定2026年11月5日至12月5日全美股硬核真实大事件)
macro_events = [
    {"date": "2026-11-10", "type": "🔴 宏观核心", "event": "美国 10 月 CPI 通胀数据公布日 (宏观波动引擎)"},
    {"date": "2026-11-20", "type": "⚡ 衍生结算", "event": "美股 11 月月度期权结算日 (流动性踩踏窗口)"},
    {"date": "2026-11-25", "type": "🏛️ 美联储", "event": "FOMC 11月会议纪要公布 (释放下一步加息/降息关键措辞)"},
    {"date": "2026-11-30", "type": "🐳 机构调仓", "event": "全球华尔街养老金与主权基金月底换仓窗口期 (尾盘异常波动)"},
    {"date": "2026-12-04", "type": "🔴 宏观核心", "event": "美国 11 月非农就业数据公布 (决议12月降息幅度的核心底座)"},
    {"date": "2026-12-08", "type": "🏛️ 美联储", "event": "美联储 FOMC 今年最后一轮议息会议开幕日 (12/8-12/9)"}
]

@st.cache_data(ttl=3600)
def fetch_earnings_dates(symbol):
    calendar = yf.Ticker(symbol).calendar
    if isinstance(calendar, pd.DataFrame):
        raw_dates = (
            calendar.loc["Earnings Date"].to_numpy().ravel()
            if "Earnings Date" in calendar.index else []
        )
    else:
        raw_dates = (calendar or {}).get("Earnings Date", [])

    dates = set()
    for value in raw_dates:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.notna(parsed):
            dates.add(parsed.date())
    return sorted(dates)


# 以纽约市场日期为准，今天算第 1 天。
today = pd.Timestamp.now(tz="America/New_York").date()
last_day = today + timedelta(days=29)
st.caption(
    f"{today} 至 {last_day}（纽约时间）。宏观日期沿用手工清单，"
    "财报日期来自 Yahoo Finance，可能是预计日期。"
)

calendar_list = []
for event in macro_events:
    event_date = datetime.strptime(event["date"], "%Y-%m-%d").date()
    if today <= event_date <= last_day:
        calendar_list.append({
            "date": event_date,
            "days_left": (event_date - today).days,
            "type": event["type"],
            "name": event["event"],
        })

failed_tickers = []
for ticker in dict.fromkeys(tickers):
    yahoo_symbol = "BRK-B" if ticker == "BRK.B" else ticker
    try:
        dates = fetch_earnings_dates(yahoo_symbol)
    except Exception:
        failed_tickers.append(ticker)
        continue

    if dates and dates[0] <= last_day and dates[-1] >= today:
        date_label = str(dates[0]) if len(dates) == 1 else f"{dates[0]} 至 {dates[-1]}"
        display_date = max(today, dates[0])
        calendar_list.append({
            "date": display_date,
            "days_left": (display_date - today).days,
            "type": "🔥 个股财报",
            "name": f"【{ticker}】财报公布日期（Yahoo Finance：{date_label}）",
        })

if failed_tickers:
    st.warning("部分代码的财报日历查询失败：" + "、".join(failed_tickers))

# 数据科学排序：严格按照日期从近到远升序排列
calendar_df = pd.DataFrame(calendar_list)
if not calendar_df.empty:
    calendar_df = calendar_df.sort_values(by="date", ascending=True).reset_index(drop=True)
    
    # 精美前端时间轴渲染
    timeline_html = '<div style="background-color: #f8f9fa; border-radius: 8px; padding: 20px; border: 1px solid #e9ecef;">'
    for idx, row in calendar_df.iterrows():
        # 根据事件紧迫度计算高亮标签颜色
        badge_color = "#FF4B4B" if row["days_left"] <= 5 else "#1f77b4"
        if "个股财报" in row["type"]: badge_color = "#ff7f0e"
        
        countdown = "今天" if row["days_left"] == 0 else f"倒计时 {row['days_left']} 天"
        timeline_html += f"""
            <div style="display: flex; align-items: flex-start; margin-bottom: 15px; border-bottom: 1px dashed #dee2e6; padding-bottom: 10px;">
                <div style="min-width: 110px; font-weight: bold; color: #495057; font-size: 14px;">📅 {row['date']}</div>
                <div style="background-color: {badge_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-right: 15px; min-width: 75px; text-align: center; font-weight: bold;">
                    {countdown}
                </div>
                <div style="font-size: 13px; color: #212529;">
                    <span style="font-weight: bold; color: {badge_color};">[{row['type']}]</span> {escape(row['name'])}
                </div>
            </div>
        """
    timeline_html += "</div>"
    st.html(timeline_html)
else:
    st.info("☀️ 当前数据源未返回这 30 天内的事件。")

st.markdown("---")

@st.cache_data(ttl=3600, max_entries=200)
def load_research(ticker, peers):
    return generate_profile(ticker, peers=peers)


@st.cache_data(ttl=86400)
def get_ticker_profile(ticker):
    yahoo_symbol = "BRK-B" if ticker == "BRK.B" else ticker
    info = yf.Ticker(yahoo_symbol).get_info()
    quote_type = info.get("quoteType")
    ticker_type = {"EQUITY": "个股", "ETF": "ETF"}.get(quote_type, "")
    description = " ".join((info.get("longBusinessSummary") or "").split())

    if ticker_type == "ETF":
        fund_name = info.get("longName") or info.get("shortName") or ""
        weighting = {
            "RSP": "等权重",
            "QQQE": "等权重",
            "SPY": "非等权重 · 市值加权",
            "VOO": "非等权重 · 市值加权",
            "QQQ": "非等权重 · 修正市值加权",
        }.get(ticker)
        if weighting is None:
            explicitly_equal = re.search(r"\bequal(?:\s+|-)weight(?:ed)?\b", fund_name, re.I)
            weighting = "等权重" if explicitly_equal else "权重待确认"
        ticker_type = f"ETF · {weighting}"
        description = description or fund_name

    name = info.get("longName") or info.get("shortName") or ticker
    if quote_type == "ETF":
        issuer = info.get("fundFamily")
        intro = f"{name}，由 {issuer} 发行。" if issuer else name
    else:
        sector = info.get("industry")
        intro = f"{name} · {sector}" if sector else name
        first_sentence = re.split(r"(?<=[.!?。])\s+", description, maxsplit=1)[0]
        if first_sentence and len(first_sentence) <= 115:
            intro = first_sentence

    if len(intro) > 115:
        intro = intro[:112].rstrip() + "…"
    return ticker_type, intro


# 5. 【首屏】：自适应响应式网格推荐卡片
st.subheader("🚨 今日强力推荐")
if "建议行动" in summary_df.columns and not summary_df.empty:
    triggered_actions = summary_df[summary_df["建议行动"].isin(["买入", "卖出"])].copy()
    if not triggered_actions.empty:
        triggered_actions = triggered_actions.sort_values(by="Priority_Score", ascending=False).reset_index(drop=True)
        st.info(f"🔔 提示：当前股票池共爆出 {len(triggered_actions)} 个交易信号。已按优先级评分排序。")
        
        records = triggered_actions.to_dict(orient="records")
        for idx, row_dict in enumerate(records):
            if idx % 3 == 0:
                card_columns = st.columns(3)
            card_column = card_columns[idx % 3]
            bg_color = "rgba(255, 75, 75, 0.05)" if row_dict["建议行动"] == "买入" else "rgba(9, 171, 92, 0.05)"
            border_color = "#FF4B4B" if row_dict["建议行动"] == "买入" else "#09AB5C"
            text_color = "#FF4B4B" if row_dict["建议行动"] == "买入" else "#09AB5C"
            emoji = "🔴" if row_dict["建议行动"] == "买入" else "🟢"
            rank_badge = f'<span style="background-color:{border_color}; color:white; padding:2px 6px; border-radius:4px; font-size:11px; margin-right:5px;">🔥 TOP {idx+1}</span>' if idx < 3 else ''
            
            ticker = row_dict["Ticker"]
            try:
                ticker_type, description = get_ticker_profile(ticker)
            except Exception:
                ticker_type, description = "", ""
            ticker_label = f"{escape(ticker)}（{ticker_type}）" if ticker_type else escape(ticker)

            profile_line = (
                f'<p style="margin: 0 0 8px; color: #475569; font-size: 13px;">'
                f'{escape(description or "暂无可用简介")}</p>'
            )

            card_html = f"""
                <div style="min-width: 0; overflow-wrap: anywhere; background-color: {bg_color}; border: 1px solid {border_color}; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); margin-bottom: 5px;">
                    <h4 style="margin: 0 0 8px; color: {text_color}; font-size: 16px;">
                        {rank_badge}{emoji} 【{row_dict['建议行动']}】{ticker_label}
                    </h4>
                    {profile_line}
                    <p style="margin: 0; font-size: 13px; color: #31333F; line-height: 1.45;">
                        <b>当前价:</b> ${row_dict['当前价格']} | <b>市值:</b> {row_dict['市值']}<br>
                        <b>PE (滚/前):</b> {row_dict['PE (滚动/前瞻)']} | <b>PEG:</b> {row_dict['PEG比率']}<br>
                        <b>最新财报战绩:</b> <span style="color:#1f77b4; font-weight:bold;">{row_dict['最新财报战绩']}</span><br>
                        <b>长线 Z-Score:</b> {row_dict['长线Z-Score']} | <b>现金收益率:</b> {row_dict['自由现金收益率']}<br>
                        <span style="color: #555555;"><b>决策情报:</b> {row_dict['触发原因']}</span>
                    </p>
                </div>
            """
            with card_column:
                st.html(card_html)
                if st.button(f"查看 {ticker} 研究资料", key=f"research_card_{ticker}_{idx}"):
                    st.session_state["research_ticker"] = ticker
                    st.session_state["research_peers"] = ()

        selected_research = st.session_state.get("research_ticker")
        if selected_research in set(triggered_actions["Ticker"]):
            st.markdown("---")
            st.subheader(f"🔎 {selected_research} 研究资料")
            if st.button("收起研究资料", key="close_research"):
                st.session_state.pop("research_ticker", None)
            else:
                with st.form(f"research_peers_{selected_research}"):
                    peer_input = st.text_input("同类对比代码（可选，用逗号分隔）")
                    if st.form_submit_button("更新对比"):
                        st.session_state["research_peers"] = tuple(dict.fromkeys(
                            peer.strip().upper()
                            for peer in peer_input.replace("，", ",").split(",")
                            if peer.strip() and peer.strip().upper() != selected_research
                        ))

                try:
                    with st.spinner(f"正在查询 {selected_research} 的研究资料…"):
                        profile = load_research(
                            selected_research, st.session_state["research_peers"]
                        )
                except ValueError as exc:
                    st.warning(str(exc))
                except Exception:
                    st.warning("暂时无法加载研究资料，请稍后重试。")
                else:
                    st.write(profile.intro)
                    st.markdown(profile.details)
                    st.markdown("**数据来源**")
                    for label, url in profile.sources:
                        st.markdown(f"- [{label}]({url})")
    else:
        st.info("☀️ 今天所有监控的 ETF 都处于正常波动范围内，暂无强力买入/卖出推荐。")

st.markdown("---")

# 6. 【中层】：全局多因子资产监控大表
st.subheader("📊 全局多因子资产监控总览")
if not summary_df_table.empty:
    table_display_df = summary_df_table.drop(columns=["Priority_Score"], errors="ignore")
    st.dataframe(table_display_df, use_container_width=True, hide_index=True)

# 7. 【底层】：穿透式详情图表
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
        yaxis_type="log", xaxis_title="日期", yaxis_title="价格 (对数轴)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)
