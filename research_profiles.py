"""Generate research cards from public Yahoo Finance data.

Usage: python research_profiles.py SOXX CIFR
       python research_profiles.py SOXX --peer SMH --peer SOXL
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import math
import re
import sys
from urllib.parse import quote

from pandas import DateOffset
import yfinance as yf


@dataclass(frozen=True)
class ResearchProfile:
    ticker: str
    intro: str
    details: str
    sources: tuple[tuple[str, str], ...]


def _symbol(value):
    value = value.strip().upper()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{0,14}", value):
        raise ValueError("请输入有效的证券代码。")
    return "BRK-B" if value == "BRK.B" else value


def _fetch(load, fallback):
    try:
        return load()
    except Exception:
        return fallback


def _number(value, spec, scale=1):
    try:
        value = float(value) * scale
        if math.isfinite(value):
            return format(value, spec)
    except (TypeError, ValueError, OverflowError):
        pass
    return "未核实"


def _returns(asset):
    history = _fetch(lambda: asset.history(period="5y", auto_adjust=True), None)
    if history is None or history.empty:
        return {}
    close = history["Close"].dropna()
    close = close[close > 0].sort_index()
    if close.empty:
        return {}
    result = {}
    for years in (1, 3):
        target = close.index[-1] - DateOffset(years=years)
        earlier = close.loc[close.index <= target]
        if earlier.empty or (target.date() - earlier.index[-1].date()).days > 7:
            continue
        window = close.loc[earlier.index[-1]:]
        result[years] = (
            str(window.index[0].date()), str(window.index[-1].date()),
            window.iloc[-1] / window.iloc[0] - 1,
            (window / window.cummax() - 1).min(),
        )
    return result


def generate_profile(ticker: str, *, peers=()) -> ResearchProfile:
    symbol = _symbol(ticker)
    asset = yf.Ticker(symbol)
    info = _fetch(asset.get_info, {}) or {}
    asset_type = info.get("quoteType")
    name = info.get("longName") or info.get("shortName") or symbol
    industry = info.get("industry") or info.get("category")
    website = info.get("website") or ""
    sources = {symbol: f"https://finance.yahoo.com/quote/{quote(symbol, safe='')}/"}
    lines = ["### 基本资料"]
    risk = []
    if asset_type == "ETF":
        fund = _fetch(asset.get_funds_data, None)
        overview = _fetch(lambda: fund.fund_overview, {}) or {}
        issuer = overview.get("family") or info.get("fundFamily")
        category = overview.get("categoryName") or industry
        intro = " · ".join(filter(None, [name, issuer, category]))
        operations = _fetch(lambda: fund.fund_operations, None)
        expense = _fetch(
            lambda: operations.loc["Annual Report Expense Ratio", symbol], None
        )
        holdings = _fetch(lambda: fund.top_holdings, None)
        weights = _fetch(lambda: holdings["Holding Percent"].dropna(), None)
        concentration = weights.sum() if weights is not None and len(weights) == 10 else None
        inception = info.get("fundInceptionDate")
        inception_date = _fetch(lambda: datetime.utcfromtimestamp(inception).date(), "未核实")
        lines.extend([
            f"- 发行方：{issuer or '未核实'}｜类别：{category or '未核实'}",
            f"- 成立日期：{inception_date}｜年报费用率：{_number(expense, '.2%')}",
            "- 追踪指数及权重规则：以发行方基金说明书为准，当前未核实。",
            f"- 前十大持仓占比：{_number(concentration, '.1%')}（来源未提供可确认的持仓日期）",
        ])
        if holdings is not None and not holdings.empty:
            lines.append("- 主要持仓：" + "、".join(map(str, holdings.index[:5])))
        risk.append("- 比较同类基金时，核查跟踪指数、持仓集中度和杠杆目标。")
    elif asset_type == "EQUITY":
        description = " ".join((info.get("longBusinessSummary") or "").split())
        first_sentence = re.split(r"(?<=[.!?。])\s+", description, maxsplit=1)[0]
        intro = first_sentence if 0 < len(first_sentence) <= 115 else f"{name} · {industry or '个股'}"
        lines.extend([
            f"- 公司：{name}｜行业：{industry or '未核实'}",
            f"- 主营业务：{first_sentence or '未核实'}",
            f"- 市值（美元）：{_number(info.get('marketCap'), ',.0f')}｜滚动市盈率：{_number(info.get('trailingPE'), '.1f')}",
        ])
        if isinstance(website, str) and website.startswith("https://"):
            sources["公司网站"] = website
        risk.append("- 研究业绩时，核查营收增速、盈利质量和经营现金流。")
    else:
        intro = name
        lines.append("- 资产类型及基本资料暂未核实。")

    results = _returns(asset)
    lines.append("\n### 横向对比")
    comparison = list(dict.fromkeys(_symbol(peer) for peer in peers))
    if not comparison:
        comparison = ["SPY"] if symbol != "SPY" else ["QQQ"]
        lines.append("- 以下为市场基准参照。同类对比可通过 --peer 指定。")
    for peer in comparison:
        if peer == symbol:
            continue
        sources[peer] = f"https://finance.yahoo.com/quote/{quote(peer, safe='')}/"
        own = results.get(1)
        other = _returns(yf.Ticker(peer)).get(1)
        if own and other and own[:2] == other[:2]:
            lines.append(f"- {symbol} vs {peer}（{own[0]} 至 {own[1]}）："
                         f"调整后回报 {own[2]:.1%} vs {other[2]:.1%}。")
        else:
            lines.append(f"- {peer}：暂无起止交易日一致的 1 年数据。")

    lines.append("\n### 纵向对比")
    for years in (1, 3):
        if years in results:
            start, end, total_return, drawdown = results[years]
            lines.append(f"- 近 {years} 年（{start} 至 {end}）："
                         f"调整后累计回报 {total_return:.1%}，最大回撤 {drawdown:.1%}。")
        else:
            lines.append(f"- 近 {years} 年：行情不足。")

    lines.append("\n### 需要知道")
    lines.extend(risk)
    lines.append("- 回报根据 Yahoo 调整后收盘价估算。免费数据可能暂时无法取得。")
    if len(intro) > 115:
        intro = intro[:112].rstrip() + "…"
    return ResearchProfile(symbol, intro, "\n".join(lines), tuple(sources.items()))


def main():
    parser = argparse.ArgumentParser(description="生成无需 API Key 的证券研究卡片")
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--peer", action="append", default=[])
    args = parser.parse_args()
    status = 0
    for ticker in args.tickers:
        try:
            profile = generate_profile(ticker, peers=args.peer)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            status = 1
            continue
        print(f"# {profile.ticker}\n\n{profile.intro}\n\n{profile.details}\n\n### 来源")
        for label, url in profile.sources:
            print(f"- {label} — {url}")
        print()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
