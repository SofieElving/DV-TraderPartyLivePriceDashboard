import os
import time
import math
import random
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo  # <-- added

import pandas as pd
import streamlit as st
import altair as alt
import base64
from pathlib import Path

st.set_page_config(page_title="Trader Bar Display", layout="wide")

# Add a floating logo via HTML + CSS
def top_right_logo_markdown(img_path: str, width_px: int = 250, top_px: int = 80, right_px: int = 40):
    p = Path(img_path)
    if not p.exists():
        st.warning(f"Logo not found at {img_path}")
        return
    mime = "image/png"
    ext = p.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif ext == ".svg":
        mime = "image/svg+xml"

    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .top-right-logo {{
            position: fixed;
            top: {top_px}px;
            right: {right_px}px;
            width: {width_px}px;
            z-index: 999999;
            pointer-events: none;
            display: block;
        }}
        </style>
        <img src="data:{mime};base64,{b64}" class="top-right-logo" />
        """,
        unsafe_allow_html=True,
    )

top_right_logo_markdown("images/bar_logo.png")

# Display timezone (CET/CEST for Copenhagen)
CET_TZ = ZoneInfo("Europe/Copenhagen")

# ---------- Helpers ----------
def parse_int_list(s, fallback):
    try:
        vals = [int(x.strip()) for x in s.split(",") if x.strip() != ""]
        return vals or fallback
    except Exception:
        return fallback

def build_schedule(start_time: datetime, interval_min: int, n_points: int):
    return [start_time + timedelta(minutes=interval_min * i) for i in range(n_points)]

def clamp(n, lo, hi):
    return max(lo, min(hi, n))

def within_notice_window(now: datetime, next_tick: datetime, min_s=60, max_s=120):
    secs = (next_tick - now).total_seconds()
    # strict upper bound to avoid edge sticking
    return (min_s <= secs < max_s), max(0, int(secs))

def seeded_choice(seed, options):
    rng = random.Random(seed)
    return rng.choice(options) if options else ""

# ---------- Defaults ----------
DEFAULT_BEER = [20, 20, 19, 21, 22, 17, 14, 25, 20, 17, 18, 22, 15, 12, 21, 20, 18, 17, 33, 30, 16, 10, 18, 23, 20, 17, 21, 17, 19, 10, 100]
DEFAULT_SHOTS = [7, 8, 7, 7, 6, 7, 9, 5, 10, 8, 7, 8, 4, 7, 6, 12, 10, 6, 5, 8, 7, 7, 4, 3, 6, 8, 7, 9, 8, 7, 20]

DEFAULT_NEWS = {
    "Beer": {
        "positive": [
            "Nisserne er på vej med Julebryg 🎅",
            "The Bartenders are Getting Drunk 🥂",
            "Did Anyone Say Beer Pong? 🏓",
            "Mind Energy Saves Students - Subsidices Beer for Everyone 🍻",
        ],
        "neutral": [
            "MatKant has Run Out of Cake 🍰",
            "Jens Ledet Runs for Mayor 👴🏼",
            "New Shrek Movie??? 🐸",
            "Mind Energy Wins Fist Fight Against InCommodities and BD Energy 💪🏼",
        ],
        "negative": [
            "Drones Spottet Above Tuborg Factory 🚁",
            "Julebryg er sneet inde ❄️",
            "New Tesla Runs on Beers - Demand Increases 🍺",
            "Datavenskab has Monopoly on Beer Prices - Students Outraged 😡",
        ],
    },
    "Shots": {
        "positive": [
            "The Bartenders are Getting Drunk 🥂",
            "Local Man Claims Sambuca Heals His Wi-Fi Signal 📶",
            "Mind Energy Shorts Sambuca - Increases Student Happiness 😃",
            "Mind Energy in Search for Student Workers - Invests in Sambuca 🍹",
        ],
        "neutral": [
            "The Pandas in Aalborg Zoo are on the Loose 🐼",
            "Your Mom has Hit the Dance Floor 💃",
            "World Health Organization Urges 'Sambuca in Moderation,' Promptly Ignored by Everyone 🍹",
            "MatKant Replaces Coffee with Sambuca for Finals Week — Grades Drop, Morale Soars 📚",
        ],
        "negative": [
            "Sambuca Shortage Spurs Home Distilling Boom — Authorities Consider Crackdown 🍸",
            "Italy Accidentally Declares Sambuca a Currency — Markets in Chaos, Bartenders Elated 💶",
            "Water round spotted — risk-off vibes 💧",
            "Worldwide Anise Shortage 🌍",
        ],
    },
}

# ---------- Sidebar Controls ----------
with st.sidebar:
    st.header("Setup")

    interval_min = st.number_input(
        "Update interval (minutes)", min_value=1, max_value=30,
        value=int(os.getenv("INTERVAL_MIN", "10"))
    )
    n_points = st.number_input(
        "Number of updates (points)", min_value=1, max_value=300,
        value=int(os.getenv("N_POINTS", "31"))
    )

    st.markdown("**Price lists (comma-separated integers):**")
    beer_str = st.text_area("Beer prices", value=",".join(map(str, DEFAULT_BEER)), height=100)
    shots_str = st.text_area("Shot prices", value=",".join(map(str, DEFAULT_SHOTS)), height=100)

    with st.expander("News pools — Beer"):
        beer_pos = st.text_area("Beer: Positive (one per line)", value="\n".join(DEFAULT_NEWS["Beer"]["positive"]), height=100)
        beer_neu = st.text_area("Beer: Neutral (one per line)",  value="\n".join(DEFAULT_NEWS["Beer"]["neutral"]),  height=100)
        beer_neg = st.text_area("Beer: Negative (one per line)", value="\n".join(DEFAULT_NEWS["Beer"]["negative"]), height=100)

    with st.expander("News pools — Shots"):
        shots_pos = st.text_area("Shots: Positive (one per line)", value="\n".join(DEFAULT_NEWS["Shots"]["positive"]), height=100)
        shots_neu = st.text_area("Shots: Neutral (one per line)",  value="\n".join(DEFAULT_NEWS["Shots"]["neutral"]),  height=100)
        shots_neg = st.text_area("Shots: Negative (one per line)", value="\n".join(DEFAULT_NEWS["Shots"]["negative"]), height=100)

    st.markdown("---")
    st.subheader("Run Control")

    invert_beer = st.checkbox("Invert green/red for Beer delta", value=True)
    invert_shots = st.checkbox("Invert green/red for Shots delta", value=True)

    now_utc = datetime.now(timezone.utc)
    if "start_time" not in st.session_state:
        st.session_state.start_time = now_utc

    if st.button("Restart run from now"):
        st.session_state.start_time = datetime.now(timezone.utc)
        st.rerun()

    start_local = st.session_state.start_time.astimezone(CET_TZ)
    st.caption(f"Run start: **{start_local.strftime('%Y-%m-%d %H:%M:%S %Z')}**")

# ---------- Data Prep ----------
beer_prices = parse_int_list(beer_str, DEFAULT_BEER)
shots_prices = parse_int_list(shots_str, DEFAULT_SHOTS)

n_points = int(n_points)
interval_min = int(interval_min)

# Normalize to n_points (pad with last value or trim)
beer_prices = (beer_prices + [beer_prices[-1]] * n_points)[:n_points]
shots_prices = (shots_prices + [shots_prices[-1]] * n_points)[:n_points]

schedule = build_schedule(st.session_state.start_time, interval_min, n_points)
now_utc = datetime.now(timezone.utc)

elapsed_min = (now_utc - st.session_state.start_time).total_seconds() / 60.0
current_idx = clamp(int(elapsed_min // interval_min), 0, n_points - 1)

next_tick = schedule[current_idx + 1] if current_idx < n_points - 1 else schedule[-1]
notice_window, secs_to_next = within_notice_window(now_utc, next_tick, 0, 120)

# Build editable news pools
news_pools = {
    "Beer": {
        "positive": [x for x in beer_pos.splitlines() if x.strip()],
        "neutral":  [x for x in beer_neu.splitlines() if x.strip()],
        "negative": [x for x in beer_neg.splitlines() if x.strip()],
    },
    "Shots": {
        "positive": [x for x in shots_pos.splitlines() if x.strip()],
        "neutral":  [x for x in shots_neu.splitlines() if x.strip()],
        "negative": [x for x in shots_neg.splitlines() if x.strip()],
    },
}

# ---------- DataFrames up to current point ----------
def make_df(name, prices):
    times = schedule[: current_idx + 1]
    return pd.DataFrame({
        "time": [t.astimezone(CET_TZ).replace(tzinfo=None) for t in times],
        "price": [int(v) for v in prices[: current_idx + 1]],
        "product": name
    })

beer_df = make_df("Beer", beer_prices)
shots_df = make_df("Shots", shots_prices)

# ---------- Sentiment picker ----------
def pick_news_for(series, product_label, seed_base):
    if current_idx == 0:
        options = ["positive", "neutral", "negative"]
    else:
        delta = series[current_idx] - series[current_idx - 1]
        if delta > 0:
            options = ["negative", "negative", "negative", "negative", "negative", "neutral", "neutral", "positive"]
        elif delta < 0:
            options = ["positive", "positive", "positive", "positive", "positive", "neutral", "neutral", "negative"]
        else:
            options = ["neutral", "positive", "negative"]
    rng = random.Random(seed_base + current_idx + (hash(product_label) % 9973))
    sentiment = rng.choice(options)
    pool = news_pools.get(product_label, {}).get(sentiment, [])
    msg = seeded_choice(seed_base + current_idx + (17 if product_label == "Beer" else 29), pool) or ""
    return sentiment, msg

# ---------- Strict window gating + placeholders ----------
in_notice_window = (current_idx < n_points - 1) and notice_window

if in_notice_window:
    beer_notice  = pick_news_for(beer_prices,  "Beer",  seed_base=42)
    shots_notice = pick_news_for(shots_prices, "Shots", seed_base=42)
else:
    beer_notice = None
    shots_notice = None

# ---------- Header ----------
h1, h2 = st.columns([1,1])
with h1:
    st.title("🍺 Trading Bar — Live Prices")
with h2:
    if current_idx < n_points - 1:
        st.metric("Next update in", f"{secs_to_next//60:02d}:{secs_to_next%60:02d}")
    else:
        st.metric("Session status", "Completed")

# ---------- Notices (above metrics) with explicit clear ----------
n1, n2 = st.columns([1,1])

with n1:
    st.subheader("Beer")
    beer_notice_ph = st.empty()   # placeholder ensures old banners are cleared
    if in_notice_window and beer_notice and beer_notice[1]:
        sent, msg = beer_notice
        beer_notice_ph.info(f"ℹ️ News: {msg}")
    else:
        beer_notice_ph.empty()

with n2:
    st.subheader("Sambuca Shots")
    shots_notice_ph = st.empty()
    if in_notice_window and shots_notice and shots_notice[1]:
        sent, msg = shots_notice
        shots_notice_ph.info(f"ℹ️ News: {msg}")
    else:
        shots_notice_ph.empty()

# ---------- Current Prices (metrics ABOVE charts) ----------
m1, m2 = st.columns([1, 1])

last_beer = int(beer_df["price"].iloc[-1])
prev_beer = int(beer_df["price"].iloc[-2]) if len(beer_df) > 1 else last_beer
pct_beer = ((last_beer - prev_beer) / prev_beer * 100.0) if prev_beer != 0 else 0.0
with m1:
    st.metric("Current price", f"{last_beer} DKK", delta=f"{pct_beer:+.1f}%",
              delta_color=("inverse" if invert_beer else "normal"))

last_shots = int(shots_df["price"].iloc[-1])
prev_shots = int(shots_df["price"].iloc[-2]) if len(shots_df) > 1 else last_shots
pct_shots = ((last_shots - prev_shots) / prev_shots * 100.0) if prev_shots != 0 else 0.0
with m2:
    st.metric("Current price", f"{last_shots} DKK", delta=f"{pct_shots:+.1f}%",
              delta_color=("inverse" if invert_shots else "normal"))

# ---------- Chart builder (integer-only y-axis, free range) ----------
def make_chart(df):
    y_min = int(min(df["price"]))
    y_max = int(max(df["price"]))
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    span = y_max - y_min
    step = max(1, math.ceil(span / 10))  # ~10 ticks
    axis_vals = list(range(y_min, y_max + 1, step))

    line = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("time:T", title="Time"),
        y=alt.Y("price:Q", title="Price",
                scale=alt.Scale(domain=[axis_vals[0], axis_vals[-1]], zero=False),
                axis=alt.Axis(values=axis_vals, format='d')),
        tooltip=[alt.Tooltip("time:T", title="Time"),
                 alt.Tooltip("price:Q", title="Price", format=".0f")],
    ).properties(height=380)

    rule_df = pd.DataFrame({
        "time": [schedule[min(current_idx + 1, len(schedule) - 1)].astimezone(CET_TZ).replace(tzinfo=None)]
    })
    rule = alt.Chart(rule_df).mark_rule(strokeDash=[4, 4]).encode(x="time:T")
    return line + rule

# ---------- Charts (below metrics) ----------
c1, c2 = st.columns([1, 1])
with c1:
    st.altair_chart(make_chart(beer_df), use_container_width=True)
with c2:
    st.altair_chart(make_chart(shots_df), use_container_width=True)

# ---------- Tables ----------
t1, t2 = st.columns([1,1])
with t1:
    st.caption("Beer price history (so far)")
    st.dataframe(beer_df[["time","price"]].rename(columns={"time":"Time","price":"Beer"}),
                 use_container_width=True, hide_index=True)
with t2:
    st.caption("Shot price history (so far)")
    st.dataframe(shots_df[["time","price"]].rename(columns={"time":"Time","price":"Shots"}),
                 use_container_width=True, hide_index=True)

# ---------- Footer ----------
st.markdown("---")
f1, f2 = st.columns([1,1])
with f1:
    if current_idx < n_points - 1:
        eta = schedule[current_idx + 1].astimezone(CET_TZ).strftime("%H:%M:%S %Z")
        st.caption(f"Next scheduled update at: **{eta}**")
    else:
        st.success("All scheduled updates have been shown. Use the sidebar to restart the run.")
with f2:
    st.caption("Tip: Edit the news pools per product. Messages appear ~1–2 min before each new price point.")

# ---------- Lightweight auto-refresh ----------
if current_idx < n_points - 1:
    time.sleep(2)   # fast enough to feel instant
    st.rerun()
