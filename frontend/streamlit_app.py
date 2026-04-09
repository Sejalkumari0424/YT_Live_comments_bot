# -*- coding: utf-8 -*-
import streamlit as st
import redis
import json
import pandas as pd
import plotly.graph_objects as go
import time
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from config import REDIS_HOST, REDIS_PORT, REDIS_DB

st.set_page_config(
    page_title="LivePulse",
    layout="wide",
    page_icon="📡",
    initial_sidebar_state="expanded"
)

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

TOPIC_LABELS = ["Appreciation", "Question", "Promo", "Spam", "General"]
TOPIC_COLOR  = {
    "Appreciation": "#f59e0b", "Question": "#3b82f6",
    "Promo": "#ec4899", "Spam": "#ef4444", "General": "#6b7280"
}
SENT_COLORS = {"Positive": "#22c55e", "Neutral": "#eab308", "Negative": "#ef4444"}

# ── JS: detect Streamlit's live theme and set data-livepulse attribute ──
THEME_JS = """<script>
(function() {
  function applyTheme() {
    const html = window.parent.document.documentElement;
    const style = window.parent.getComputedStyle(html);
    const bg = style.getPropertyValue('--background-color').trim();
    let isDark = true;
    const m = bg.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
    if (m) { isDark = (0.299*m[1] + 0.587*m[2] + 0.114*m[3]) < 128; }
    else {
      // fallback: check body background
      const bodyBg = window.parent.getComputedStyle(window.parent.document.body).backgroundColor;
      const m2 = bodyBg.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
      if (m2) { isDark = (0.299*m2[1] + 0.587*m2[2] + 0.114*m2[3]) < 128; }
    }
    html.setAttribute('data-livepulse', isDark ? 'dark' : 'light');
  }
  applyTheme();
  const obs = new MutationObserver(applyTheme);
  obs.observe(window.parent.document.documentElement, { attributes: true, attributeFilter: ['style','class'] });
  obs.observe(window.parent.document.body, { attributes: true, attributeFilter: ['style','class'] });
})();
</script>"""

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&display=swap');

:root, [data-livepulse="dark"] {
  --bg:#07070f; --bg-card:#0f0f1e; --border:rgba(255,255,255,0.07);
  --text-1:#f1f5f9; --text-2:#94a3b8; --text-3:#475569;
  --accent:#7c3aed; --accent2:#4f46e5; --accent-text:#a78bfa;
  --live:#22c55e; --input-bg:rgba(255,255,255,0.04); --input-border:rgba(255,255,255,0.1);
  --divider:rgba(255,255,255,0.06); --badge-bg:rgba(255,255,255,0.05);
  --shadow:0 4px 24px rgba(0,0,0,0.4); --shadow-sm:0 2px 8px rgba(0,0,0,0.3);
  --pill-bg:rgba(124,58,237,0.15); --pill-border:rgba(124,58,237,0.3); --pill-text:#a78bfa;
  --plotly-paper:rgba(0,0,0,0); --plotly-plot:rgba(255,255,255,0.015); --plotly-grid:rgba(255,255,255,0.05); --plotly-text:#94a3b8;
}
[data-livepulse="light"] {
  --bg:#f4f6ff; --bg-card:#ffffff; --border:rgba(99,102,241,0.12);
  --text-1:#0f172a; --text-2:#475569; --text-3:#94a3b8;
  --accent:#6d28d9; --accent2:#4338ca; --accent-text:#6d28d9;
  --live:#16a34a; --input-bg:#ffffff; --input-border:rgba(99,102,241,0.2);
  --divider:rgba(99,102,241,0.1); --badge-bg:rgba(99,102,241,0.06);
  --shadow:0 4px 24px rgba(99,102,241,0.12); --shadow-sm:0 2px 8px rgba(99,102,241,0.08);
  --pill-bg:rgba(109,40,217,0.08); --pill-border:rgba(109,40,217,0.2); --pill-text:#6d28d9;
  --plotly-paper:rgba(0,0,0,0); --plotly-plot:rgba(255,255,255,0.7); --plotly-grid:rgba(0,0,0,0.06); --plotly-text:#475569;
}

html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"],.main .block-container {
  background:var(--bg)!important; color:var(--text-1)!important;
  font-family:'Space Grotesk',sans-serif!important; transition:background 0.3s,color 0.3s;
}
[data-testid="stSidebar"] { background:var(--bg-card)!important; border-right:1px solid var(--border)!important; transition:background 0.3s; }
[data-testid="stHeader"] { background:transparent!important; }
::-webkit-scrollbar{width:4px;} ::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:linear-gradient(var(--accent),var(--accent2));border-radius:4px;}

[data-testid="metric-container"] {
  background:var(--bg-card)!important; border:1px solid var(--border)!important;
  border-radius:16px!important; padding:18px!important; box-shadow:var(--shadow-sm)!important; transition:background 0.3s;
}
[data-testid="stMetricLabel"]{color:var(--text-2)!important;font-size:0.8rem!important;}
[data-testid="stMetricValue"]{color:var(--text-1)!important;font-weight:700!important;}
[data-testid="stMetricDelta"]{color:var(--accent-text)!important;}

.stTextInput input { background:var(--input-bg)!important; border:1px solid var(--input-border)!important; border-radius:10px!important; color:var(--text-1)!important; }
[data-baseweb="select"]>div { background:var(--input-bg)!important; border:1px solid var(--input-border)!important; border-radius:10px!important; color:var(--text-1)!important; }
.stButton>button { background:linear-gradient(135deg,var(--accent),var(--accent2))!important; color:#fff!important; border:none!important; border-radius:10px!important; font-weight:600!important; font-family:'Space Grotesk',sans-serif!important; box-shadow:0 4px 16px rgba(124,58,237,0.3)!important; transition:all 0.2s!important; }
.stButton>button:hover{transform:translateY(-2px)!important;}
hr{border:none!important;border-top:1px solid var(--divider)!important;margin:1.2rem 0!important;}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] .stMarkdown p{color:var(--text-2)!important;font-size:0.83rem!important;}

/* Download button override — keep it subtle */
.dl-btn>button { background:var(--badge-bg)!important; color:var(--text-2)!important; border:1px solid var(--border)!important; border-radius:8px!important; font-size:0.75rem!important; padding:4px 12px!important; box-shadow:none!important; }
.dl-btn>button:hover{background:var(--pill-bg)!important;color:var(--accent-text)!important;}

@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(34,197,94,0.7);}70%{box-shadow:0 0 0 10px rgba(34,197,94,0);}100%{box-shadow:0 0 0 0 rgba(34,197,94,0);}}
.live-dot{display:inline-block;width:9px;height:9px;background:var(--live);border-radius:50%;animation:pulse 1.8s infinite;margin-right:6px;vertical-align:middle;}

.stat-grid{display:flex;gap:12px;margin:10px 0 18px;flex-wrap:wrap;}
.stat-card{flex:1;min-width:130px;background:var(--bg-card);border:1px solid var(--border);border-radius:20px;padding:22px 18px;text-align:center;transition:transform 0.2s,box-shadow 0.2s,background 0.3s;position:relative;overflow:hidden;box-shadow:var(--shadow-sm);}
.stat-card:hover{transform:translateY(-4px);box-shadow:var(--shadow);}
.stat-accent{position:absolute;top:0;left:0;right:0;height:3px;border-radius:20px 20px 0 0;}
.stat-number{font-size:2.6rem;font-weight:800;line-height:1;margin-bottom:6px;letter-spacing:-0.03em;}
.stat-label{font-size:0.82rem;color:var(--text-2);font-weight:600;text-transform:uppercase;letter-spacing:0.06em;}
.stat-sub{font-size:0.7rem;color:var(--text-3);margin-top:4px;}

.sec-hdr{display:flex;align-items:center;gap:10px;margin:6px 0 14px;}
.sec-ttl{font-size:1rem;font-weight:700;color:var(--text-1);letter-spacing:-0.01em;}
.sec-pill{background:var(--pill-bg);border:1px solid var(--pill-border);border-radius:20px;padding:2px 10px;font-size:0.68rem;color:var(--pill-text);font-weight:700;text-transform:uppercase;letter-spacing:0.08em;}

.chart-wrap{background:var(--bg-card);border:1px solid var(--border);border-radius:20px;padding:14px 14px 6px;box-shadow:var(--shadow-sm);transition:background 0.3s,border 0.3s;}
.chart-title{font-size:0.88rem;font-weight:700;color:var(--text-1);margin-bottom:2px;}
.chart-sub{font-size:0.72rem;color:var(--text-3);margin-bottom:10px;}

.topic-grid{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px;}
.topic-pill{background:var(--bg-card);border-radius:16px;padding:14px 20px;text-align:center;min-width:110px;box-shadow:var(--shadow-sm);transition:transform 0.2s,box-shadow 0.2s;}
.topic-pill:hover{transform:translateY(-3px);box-shadow:var(--shadow);}
.topic-count{font-size:1.4rem;font-weight:800;letter-spacing:-0.02em;}
.topic-name{font-size:0.7rem;color:var(--text-3);margin-top:3px;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;}

@keyframes slideIn{from{opacity:0;transform:translateY(6px);}to{opacity:1;transform:translateY(0);}}
.chat-card{background:var(--bg-card);border:1px solid var(--border);border-radius:16px;padding:14px 16px;margin-bottom:10px;border-left:3px solid transparent;animation:slideIn 0.2s ease;transition:background 0.2s,transform 0.15s,box-shadow 0.2s;box-shadow:var(--shadow-sm);}
.chat-card:hover{transform:translateX(4px);box-shadow:var(--shadow);}
.chat-positive{border-left-color:#22c55e;} .chat-negative{border-left-color:#ef4444;} .chat-neutral{border-left-color:#eab308;}
.chat-author{font-weight:700;font-size:0.83rem;color:var(--accent-text);margin-bottom:5px;}
.chat-text{font-size:0.92rem;color:var(--text-2);line-height:1.55;margin-bottom:9px;}
.chat-badges{display:flex;gap:6px;flex-wrap:wrap;}
.badge{display:inline-flex;align-items:center;background:var(--badge-bg);border:1px solid var(--border);border-radius:20px;padding:3px 10px;font-size:0.7rem;font-weight:600;color:var(--text-2);}

.empty-state{text-align:center;padding:80px 20px;background:var(--bg-card);border:1px solid var(--border);border-radius:24px;margin:40px 0;box-shadow:var(--shadow-sm);}
.empty-icon{font-size:3.5rem;margin-bottom:16px;}
.empty-title{font-size:1.1rem;color:var(--text-2);font-weight:700;}
.empty-sub{font-size:0.84rem;color:var(--text-3);margin-top:6px;}
</style>"""

st.markdown(THEME_JS, unsafe_allow_html=True)
st.markdown(CSS, unsafe_allow_html=True)


# ── HELPERS ──────────────────────────────────────────────────
def extract_video_id(url_or_id):
    url_or_id = url_or_id.strip()
    match = re.search(r"(?:v=|/live/|youtu\.be/)([A-Za-z0-9_-]{11})", url_or_id)
    if match:
        return match.group(1)
    if re.match(r"^[A-Za-z0-9_-]{11}$", url_or_id):
        return url_or_id
    return url_or_id

def update_config_video_id(video_id):
    config_path = os.path.join(os.path.dirname(__file__), '..', 'backend', 'config.py')
    with open(config_path, 'r') as f:
        content = f.read()
    content = re.sub(r'VIDEO_ID\s*=\s*".*?"', f'VIDEO_ID = "{video_id}"', content)
    with open(config_path, 'w') as f:
        f.write(content)

def clean_topic(val):
    """Normalize topic — replace None/NaN/empty with General."""
    if pd.isna(val) or str(val).strip() == "" or str(val).strip().lower() == "nan":
        return "General"
    return str(val).strip()

def clean_sentiment(val):
    if str(val).strip() in ("Positive", "Negative", "Neutral"):
        return str(val).strip()
    return "Neutral"

def plotly_layout(height=280):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(family="Space Grotesk"),
        xaxis=dict(showgrid=False, zeroline=False, showline=False,
                   tickfont=dict(size=11), title=None),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.12)",
                   zeroline=False, showline=False, tickfont=dict(size=11), title=None),
        showlegend=False,
        hoverlabel=dict(font_family="Space Grotesk", font_size=12),
    )

def csv_download(df_export, label, filename):
    csv = df_export.to_csv(index=False).encode("utf-8")
    st.download_button(label=f"⬇ {label}", data=csv,
                       file_name=filename, mime="text/csv", key=filename)


# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="padding:12px 0 20px;">'
        '<div style="font-size:1.35rem;font-weight:800;color:var(--text-1);letter-spacing:-0.02em;">📡 LivePulse</div>'
        '<div style="font-size:0.75rem;color:var(--text-3);margin-top:2px;">YouTube Chat Analytics</div>'
        '</div>', unsafe_allow_html=True
    )
    st.divider()
    st.markdown('<p style="font-size:0.68rem;font-weight:700;color:var(--accent);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">Display Settings</p>', unsafe_allow_html=True)
    refresh_rate = st.slider("Refresh interval (s)", 5, 60, 15)
    msg_limit    = st.slider("Message window", 10, 200, 50)
    auto_refresh = st.toggle("Live auto-refresh", value=True)
    st.divider()
    st.markdown('<p style="font-size:0.68rem;font-weight:700;color:#ef4444;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">Danger Zone</p>', unsafe_allow_html=True)
    if st.button("🗑 Clear all data", use_container_width=True):
        r.delete("chat_messages")
        st.success("Redis cleared.")
    st.divider()
    st.markdown(
        '<div style="font-size:0.72rem;color:var(--text-3);text-align:center;line-height:1.6;">'
        'Theme follows Streamlit settings<br>'
        '<span style="font-size:0.65rem;">☰ → Settings → Theme</span>'
        '</div>', unsafe_allow_html=True
    )

# ── PAGE HEADER ───────────────────────────────────────────────
col_title, col_live = st.columns([7, 1])
with col_title:
    st.markdown(
        '<div style="padding:8px 0 4px;">'
        '<div style="font-size:2rem;font-weight:800;color:var(--text-1);letter-spacing:-0.04em;">YouTube Live Chat Analytics</div>'
        '<div style="font-size:0.85rem;color:var(--text-3);margin-top:4px;">Real-time sentiment · topic classification · engagement insights</div>'
        '</div>', unsafe_allow_html=True
    )
with col_live:
    st.markdown(
        '<div style="text-align:right;padding-top:22px;">'
        '<span class="live-dot"></span>'
        '<span style="font-size:0.78rem;color:var(--live);font-weight:700;letter-spacing:0.05em;">LIVE</span>'
        '</div>', unsafe_allow_html=True
    )

st.divider()

# ── DATA LOAD ─────────────────────────────────────────────────
all_raw  = r.lrange("chat_messages", 0, -1)
all_data = [json.loads(m) for m in all_raw]
raw      = r.lrange("chat_messages", -msg_limit, -1)
data     = [json.loads(m) for m in raw]

if not all_data:
    st.markdown(
        '<div class="empty-state">'
        '<div class="empty-icon">📭</div>'
        '<div class="empty-title">No messages yet</div>'
        '<div class="empty-sub">Set a video ID in the sidebar, then run <code>python scraper.py</code></div>'
        '</div>', unsafe_allow_html=True
    )
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()
    st.stop()

df     = pd.DataFrame(data)
all_df = pd.DataFrame(all_data)

# Normalize — kill undefined/NaN values
df["sentiment"]  = df["sentiment"].apply(clean_sentiment)
df["topic"]      = df["topic"].apply(clean_topic) if "topic" in df.columns else "General"
all_df["sentiment"] = all_df["sentiment"].apply(clean_sentiment)
all_df["topic"]     = all_df["topic"].apply(clean_topic) if "topic" in all_df.columns else "General"

# ── CUMULATIVE STATS ──────────────────────────────────────────
all_counts = all_df["sentiment"].value_counts().to_dict()
c_pos   = all_counts.get("Positive", 0)
c_neu   = all_counts.get("Neutral",  0)
c_neg   = all_counts.get("Negative", 0)
c_total = max(c_pos + c_neu + c_neg, 1)

st.markdown(
    '<div class="sec-hdr"><span class="sec-ttl">Cumulative Sentiment</span><span class="sec-pill">All Time</span></div>',
    unsafe_allow_html=True
)
st.markdown(
    f'<div class="stat-grid">'
    f'<div class="stat-card"><div class="stat-accent" style="background:linear-gradient(90deg,#22c55e,#16a34a);"></div>'
    f'<div class="stat-number" style="color:#22c55e;">{c_pos}</div><div class="stat-label">Positive</div><div class="stat-sub">{c_pos/c_total*100:.1f}% of total</div></div>'
    f'<div class="stat-card"><div class="stat-accent" style="background:linear-gradient(90deg,#eab308,#ca8a04);"></div>'
    f'<div class="stat-number" style="color:#eab308;">{c_neu}</div><div class="stat-label">Neutral</div><div class="stat-sub">{c_neu/c_total*100:.1f}% of total</div></div>'
    f'<div class="stat-card"><div class="stat-accent" style="background:linear-gradient(90deg,#ef4444,#dc2626);"></div>'
    f'<div class="stat-number" style="color:#ef4444;">{c_neg}</div><div class="stat-label">Negative</div><div class="stat-sub">{c_neg/c_total*100:.1f}% of total</div></div>'
    f'<div class="stat-card"><div class="stat-accent" style="background:linear-gradient(90deg,#7c3aed,#4f46e5);"></div>'
    f'<div class="stat-number" style="color:var(--accent-text);">{c_total}</div><div class="stat-label">Total</div><div class="stat-sub">all time</div></div>'
    f'</div>',
    unsafe_allow_html=True
)

# ── WINDOW METRICS ────────────────────────────────────────────
st.divider()
counts = df["sentiment"].value_counts().to_dict()
pos    = counts.get("Positive", 0)
neu    = counts.get("Neutral",  0)
neg    = counts.get("Negative", 0)
total  = max(pos + neu + neg, 1)

st.markdown(
    f'<div class="sec-hdr"><span class="sec-ttl">Window Snapshot</span><span class="sec-pill">Last {msg_limit} msgs</span></div>',
    unsafe_allow_html=True
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Messages",  total)
c2.metric("Positive",  pos,  f"{pos/total*100:.1f}%")
c3.metric("Neutral",   neu,  f"{neu/total*100:.1f}%")
c4.metric("Negative",  neg,  f"{neg/total*100:.1f}%")

# ── SENTIMENT CHARTS ──────────────────────────────────────────
st.divider()
col_l, col_r = st.columns(2)

# ── Bar chart ──
with col_l:
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Sentiment Distribution</div><div class="chart-sub">Message count by sentiment class</div>', unsafe_allow_html=True)

    fig_bar = go.Figure(go.Bar(
        x=["Positive", "Neutral", "Negative"],
        y=[pos, neu, neg],
        marker_color=["#22c55e", "#eab308", "#ef4444"],
        marker_line_width=0,
        text=[pos, neu, neg],
        textposition="outside",
        textfont=dict(size=12),
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
    ))
    fig_bar.update_layout(**plotly_layout(260))
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    bar_hdr, bar_dl = st.columns([1, 1])
    with bar_hdr:
        show_bar_data = st.checkbox("View data", key="show_bar")
    with bar_dl:
        bar_df = pd.DataFrame({"Sentiment": ["Positive", "Neutral", "Negative"], "Count": [pos, neu, neg]})
        csv_download(bar_df, "Download CSV", "sentiment_distribution.csv")

    if show_bar_data:
        st.dataframe(bar_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Donut chart ──
with col_r:
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Sentiment Breakdown</div><div class="chart-sub">Proportional share per class</div>', unsafe_allow_html=True)

    fig_pie = go.Figure(go.Pie(
        labels=["Positive", "Neutral", "Negative"],
        values=[pos, neu, neg],
        marker_colors=["#22c55e", "#eab308", "#ef4444"],
        hole=0.58,
        textinfo="percent",
        hovertemplate="<b>%{label}</b><br>%{value} messages (%{percent})<extra></extra>",
    ))
    fig_pie.update_layout(
        **{**plotly_layout(260),
           "showlegend": True,
           "legend": dict(orientation="h", y=-0.08, font=dict(size=11))}
    )
    st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    pie_hdr, pie_dl = st.columns([1, 1])
    with pie_hdr:
        show_pie_data = st.checkbox("View data", key="show_pie")
    with pie_dl:
        pie_df = pd.DataFrame({
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Count": [pos, neu, neg],
            "Percentage": [f"{pos/total*100:.1f}%", f"{neu/total*100:.1f}%", f"{neg/total*100:.1f}%"]
        })
        csv_download(pie_df, "Download CSV", "sentiment_breakdown.csv")

    if show_pie_data:
        st.dataframe(pie_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Confidence trend ──────────────────────────────────────────
if "confidence" in df.columns:
    st.divider()
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Confidence Trend</div><div class="chart-sub">Model confidence per message in current window</div>', unsafe_allow_html=True)

    conf_df = df[["confidence"]].reset_index(drop=True)
    conf_df.index.name = "message_index"

    fig_line = go.Figure(go.Scatter(
        x=conf_df.index,
        y=conf_df["confidence"],
        mode="lines",
        line=dict(color="#7c3aed", width=2),
        fill="tozeroy",
        fillcolor="rgba(124,58,237,0.08)",
        hovertemplate="Msg %{x}: <b>%{y:.2f}</b><extra></extra>",
    ))
    fig_line.update_layout(**plotly_layout(180))
    fig_line.update_yaxes(range=[0, 1])
    st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})

    conf_hdr, conf_dl = st.columns([1, 1])
    with conf_hdr:
        show_conf_data = st.checkbox("View data", key="show_conf")
    with conf_dl:
        conf_export = conf_df.reset_index()
        conf_export.columns = ["message_index", "confidence"]
        csv_download(conf_export, "Download CSV", "confidence_trend.csv")

    if show_conf_data:
        st.dataframe(conf_export, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── TOPIC DISTRIBUTION ────────────────────────────────────────
st.divider()
st.markdown(
    '<div class="sec-hdr"><span class="sec-ttl">Topic Distribution</span><span class="sec-pill">All Time</span></div>',
    unsafe_allow_html=True
)

topic_counts = {
    label: int((all_df["topic"] == label).sum())
    for label in TOPIC_LABELS
}

# Topic pill cards
pills = '<div class="topic-grid">'
for label in TOPIC_LABELS:
    color = TOPIC_COLOR[label]
    count = topic_counts[label]
    pills += (
        f'<div class="topic-pill" style="border:1px solid {color}44;">'
        f'<div class="topic-count" style="color:{color};">{count}</div>'
        f'<div class="topic-name">{label}</div>'
        f'</div>'
    )
pills += '</div>'
st.markdown(pills, unsafe_allow_html=True)

st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
st.markdown('<div class="chart-title">Topic Breakdown</div><div class="chart-sub">All-time message count per topic category</div>', unsafe_allow_html=True)

fig_topic = go.Figure(go.Bar(
    x=TOPIC_LABELS,
    y=[topic_counts[l] for l in TOPIC_LABELS],
    marker_color=[TOPIC_COLOR[l] for l in TOPIC_LABELS],
    marker_line_width=0,
    text=[topic_counts[l] for l in TOPIC_LABELS],
    textposition="outside",
    textfont=dict(size=11),
    hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
))
fig_topic.update_layout(**plotly_layout(250))
st.plotly_chart(fig_topic, use_container_width=True, config={"displayModeBar": False})

topic_hdr, topic_dl = st.columns([1, 1])
with topic_hdr:
    show_topic_data = st.checkbox("View data", key="show_topic")
with topic_dl:
    topic_df = pd.DataFrame({"Topic": TOPIC_LABELS, "Count": [topic_counts[l] for l in TOPIC_LABELS]})
    csv_download(topic_df, "Download CSV", "topic_distribution.csv")

if show_topic_data:
    st.dataframe(topic_df, use_container_width=True, hide_index=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── LIVE CHAT FEED ────────────────────────────────────────────
st.divider()
st.markdown('<div class="sec-hdr"><span class="sec-ttl">Live Chat Feed</span></div>', unsafe_allow_html=True)

f1, f2, f3 = st.columns([1, 1, 2])
with f1:
    sentiment_filter = st.selectbox("Sentiment", ["All", "Positive", "Neutral", "Negative"])
with f2:
    topic_filter = st.selectbox("Topic", ["All"] + TOPIC_LABELS)
with f3:
    search_term = st.text_input("Search messages", placeholder="Filter by keyword...")

filtered = df.copy()
if sentiment_filter != "All":
    filtered = filtered[filtered["sentiment"] == sentiment_filter]
if topic_filter != "All":
    filtered = filtered[filtered["topic"] == topic_filter]
if search_term:
    filtered = filtered[filtered["text"].str.contains(search_term, case=False, na=False)]

feed_hdr, feed_dl = st.columns([3, 1])
with feed_hdr:
    st.markdown(
        f'<div style="font-size:0.78rem;color:var(--text-3);margin-bottom:12px;">Showing {len(filtered)} of {len(df)} messages</div>',
        unsafe_allow_html=True
    )
with feed_dl:
    if not filtered.empty:
        csv_download(filtered[["author","text","sentiment","confidence","topic","time"]]
                     if all(c in filtered.columns for c in ["author","text","sentiment","confidence","topic","time"])
                     else filtered,
                     "Download Feed CSV", "chat_feed.csv")

SENT_ICON = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}

for _, row in filtered.iloc[::-1].iterrows():
    s         = row.get("sentiment", "Neutral")
    conf_pct  = int(row.get("confidence", 0) * 100)
    topic     = clean_topic(row.get("topic", "General"))
    t_color   = TOPIC_COLOR.get(topic, "#6b7280")
    s_color   = SENT_COLORS.get(s, "#6b7280")
    s_icon    = SENT_ICON.get(s, "⚪")
    conf_color = "#22c55e" if conf_pct >= 70 else "#eab308" if conf_pct >= 40 else "#ef4444"

    st.markdown(
        f'<div class="chat-card chat-{s.lower()}">'
        f'<div class="chat-author">{s_icon} {row.get("author", "Unknown")}</div>'
        f'<div class="chat-text">{row.get("text", "")}</div>'
        f'<div class="chat-badges">'
        f'<span class="badge" style="color:{s_color};border-color:{s_color}33;">{s}</span>'
        f'<span class="badge" style="color:{conf_color};">Confidence: {conf_pct}%</span>'
        f'<span class="badge" style="color:{t_color};border-color:{t_color}33;">{topic}</span>'
        f'</div></div>',
        unsafe_allow_html=True
    )

# ── AUTO REFRESH ──────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
