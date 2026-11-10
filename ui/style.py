from __future__ import annotations

import html as _html

import streamlit as st

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important}
.main .block-container{padding-top:1.4rem;padding-bottom:2rem;max-width:1150px}
[data-testid="stSidebar"]{background:#0f172a!important;border-right:1px solid #1e293b!important}
[data-testid="stSidebar"] *{color:#e2e8f0!important}
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"]{display:none!important}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"]{display:flex!important;flex-direction:column!important;gap:2px!important}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label{display:flex!important;align-items:center!important;padding:10px 14px!important;border-radius:10px!important;cursor:pointer!important;border-left:3px solid transparent!important}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover{background:#1e3a5f!important;border-left-color:#60a5fa!important}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p{font-size:0.95rem!important;font-weight:500!important;margin:0!important}
.hero{background:linear-gradient(135deg,#0d1b2a 0%,#1a3a5c 55%,#9a3412 100%);border-radius:16px;padding:28px;color:#fff;margin-bottom:1.2rem}
.hero h1{font-size:1.9rem;font-weight:800;margin:0 0 6px;color:#fff}
.hero p{font-size:0.98rem;opacity:0.9;margin:0;color:#e2e8f0}
.pill{display:inline-block;background:rgba(255,255,255,0.14);padding:3px 11px;border-radius:20px;font-size:0.76rem;margin:8px 6px 0 0;color:#fff}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:16px 18px;margin-bottom:12px;box-shadow:0 1px 4px rgba(15,23,42,0.05)}
.card h3{margin:0 0 4px;font-size:1.02rem;color:#0f172a}
.muted{color:#64748b;font-size:0.8rem}
.kpis{display:flex;gap:10px;flex-wrap:wrap;margin:0.6rem 0 1rem}
.kpi{background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:10px 14px;flex:1;min-width:120px}
.kpi .v{font-size:1.45rem;font-weight:800;color:#0f172a;line-height:1.2}
.kpi .l{font-size:0.68rem;color:#64748b;text-transform:uppercase;letter-spacing:0.5px}
.kpi .s{font-size:0.72rem;color:#64748b}
.badge{display:inline-block;padding:2px 9px;border-radius:10px;font-size:0.72rem;font-weight:700;color:#fff;margin-right:4px}
.verdict{border-radius:12px;padding:12px 14px;margin-bottom:10px;border:1px solid #e2e8f0;border-left-width:5px}
.verdict.SUPPORTED{border-left-color:#16a34a;background:#f0fdf4}
.verdict.REFUTED{border-left-color:#dc2626;background:#fef2f2}
.verdict.NOT_ENOUGH_INFO{border-left-color:#94a3b8;background:#f8fafc}
.evidence{font-size:0.8rem;color:#334155;background:#fff;border:1px dashed #cbd5e1;border-radius:8px;padding:8px 10px;margin-top:6px}
.bar{background:#e2e8f0;border-radius:6px;height:9px;overflow:hidden;margin:3px 0 8px}
.bar>div{height:100%;border-radius:6px}
.split{height:24px;border-radius:12px;overflow:hidden;display:flex;margin:0.6rem 0;font-size:0.72rem;color:#fff;font-weight:700}
.split>div{display:flex;align-items:center;justify-content:center;min-width:28px}
.sec{font-size:1.05rem;font-weight:700;color:#0f172a;margin:1.3rem 0 0.6rem;padding-bottom:5px;border-bottom:2px solid #e2e8f0}
.step{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:0.8rem;background:#0f172a;color:#e2e8f0;border-radius:8px;padding:8px 10px;margin:4px 0}
.reply{background:linear-gradient(135deg,#eff6ff,#dbeafe);border:1px solid #bfdbfe;border-radius:12px;padding:14px 16px;margin:0.6rem 0}
.synthetic{background:#fef3c7;color:#92400e;border:1px solid #fde68a;border-radius:10px;padding:8px 12px;font-size:0.8rem;margin-bottom:0.8rem}
</style>"""

VERDICT_COLOR = {"SUPPORTED": "#16a34a", "REFUTED": "#dc2626", "NOT_ENOUGH_INFO": "#64748b"}
STANCE_COLOR = {"Support": "#16a34a", "Oppose": "#dc2626", "Neutral": "#64748b"}
CATEGORY_COLOR = {"Politics": "#dc2626", "Social": "#2563eb", "Environment": "#16a34a", "Economy": "#d97706"}


def inject() -> None:
    st.html(CSS)


def esc(text: object) -> str:
    return _html.escape(str(text if text is not None else ""))


def badge(text: str, color: str) -> str:
    return f'<span class="badge" style="background:{color}">{esc(text)}</span>'


def kpis(items: list[tuple[str, str, str]]) -> None:
    cells = "".join(f'<div class="kpi"><div class="v">{esc(v)}</div><div class="l">{esc(label)}</div>'
                    f'<div class="s">{esc(sub)}</div></div>' for label, v, sub in items)
    st.html(f'<div class="kpis">{cells}</div>')


def bar(value: float | None, color: str) -> str:
    width = 0 if value is None else max(2, min(100, value))
    return f'<div class="bar"><div style="width:{width}%;background:{color}"></div></div>'


def stance_split(counts: dict[str, int]) -> str:
    total = sum(counts.values()) or 1
    parts = []
    for stance in ("Support", "Neutral", "Oppose"):
        pct = round(100 * counts.get(stance, 0) / total)
        if pct:
            parts.append(f'<div style="width:{pct}%;background:{STANCE_COLOR[stance]}">{pct}% {stance}</div>')
    return f'<div class="split">{"".join(parts)}</div>'


def section(title: str) -> None:
    st.html(f'<div class="sec">{esc(title)}</div>')
