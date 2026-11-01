import streamlit as st
import json
import re
from collections import Counter
from groq import Groq
from supabase import create_client
import plotly.graph_objects as go

GROQ_API_KEY  = st.secrets["GROQ_API_KEY"]
SUPABASE_URL  = st.secrets["SUPABASE_URL"]
SUPABASE_KEY  = st.secrets["SUPABASE_KEY"]
AUTHORITY_PWD = st.secrets["AUTHORITY_PWD"]

INDIAN_STATES = ["Not Specified","Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh","Goa","Gujarat","Haryana","Himachal Pradesh","Jharkhand","Karnataka","Kerala","Madhya Pradesh","Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland","Odisha","Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura","Uttar Pradesh","Uttarakhand","West Bengal","Andaman and Nicobar Islands","Chandigarh","Dadra and Nagar Haveli","Daman and Diu","Delhi","Jammu and Kashmir","Ladakh","Lakshadweep","Puducherry"]
CAT_CLS = {"Politics":"#e74c3c","Social":"#3498db","Environment":"#27ae60","Economy":"#f39c12"}

@st.cache_resource
def get_clients():
    return Groq(api_key=GROQ_API_KEY), create_client(SUPABASE_URL, SUPABASE_KEY)

def db_get_protests():
    try:
        _, sb = get_clients()
        return sb.table("protests").select("*").order("id").execute().data or []
    except Exception as e:
        st.error(f"DB error: {e}"); return []

def db_save_opinion(iid, ititle, uname, state, stance, text, score, analysis):
    try:
        _, sb = get_clients()
        sb.table("opinions").insert({"issue_id":str(iid),"issue_title":ititle,"user_name":uname,"state":state,"stance":stance,"opinion_text":text,"credibility_score":score,"ai_analysis":analysis}).execute()
        return True
    except Exception as e:
        st.error(f"Save error: {e}"); return False

def db_get_opinions(issue_id, user_name=None):
    try:
        _, sb = get_clients()
        q = sb.table("opinions").select("*").eq("issue_id",str(issue_id)).order("created_at",desc=True)
        if user_name: q = q.eq("user_name", user_name)
        return q.execute().data or []
    except Exception as e:
        st.error(f"DB error: {e}"); return []

def db_update_opinion(oid, text):
    try:
        _, sb = get_clients()
        sb.table("opinions").update({"opinion_text":text}).eq("id",oid).execute()
        return True
    except Exception as e:
        st.error(f"Update error: {e}"); return False

def db_get_replies(issue_title):
    try:
        _, sb = get_clients()
        return sb.table("authority_replies").select("*").eq("issue_title",issue_title).order("created_at",desc=True).execute().data or []
    except Exception as e:
        st.error(f"DB error: {e}"); return []

def db_save_reply(iid, ititle, aname, rtext, status):
    try:
        _, sb = get_clients()
        sb.table("authority_replies").insert({"issue_id":str(iid),"issue_title":ititle,"authority_name":aname,"reply_text":rtext,"status":status}).execute()
        return True
    except Exception as e:
        st.error(f"Save error: {e}"); return False

def db_save_reaction(iid, reaction):
    try:
        _, sb = get_clients()
        sb.table("citizen_reactions").insert({"issue_id":str(iid),"reaction":reaction}).execute()
        return True
    except Exception as e:
        return False

def db_get_reactions(iid):
    try:
        _, sb = get_clients()
        data = sb.table("citizen_reactions").select("reaction").eq("issue_id",str(iid)).execute().data or []
        return sum(1 for r in data if r["reaction"]=="up"), sum(1 for r in data if r["reaction"]=="down")
    except: return 0, 0

def db_get_all_opinions():
    try:
        _, sb = get_clients()
        return sb.table("opinions").select("*").order("created_at",desc=True).execute().data or []
    except Exception as e:
        st.error(f"DB error: {e}"); return []

def ai_call(prompt, max_tokens=800):
    try:
        groq, _ = get_clients()
        return groq.chat.completions.create(model="openai/gpt-oss-20b",messages=[{"role":"user","content":prompt}],max_tokens=max_tokens,temperature=0.7).choices[0].message.content.strip()
    except Exception as e: return f"AI unavailable: {e}"

def ai_summary(title, description):
    key = f"sum_{title[:25]}"
    if key not in st.session_state:
        st.session_state[key] = ai_call(f"In 2-3 sentences, summarise this Indian protest for a citizen platform:\nTitle: {title}\nContext: {description}\nBe factual, neutral, concise.")
    return st.session_state[key]

def ai_analyze(title, summary, stance, opinion):
    raw = ai_call(f'Analyse this citizen opinion. Return ONLY valid JSON.\nProtest: {title}\nSummary: {summary}\nStance: {stance}\nOpinion: {opinion}\nReturn: {{"credibility_score":1-100,"factual_accuracy":"High/Medium/Low","emotional_bias":"High/Medium/Low","key_points":["p1","p2"],"improvement_suggestion":"brief","overall_assessment":"one sentence"}}', 500)
    try:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        return json.loads(m.group()) if m else {}
    except: return {}

def ai_extract_reasons(title, opinions):
    key = f"reasons_{title[:20]}_{len(opinions)}"
    if key in st.session_state: return st.session_state[key]
    if not opinions: return {"for":[],"against":[]}
    texts = "\n".join([f"[{o['stance'].upper()}] {o['opinion_text'][:200]}" for o in opinions[:15]])
    raw = ai_call(f'Extract top 3 reasons FOR and AGAINST this protest from opinions.\nProtest: {title}\nOpinions:\n{texts}\nReturn ONLY JSON: {{"for":["r1","r2","r3"],"against":["r1","r2","r3"]}}', 400)
    try:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        result = json.loads(m.group()) if m else {"for":[],"against":[]}
    except: result = {"for":[],"against":[]}
    st.session_state[key] = result
    return result

def ai_rate_reply(title, reply_text):
    raw = ai_call(f'Rate this authority reply on 1-10 scale.\nProtest: {title}\nReply: {reply_text}\nReturn ONLY JSON: {{"specificity":1-10,"addresses_concerns":1-10,"promises_action":1-10,"overall":1-10,"assessment":"one sentence"}}', 300)
    try:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        return json.loads(m.group()) if m else {}
    except: return {}

def ai_report(title, opinions):
    if not opinions: return "No opinions submitted yet."
    texts = "\n".join([f"[{o['stance'].upper()}|Score:{o.get('credibility_score','?')}] {o['opinion_text'][:150]}" for o in opinions[:20]])
    return ai_call(f"Generate a formal civic report on '{title}' based on {len(opinions)} citizen opinions.\nSample:\n{texts}\nInclude: Executive Summary, Key Findings, Sentiment Analysis, Recommendations. Under 400 words.", 600)

def status_chip(status):
    m = {"Ongoing":("#fff3cd","#856404","🔴"),"Authority Replied":("#d1ecf1","#0c5460","🔵"),"Resolved":("#d4edda","#155724","✅")}
    bg,fg,icon = m.get(status,("#f8f9fa","#495057","⚪"))
    return f'<span style="background:{bg};color:{fg};padding:2px 10px;border-radius:12px;font-size:0.75rem;font-weight:600">{icon} {status}</span>'

def cat_badge(cat):
    c = CAT_CLS.get(cat,"#6c757d")
    return f'<span style="background:{c};color:#fff;padding:2px 8px;border-radius:10px;font-size:0.7rem;font-weight:600">{cat}</span>'

def strength_info(avg):
    if avg>=75: return "Highly Credible","#27ae60",85
    elif avg>=55: return "Moderately Credible","#f39c12",60
    elif avg>=35: return "Low Credibility","#e67e22",35
    else: return "Very Low Credibility","#e74c3c",15

def h(html): st.html(html)

def inject_css():
    st.html("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important}
.main .block-container{padding-top:1.5rem;padding-bottom:2rem;max-width:1100px}
[data-testid="stSidebar"]{background:#0f172a!important;border-right:1px solid #1e293b!important;box-shadow:2px 0 16px rgba(0,0,0,0.3)!important}
[data-testid="stSidebar"] *{color:#e2e8f0!important}
[data-testid="stSidebar"] .stRadio>div>div{gap:2px!important}
[data-testid="stSidebar"] .stRadio label{border-radius:10px!important;padding:11px 16px!important;margin:1px 0!important;transition:all 0.15s!important;border-left:3px solid transparent!important}
[data-testid="stSidebar"] .stRadio label:hover{background:#1e3a5f!important;border-left-color:#60a5fa!important}
[data-testid="stSidebar"] .stRadio label p{font-size:0.97rem!important;font-weight:500!important;margin:0!important;color:#e2e8f0!important}
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"]{display:none!important}
.protest-card{background:#fff;border:1px solid #e2e8f0;border-radius:16px;overflow:hidden;margin-bottom:1rem;box-shadow:0 2px 8px rgba(0,0,0,0.06);transition:all 0.3s}
.protest-card:hover{transform:translateY(-3px);box-shadow:0 8px 24px rgba(0,0,0,0.12)}
.card-hdr{background:linear-gradient(135deg,#1a3a5c,#0d2137);padding:16px 20px 12px}
.card-hdr h3{margin:0;font-size:1rem;font-weight:700;color:#fff;line-height:1.4}
.card-bdy{padding:14px 20px 16px}
.card-meta{font-size:0.75rem;color:#718096;margin:4px 0}
.card-desc{font-size:0.85rem;color:#4a5568;margin:8px 0;line-height:1.6}
.card-stats{display:flex;gap:16px;font-size:0.8rem;color:#718096;margin-top:8px;flex-wrap:wrap}
.stats-row{display:flex;gap:12px;flex-wrap:wrap;margin:1rem 0}
.stat-box{background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:12px 18px;text-align:center;flex:1;min-width:90px}
.stat-val{font-size:1.8rem;font-weight:800;color:#1a3a5c}
.stat-lbl{font-size:0.7rem;color:#718096;text-transform:uppercase;letter-spacing:0.5px}
.split-bar{height:28px;border-radius:14px;overflow:hidden;display:flex;background:#f0f0f0;margin:1rem 0}
.split-for{background:linear-gradient(90deg,#27ae60,#2ecc71);display:flex;align-items:center;justify-content:flex-end;padding-right:8px;font-size:0.75rem;color:#fff;font-weight:700;min-width:30px}
.split-agn{background:linear-gradient(90deg,#e74c3c,#c0392b);display:flex;align-items:center;padding-left:8px;font-size:0.75rem;color:#fff;font-weight:700;min-width:30px}
.str-meter{background:#f0f0f0;border-radius:10px;height:18px;overflow:hidden;margin:6px 0}
.str-fill{height:100%;border-radius:10px}
.for-card{background:#d4edda;border-left:4px solid #27ae60;border-radius:8px;padding:10px 14px;margin-bottom:8px;font-size:0.85rem;color:#155724}
.agn-card{background:#f8d7da;border-left:4px solid #dc3545;border-radius:8px;padding:10px 14px;margin-bottom:8px;font-size:0.85rem;color:#3c1a1a}
.reply-card{background:linear-gradient(135deg,#e8f4fd,#d6eaf8);border:1px solid #bee3f8;border-radius:12px;padding:18px;margin:1rem 0}
.reply-card h4{color:#0c5460;margin:0 0 8px;font-size:0.95rem}
.reply-meta{font-size:0.75rem;color:#718096;margin-bottom:10px}
.reply-txt{font-size:0.88rem;color:#2d3748;line-height:1.7}
.op-card{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:14px;margin-bottom:10px}
.score-badge{display:inline-block;background:#1a3a5c;color:#fff;border-radius:20px;padding:2px 10px;font-size:0.75rem;font-weight:700}
.outcome-card{background:linear-gradient(135deg,#d4edda,#c3e6cb);border:1px solid #b8dacc;border-radius:12px;padding:20px;margin:1rem 0}
.rating-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px;margin:12px 0}
.rating-item{background:#fff;border-radius:10px;padding:12px;text-align:center;border:1px solid #e2e8f0}
.rating-score{font-size:1.5rem;font-weight:800;color:#1a3a5c}
.rating-lbl{font-size:0.7rem;color:#718096;text-transform:uppercase}
.sec-hdr{font-size:1.1rem;font-weight:700;color:#1a3a5c;margin:1.5rem 0 0.8rem;padding-bottom:6px;border-bottom:2px solid #e2e8f0}
.hero{background:linear-gradient(135deg,#0d1b2a 0%,#1a3a5c 50%,#c0392b 100%);border-radius:16px;padding:32px 28px;color:#fff;margin-bottom:1.5rem;text-align:center}
.hero h1{font-size:2rem;font-weight:800;margin:0 0 8px;color:#fff}
.hero p{font-size:1rem;opacity:0.9;margin:0;color:#e8f4f8}
</style>""")

def page_home():
    h('<div class="hero"><h1>🇮🇳 FactProtest India</h1><p>AI-Powered Protest Credibility Checker · Civic Voice Platform</p><div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-top:12px"><span style="background:rgba(255,255,255,0.15);padding:4px 12px;border-radius:20px;font-size:0.8rem">🤖 AI Opinion Scoring</span><span style="background:rgba(255,255,255,0.15);padding:4px 12px;border-radius:20px;font-size:0.8rem">📊 Credibility Analysis</span><span style="background:rgba(255,255,255,0.15);padding:4px 12px;border-radius:20px;font-size:0.8rem">🏛 Authority Reply Rater</span><span style="background:rgba(255,255,255,0.15);padding:4px 12px;border-radius:20px;font-size:0.8rem">📄 AI Civic Reports</span></div></div>')
    protests = db_get_protests()
    if not protests:
        st.warning("No protests found. Check your Supabase connection."); return
    col1,col2,col3 = st.columns([3,1.5,1.5])
    with col1: search = st.text_input("🔍 Search protests", placeholder="Type protest name...")
    with col2: level_f = st.selectbox("Level",["All","Central","State"])
    with col3: status_f = st.selectbox("Status",["All","Ongoing","Authority Replied","Resolved"])
    cat_f = st.radio("Category",["All","Politics","Social","Environment","Economy"],horizontal=True)
    filtered = protests
    if search:
        sl = search.lower(); filtered = [p for p in filtered if sl in p["title"].lower() or sl in p["description"].lower()]
    if level_f!="All": filtered = [p for p in filtered if p.get("level")==level_f]
    if status_f!="All": filtered = [p for p in filtered if p.get("status")==status_f]
    if cat_f!="All": filtered = [p for p in filtered if p.get("category")==cat_f]
    st.markdown(f"**{len(filtered)} protest(s) found**")
    for p in filtered:
        opinions = db_get_opinions(p["id"])
        total = len(opinions)
        avg = int(sum(o.get("credibility_score",0) or 0 for o in opinions)/total) if total else 0
        fc = sum(1 for o in opinions if o.get("stance")=="Support")
        ac = sum(1 for o in opinions if o.get("stance")=="Oppose")
        stats = f'<span>💬 {total} opinions</span><span style="margin-left:12px">✅ {fc} support</span><span style="margin-left:12px">❌ {ac} oppose</span>'
        if total:
            lbl,col,_ = strength_info(avg)
            stats += f'<span style="margin-left:12px">⭐ {avg}/100</span>'
            ai_badge = f'<span style="background:{col};color:#fff;padding:2px 10px;border-radius:10px;font-size:0.72rem;font-weight:700">🤖 {lbl}</span>'
        else:
            ai_badge = '<span style="background:#e2e8f0;color:#718096;padding:2px 10px;border-radius:10px;font-size:0.72rem">🤖 No AI score yet</span>'
        h(f'<div class="protest-card"><div class="card-hdr"><h3>{p["title"]}</h3><div style="margin-top:6px;display:flex;gap:6px;flex-wrap:wrap">{cat_badge(p.get("category",""))} {status_chip(p.get("status","Ongoing"))} {ai_badge}</div></div><div class="card-bdy"><div class="card-meta">📍 {p.get("location","India")} &nbsp;|&nbsp; 📅 {p.get("date_started","")} &nbsp;|&nbsp; 🏛 {p.get("level","Central")}</div><p class="card-desc">{p["description"][:200]}...</p><div class="card-stats">{stats}</div></div></div>')
        if st.button("View Details →", key=f"btn_{p['id']}"):
            st.session_state["page"]="detail"; st.session_state["sel"]=p["id"]; st.rerun()

def page_detail():
    protests = db_get_protests()
    protest = next((p for p in protests if p["id"]==st.session_state.get("sel")), None)
    if not protest:
        st.error("Protest not found.")
        if st.button("← Back"): st.session_state["page"]="home"; st.rerun()
        return
    iid,title,status = protest["id"],protest["title"],protest.get("status","Ongoing")
    if st.button("← Back to Home"): st.session_state["page"]="home"; st.rerun()
    h(f'<div style="background:linear-gradient(135deg,#0d1b2a,#1a3a5c);border-radius:16px;padding:24px;color:#fff;margin-bottom:1rem"><div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px"><div>{cat_badge(protest.get("category",""))} {status_chip(status)}</div><div style="font-size:0.8rem;opacity:0.8">📅 {protest.get("date_started","")} | 📍 {protest.get("location","")}</div></div><h2 style="color:#fff;margin:12px 0 6px;font-size:1.3rem">{title}</h2><p style="font-size:0.9rem;opacity:0.85;margin:0;line-height:1.6">{protest.get("description","")}</p></div>')
    opinions = db_get_opinions(iid)
    total = len(opinions)
    fc = sum(1 for o in opinions if o.get("stance")=="Support")
    ac = sum(1 for o in opinions if o.get("stance")=="Oppose")
    avg = int(sum(o.get("credibility_score",0) or 0 for o in opinions)/total) if total else 0
    label,color,fill = strength_info(avg)
    replies = db_get_replies(title)
    h(f'<div class="stats-row"><div class="stat-box"><div class="stat-val">{total}</div><div class="stat-lbl">Total Opinions</div></div><div class="stat-box"><div class="stat-val" style="color:#27ae60">{fc}</div><div class="stat-lbl">Support</div></div><div class="stat-box"><div class="stat-val" style="color:#e74c3c">{ac}</div><div class="stat-lbl">Oppose</div></div><div class="stat-box"><div class="stat-val">{avg}</div><div class="stat-lbl">Avg Score</div></div><div class="stat-box"><div class="stat-val">{len(replies)}</div><div class="stat-lbl">Auth Replies</div></div></div>')
    if status!="Resolved":
        h('<div class="sec-hdr">📝 Share Your Opinion</div>')
        with st.form(f"form_{iid}", clear_on_submit=True):
            uname = st.text_input("Your Name *")
            state = st.selectbox("Your State", INDIAN_STATES)
            stance = st.radio("Stance",["Support","Oppose","Neutral"],horizontal=True)
            otxt = st.text_area("Your Opinion *",placeholder="Share your views...",height=120)
            sub = st.form_submit_button("🚀 Submit & Get AI Score",use_container_width=True)
        if sub:
            if not uname.strip() or not otxt.strip():
                st.error("Name and opinion required.")
            else:
                with st.spinner("AI analysing..."):
                    summ = ai_summary(title, protest.get("description",""))
                    analysis = ai_analyze(title,summ,stance,otxt)
                    score = analysis.get("credibility_score",50)
                    if db_save_opinion(iid,title,uname.strip(),state,stance,otxt,score,json.dumps(analysis)):
                        st.success(f"✅ Submitted! Score: **{score}/100**")
                        if analysis:
                            c1,c2,c3=st.columns(3)
                            c1.metric("Factual",analysis.get("factual_accuracy","N/A"))
                            c2.metric("Bias",analysis.get("emotional_bias","N/A"))
                            c3.metric("Score",f"{score}/100")
                            if analysis.get("overall_assessment"): st.info(f"💡 {analysis['overall_assessment']}")
                        st.rerun()
    if total==0: st.info("No opinions yet. Be the first!"); return
    fp = int((fc/total)*100) if total else 50; ap = 100-fp
    h(f'<div class="sec-hdr">📊 Support vs Opposition</div><div class="split-bar"><div class="split-for" style="width:{fp}%">{fp}% Support</div><div class="split-agn" style="width:{ap}%">{ap}% Oppose</div></div><p style="font-size:0.85rem;color:#718096;margin:8px 0 2px">AI Credibility: <strong style="color:{color}">{label}</strong> ({avg}/100)</p><div class="str-meter"><div class="str-fill" style="width:{fill}%;background:{color}"></div></div>')
    h('<div class="sec-hdr">🤖 AI-Extracted Public Reasons</div>')
    with st.spinner("AI extracting reasons..."): reasons=ai_extract_reasons(title,opinions)
    c1,c2=st.columns(2)
    with c1:
        st.markdown("**✅ Top Reasons FOR**")
        for r in reasons.get("for",[]): h(f'<div class="for-card">✅ {r}</div>')
    with c2:
        st.markdown("**❌ Top Reasons AGAINST**")
        for r in reasons.get("against",[]): h(f'<div class="agn-card">❌ {r}</div>')
    h('<div class="sec-hdr">🏛 Authority Response</div>')
    if replies:
        for rep in replies:
            h(f'<div class="reply-card"><h4>🏛 {rep.get("authority_name","Authority")}</h4><div class="reply-meta">{status_chip(rep.get("status","Acknowledged"))} &nbsp; {rep.get("created_at","")[:10]}</div><div class="reply-txt">{rep.get("reply_text","")}</div></div>')
            up,dn=db_get_reactions(iid)
            c1,c2,c3=st.columns([1,1,4])
            with c1:
                if st.button(f"👍 {up}",key=f"up_{iid}_{rep['id']}"): db_save_reaction(iid,"up"); st.rerun()
            with c2:
                if st.button(f"👎 {dn}",key=f"dn_{iid}_{rep['id']}"): db_save_reaction(iid,"down"); st.rerun()
            rkey=f"rating_{rep['id']}"
            if rkey not in st.session_state:
                if st.button("🤖 Rate Reply Quality",key=f"rate_{rep['id']}"):
                    with st.spinner("Rating..."): st.session_state[rkey]=ai_rate_reply(title,rep.get("reply_text",""))
                    st.rerun()
            else:
                rt=st.session_state[rkey]
                if rt:
                    h(f'<div class="rating-grid"><div class="rating-item"><div class="rating-score">{rt.get("specificity","?")+"/10" if isinstance(rt.get("specificity"),int) else "?"}</div><div class="rating-lbl">Specificity</div></div><div class="rating-item"><div class="rating-score">{str(rt.get("addresses_concerns","?"))+"/10" if isinstance(rt.get("addresses_concerns"),int) else "?"}</div><div class="rating-lbl">Addresses Concerns</div></div><div class="rating-item"><div class="rating-score">{str(rt.get("promises_action","?"))+"/10" if isinstance(rt.get("promises_action"),int) else "?"}</div><div class="rating-lbl">Promises Action</div></div><div class="rating-item"><div class="rating-score">{str(rt.get("overall","?"))+"/10" if isinstance(rt.get("overall"),int) else "?"}</div><div class="rating-lbl">Overall</div></div></div>')
                    if rt.get("assessment"): st.info(f"💡 {rt['assessment']}")
    else: st.info("No authority reply yet.")
    if status=="Resolved" and protest.get("outcome"):
        h(f'<div class="outcome-card"><h4 style="color:#155724;margin:0 0 8px">✅ Protest Outcome</h4><p style="color:#155724;font-size:0.9rem;line-height:1.7;margin:0">{protest["outcome"]}</p></div>')
    h('<div class="sec-hdr">💬 Recent Opinions</div>')
    for o in opinions[:8]:
        score=o.get("credibility_score") or 0
        sc_c="#27ae60" if o.get("stance")=="Support" else "#e74c3c" if o.get("stance")=="Oppose" else "#718096"
        si="✅" if o.get("stance")=="Support" else "❌" if o.get("stance")=="Oppose" else "⚪"
        h(f'<div class="op-card"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px"><div><span style="color:{sc_c};font-weight:700;font-size:0.8rem">{si} {o.get("stance","")}</span> <strong style="font-size:0.85rem">{o.get("user_name","Anonymous")}</strong> <span style="font-size:0.75rem;color:#718096">· {o.get("state","")}</span></div><span class="score-badge">AI Score: {score}</span></div><p style="font-size:0.85rem;color:#4a5568;margin:0;line-height:1.6">{o.get("opinion_text","")[:300]}</p><div style="font-size:0.72rem;color:#a0aec0;margin-top:6px">{o.get("created_at","")[:10]}</div></div>')

def page_replies():
    st.title("🏛 Authority Replies")
    protests,found = db_get_protests(),False
    for p in protests:
        reps = db_get_replies(p["title"])
        if reps:
            found=True; st.markdown(f"### {p['title']}")
            for rep in reps:
                h(f'<div class="reply-card"><h4>🏛 {rep.get("authority_name","Authority")}</h4><div class="reply-meta">{status_chip(rep.get("status","Acknowledged"))} &nbsp; {rep.get("created_at","")[:10]}</div><div class="reply-txt">{rep.get("reply_text","")}</div></div>')
            st.markdown("---")
    if not found: st.info("No authority replies yet.")


def page_report():
    st.title("AI Report Generator")
    protests=db_get_protests(); titles=[p["title"] for p in protests]
    sel=st.selectbox("Select a protest to generate report:",titles)
    protest=next((p for p in protests if p["title"]==sel),None)
    rkey=f"report_{sel}"
    if protest and st.button("Generate AI Civic Report", use_container_width=True):
        with st.spinner("AI is generating a formal civic report..."):
            st.session_state[rkey]=ai_report(sel,db_get_opinions(protest["id"]))
    if st.session_state.get(rkey):
        report=st.session_state[rkey]
        h('<div class="sec-hdr">Generated Report</div>')
        st.markdown(report)
        st.download_button("Download Report (.txt)",report,file_name=f"report_{sel[:30].replace(' ','_')}.txt",use_container_width=True)

def page_authority():
    st.title("🏛 Authority Portal")
    st.info("Secure portal for government authorities.")
    if "auth_in" not in st.session_state: st.session_state["auth_in"]=False
    if not st.session_state["auth_in"]:
        pwd=st.text_input("Password:",type="password")
        if st.button("Login"):
            if pwd==AUTHORITY_PWD: st.session_state["auth_in"]=True; st.rerun()
            else: st.error("Wrong password.")
        return
    st.success("✅ Logged in as Authority")
    if st.button("Logout"): st.session_state["auth_in"]=False; st.rerun()

    # AI-powered analytics dashboard for authorities
    all_ops = db_get_all_opinions()
    protests_list = db_get_protests()
    if all_ops:
        total = len(all_ops)
        avg = int(sum(o.get("credibility_score",0) or 0 for o in all_ops)/total)
        supp = sum(1 for o in all_ops if o.get("stance")=="Support")
        opp  = sum(1 for o in all_ops if o.get("stance")=="Oppose")
        neut = sum(1 for o in all_ops if o.get("stance")=="Neutral")
        high_cred = sum(1 for o in all_ops if (o.get("credibility_score") or 0)>=70)
        top_states = Counter(o.get("state","") for o in all_ops if o.get("state") and o.get("state")!="Not Specified")
        top_state = top_states.most_common(1)[0][0] if top_states else "N/A"
        h('<div class="sec-hdr">🤖 AI Analytics — Citizen Feedback Overview</div>')
        h(f'<div class="stats-row"><div class="stat-box"><div class="stat-val">{total}</div><div class="stat-lbl">Total Opinions</div></div><div class="stat-box"><div class="stat-val" style="color:#27ae60">{supp}</div><div class="stat-lbl">Support</div></div><div class="stat-box"><div class="stat-val" style="color:#e74c3c">{opp}</div><div class="stat-lbl">Oppose</div></div><div class="stat-box"><div class="stat-val">{avg}</div><div class="stat-lbl">Avg AI Score</div></div><div class="stat-box"><div class="stat-val" style="color:#3498db">{high_cred}</div><div class="stat-lbl">High AI Credibility</div></div><div class="stat-box"><div class="stat-val" style="font-size:1rem">{top_state}</div><div class="stat-lbl">Most Active State</div></div></div>')
        fp = int(supp/total*100); ap = int(opp/total*100); np_ = 100-fp-ap
        h(f'<p style="font-size:0.8rem;color:#718096;margin:4px 0 2px">Overall Public Sentiment</p><div class="split-bar"><div class="split-for" style="width:{fp}%">{fp}% Support</div><div class="split-agn" style="width:{ap}%">{ap}% Oppose</div><div style="background:#95a5a6;width:{np_}%;display:flex;align-items:center;padding-left:6px;font-size:0.75rem;color:#fff;font-weight:700">{np_}% Neutral</div></div>')
        # Chart 1: Average AI Credibility Score per Protest — color-coded red→green
        cred_data = []
        for p in protests_list:
            p_ops = [o for o in all_ops if o["issue_id"] == str(p["id"])]
            if p_ops:
                avg_score = int(sum(o.get("credibility_score", 0) or 0 for o in p_ops) / len(p_ops))
                sup_pct = int(sum(1 for o in p_ops if o.get("stance") == "Support") / len(p_ops) * 100)
                cred_data.append((p["title"][:40], avg_score, sup_pct, len(p_ops)))
        cred_data.sort(key=lambda x: x[1])
        c_labels = [x[0] for x in cred_data]
        c_scores = [x[1] for x in cred_data]
        c_sup    = [x[2] for x in cred_data]
        def score_color(s):
            if s >= 70: return "#27ae60"
            elif s >= 55: return "#f39c12"
            else: return "#e74c3c"
        bar_colors = [score_color(s) for s in c_scores]
        fig1 = go.Figure(go.Bar(
            x=c_scores, y=c_labels, orientation="h",
            marker=dict(color=bar_colors, line=dict(color="rgba(255,255,255,0.4)", width=1)),
            text=[f" {s}/100" for s in c_scores], textposition="inside",
            textfont=dict(color="white", size=12, family="Arial Black"),
            customdata=c_sup,
            hovertemplate="<b>%{y}</b><br>Avg Credibility: <b>%{x}/100</b><br>Support: %{customdata}%<extra></extra>"
        ))
        fig1.add_vline(x=70, line_dash="dash", line_color="#27ae60", line_width=1.5,
                       annotation_text="High AI Credibility", annotation_font_size=10, annotation_font_color="#27ae60")
        fig1.add_vline(x=50, line_dash="dot", line_color="#e67e22", line_width=1,
                       annotation_text="Threshold", annotation_font_size=9, annotation_font_color="#e67e22", annotation_y=0.02)
        fig1.update_layout(
            title=dict(text="Average AI Credibility Score per Protest", font=dict(size=15, color="#1a3a5c"), x=0),
            plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
            xaxis=dict(showgrid=True, gridcolor="#e2e8f0", title="Avg Credibility Score (0–100)", color="#64748b", range=[0, 105]),
            yaxis=dict(showgrid=False, color="#374151", automargin=True),
            height=390, margin=dict(l=10, r=20, t=50, b=40),
            font=dict(family="Arial")
        )
        st.plotly_chart(fig1, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            # Chart 2: Stance breakdown per protest — stacked horizontal bar
            supp_counts = [len([o for o in all_ops if o["issue_id"]==str(p["id"]) and o.get("stance")=="Support"]) for p in protests_list]
            opp_counts  = [len([o for o in all_ops if o["issue_id"]==str(p["id"]) and o.get("stance")=="Oppose"]) for p in protests_list]
            neut_counts = [len([o for o in all_ops if o["issue_id"]==str(p["id"]) and o.get("stance")=="Neutral"]) for p in protests_list]
            short_labels = [p["title"][:22]+"…" for p in protests_list]
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name="Support", x=short_labels, y=supp_counts, marker_color="#27ae60",
                hovertemplate="%{x}<br>Support: %{y}<extra></extra>"))
            fig2.add_trace(go.Bar(name="Oppose", x=short_labels, y=opp_counts, marker_color="#e74c3c",
                hovertemplate="%{x}<br>Oppose: %{y}<extra></extra>"))
            fig2.add_trace(go.Bar(name="Neutral", x=short_labels, y=neut_counts, marker_color="#95a5a6",
                hovertemplate="%{x}<br>Neutral: %{y}<extra></extra>"))
            fig2.update_layout(
                barmode="stack",
                title=dict(text="Support vs Opposition per Protest", font=dict(size=13, color="#1a3a5c"), x=0),
                plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
                xaxis=dict(showgrid=False, tickangle=-35, color="#64748b", tickfont=dict(size=9)),
                yaxis=dict(showgrid=True, gridcolor="#e2e8f0", color="#64748b"),
                legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)),
                height=340, margin=dict(l=10, r=10, t=60, b=80),
                font=dict(family="Arial")
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            # Chart 3: AI Credibility donut chart
            buckets = {"Very Low\n(0-20)": 0, "Low\n(21-40)": 0, "Moderate\n(41-60)": 0, "High\n(61-80)": 0, "Very High\n(81-100)": 0}
            for o in all_ops:
                s = o.get("credibility_score") or 0
                if s <= 20: buckets["Very Low\n(0-20)"] += 1
                elif s <= 40: buckets["Low\n(21-40)"] += 1
                elif s <= 60: buckets["Moderate\n(41-60)"] += 1
                elif s <= 80: buckets["High\n(61-80)"] += 1
                else: buckets["Very High\n(81-100)"] += 1
            fig3 = go.Figure(go.Pie(
                labels=list(buckets.keys()), values=list(buckets.values()),
                hole=0.52,
                marker=dict(colors=["#e74c3c","#e67e22","#f1c40f","#2ecc71","#27ae60"],
                            line=dict(color="#ffffff", width=2)),
                textfont=dict(size=11),
                hovertemplate="<b>%{label}</b><br>%{value} opinions (%{percent})<extra></extra>"
            ))
            fig3.add_annotation(text=f"<b>{total}</b><br><span style='font-size:10px'>opinions</span>",
                x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="#1a3a5c"), align="center")
            fig3.update_layout(
                title=dict(text="AI Credibility Score Distribution", font=dict(size=13, color="#1a3a5c"), x=0),
                paper_bgcolor="#ffffff",
                legend=dict(orientation="v", x=1.0, y=0.5, font=dict(size=10)),
                height=340, margin=dict(l=10, r=10, t=50, b=20),
                font=dict(family="Arial")
            )
            st.plotly_chart(fig3, use_container_width=True)
        st.markdown("---")

    h('<div class="sec-hdr">📋 Protest-wise Opinions — Submit Official Replies</div>')
    for p in protests_list:
        ops=db_get_opinions(p["id"])
        if not ops: continue
        with st.expander(f"{p['title']} — {len(ops)} citizen opinions"):
            st.caption(f"Showing top 5 of {len(ops)} opinions")
            for o in ops[:5]: st.markdown(f"- **[{o.get('stance')}]** {o.get('user_name','Anon')} ({o.get('state','')}): {o.get('opinion_text','')[:150]}...")
            st.markdown("---")
            with st.form(f"af_{p['id']}"):
                aname=st.text_input("Ministry/Authority Name"); rtext=st.text_area("Official Reply",height=150)
                rs=st.selectbox("Status",["Acknowledged","Under Review","Action Taken","Resolved"])
                if st.form_submit_button("Submit Reply"):
                    if aname and rtext:
                        if db_save_reply(p["id"],p["title"],aname,rtext,rs): st.success("Reply submitted!"); st.rerun()
                    else: st.error("Name and reply required.")

def main():
    st.set_page_config(page_title="FactProtest India",page_icon="🇮🇳",layout="wide",initial_sidebar_state="expanded")
    inject_css()
    if "page" not in st.session_state: st.session_state["page"]="home"
    with st.sidebar:
        h('<div style="padding:20px 16px 14px;border-bottom:1px solid #1e293b;margin-bottom:8px"><div style="display:flex;align-items:center;gap:10px"><div style="font-size:1.8rem;line-height:1">🇮🇳</div><div><div style="font-size:1.15rem;font-weight:800;color:#f1f5f9;letter-spacing:-0.3px">FactProtest India</div><div style="font-size:0.72rem;color:#94a3b8;margin-top:1px;font-weight:500">AI Protest Credibility Checker</div></div></div></div>')
        nav=st.radio("Nav",["Home","Authority Replies","Reports","Authority Portal"],label_visibility="collapsed")
        h('<div style="margin-top:32px;padding:14px 16px;border-top:1px solid #1e293b;background:#0a1020;border-radius:0 0 8px 8px"><div style="font-size:0.72rem;color:#64748b;font-weight:500">Powered by Groq AI &nbsp;·&nbsp; Supabase<br><span style="color:#64748b">FORGE Buildathon 2026</span></div></div>')
    pages={"Home":"home","Authority Replies":"replies","Reports":"report","Authority Portal":"authority"}
    np=pages[nav]
    if np=="home":
        if st.session_state["page"] not in ("home","detail"): st.session_state["page"]="home"
    else: st.session_state["page"]=np
    page=st.session_state.get("page","home")
    if page=="home": page_home()
    elif page=="detail": page_detail()
    elif page=="replies": page_replies()

    elif page=="report": page_report()
    elif page=="authority": page_authority()

if __name__=="__main__": main()
