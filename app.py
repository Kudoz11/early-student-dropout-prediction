import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────
# PAGE CONFIG — sidebar hidden permanently
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EduGuard AI — Dropout Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if 'analyzed' not in st.session_state:
    st.session_state['analyzed'] = False

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #080c14; color: #e2e8f0; }

/* Hide sidebar and its toggle arrow completely */
section[data-testid="stSidebar"] { display: none !important; }
button[data-testid="collapsedControl"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem; max-width: 100%; }

/* Top bar */
.topbar {
    background: linear-gradient(135deg, #0d1220 0%, #111827 100%);
    border-bottom: 1px solid #1e3a5f;
    padding: 16px 2rem;
    margin: -1rem -2rem 1.5rem -2rem;
    display: flex; align-items: center; gap: 16px;
}
.topbar-logo {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #2563eb, #0ea5e9);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
}
.topbar-title {
    font-family: 'Syne', sans-serif; font-size: 21px; font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.topbar-sub { font-size: 11px; color: #475569; margin-top: 1px; }
.badge {
    margin-left: auto;
    background: #0f2044; border: 1px solid #1e3a5f;
    color: #60a5fa; font-size: 11px; padding: 4px 12px;
    border-radius: 20px;
}

/* Form panel */
.form-panel {
    background: #0d1220;
    border: 1px solid #1a2540;
    border-radius: 16px;
    padding: 20px 18px;
}
.form-section-title {
    font-family: 'Syne', sans-serif;
    font-size: 10px; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase;
    color: #3b82f6;
    margin: 18px 0 8px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #1a2540;
}
.form-section-title:first-child { margin-top: 0; }

/* Analyze button */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #0284c7) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 14px !important;
    padding: 14px 0 !important; width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Section label */
.section-label {
    font-family: 'Syne', sans-serif; font-size: 10px; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase;
    color: #3b82f6; margin-bottom: 6px; margin-top: 18px;
}

/* Metric cards */
.metric-row { display: flex; gap: 12px; margin: 16px 0 8px 0; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 110px;
    background: #0d1220; border: 1px solid #1a2540;
    border-radius: 14px; padding: 16px 18px;
    position: relative; overflow: hidden;
}
.metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
}
.metric-card.blue::before  { background: linear-gradient(90deg,#2563eb,#0ea5e9); }
.metric-card.green::before { background: linear-gradient(90deg,#10b981,#34d399); }
.metric-card.red::before   { background: linear-gradient(90deg,#ef4444,#f97316); }
.metric-card.gold::before  { background: linear-gradient(90deg,#f59e0b,#fbbf24); }
.metric-label { font-size:10px; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; color:#475569; margin-bottom:6px; }
.metric-value { font-family:'Syne',sans-serif; font-size:26px; font-weight:800; line-height:1; }
.metric-value.blue  { color:#60a5fa; }
.metric-value.green { color:#34d399; }
.metric-value.red   { color:#f87171; }
.metric-value.gold  { color:#fbbf24; }
.metric-sub { font-size:11px; color:#334155; margin-top:4px; }

/* Result banner */
.result-banner {
    border-radius: 14px; padding: 20px 24px; margin: 0 0 16px 0;
    display: flex; align-items: center; gap: 14px;
}
.result-banner.safe   { background:linear-gradient(135deg,#022c22,#064e3b); border:1px solid #065f46; }
.result-banner.risk   { background:linear-gradient(135deg,#3b0a0a,#7f1d1d); border:1px solid #991b1b; }
.result-banner.medium { background:linear-gradient(135deg,#1c1003,#451a03); border:1px solid #78350f; }
.result-icon { font-size:32px; }
.result-title { font-family:'Syne',sans-serif; font-size:20px; font-weight:800; }
.result-desc  { font-size:12px; color:#94a3b8; margin-top:3px; }
.result-banner.safe   .result-title { color:#6ee7b7; }
.result-banner.risk   .result-title { color:#fca5a5; }
.result-banner.medium .result-title { color:#fcd34d; }

/* Rec cards */
.rec-card {
    border-radius: 10px; padding: 10px 14px; margin: 6px 0;
    display: flex; align-items: center; gap: 10px; font-size: 12px;
}
.rec-card.danger { background:#1a0808; border:1px solid #7f1d1d; color:#fca5a5; }
.rec-card.warn   { background:#1a1108; border:1px solid #78350f; color:#fcd34d; }
.rec-card.ok     { background:#071a12; border:1px solid #064e3b; color:#6ee7b7; }

.divider { height:1px; background:linear-gradient(90deg,transparent,#1e3a5f,transparent); margin:20px 0; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background:#0d1220; border-radius:10px;
    padding:4px; border:1px solid #1a2540; gap:4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius:8px; color:#64748b; font-weight:500; font-size:13px; padding:8px 20px;
}
.stTabs [aria-selected="true"] { background:#1e3a5f !important; color:#60a5fa !important; }
.stSelectbox > div > div { background:#0d1220 !important; border-color:#1a2540 !important; }

/* Info box */
.info-box {
    background:#0d1220; border:1px solid #1a2540;
    border-radius:12px; padding:18px 20px; margin:10px 0;
}
.info-box h4 {
    font-family:'Syne',sans-serif; font-size:12px; font-weight:700;
    color:#60a5fa; margin:0 0 8px 0;
    text-transform:uppercase; letter-spacing:1px;
}
.info-box p { font-size:12px; color:#64748b; margin:0; line-height:1.7; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('dropout_model.pkl')

model = load_model()

FEATURES = [
    'Marital status', 'Application mode', 'Application order',
    'Course', 'Daytime/evening attendance', 'Previous qualification',
    'Nacionality', "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", 'Displaced',
    'Educational special needs', 'Debtor', 'Tuition fees up to date',
    'Gender', 'Scholarship holder', 'Age at enrollment', 'International',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)'
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def gauge(prob):
    pct = prob * 100
    if pct > 65:   needle_color, zone = "#ef4444", "HIGH RISK"
    elif pct > 40: needle_color, zone = "#f59e0b", "MEDIUM RISK"
    else:          needle_color, zone = "#10b981", "LOW RISK"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct,
        number={'suffix':"%",'font':{'size':36,'color':needle_color,'family':'Syne'}},
        gauge={
            'axis':{'range':[0,100],'tickcolor':'#1e3a5f','tickfont':{'color':'#334155','size':9},'tickwidth':1},
            'bar':{'color':needle_color,'thickness':0.22},
            'bgcolor':'rgba(0,0,0,0)','borderwidth':0,
            'steps':[
                {'range':[0,40],'color':'#071a12'},
                {'range':[40,65],'color':'#1c1003'},
                {'range':[65,100],'color':'#1a0808'},
            ],
            'threshold':{'line':{'color':'#ffffff','width':2},'thickness':0.82,'value':65}
        }
    ))
    fig.add_annotation(text=zone, x=0.5, y=0.18,
        font={'size':10,'color':'#475569','family':'Syne'}, showarrow=False)
    fig.update_layout(height=220, margin=dict(t=16,b=0,l=16,r=16),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color':'#94a3b8'})
    return fig

def mini_bar(label, value, max_val, color):
    fig = go.Figure(go.Bar(
        x=[value], y=[label], orientation='h',
        marker_color=color, marker_line_width=0,
    ))
    fig.update_layout(height=48, margin=dict(t=0,b=0,l=0,r=8),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0,max_val], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False), bargap=0.3)
    return fig

# ─────────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-logo">🎓</div>
  <div>
    <div class="topbar-title">EduGuard AI</div>
    <div class="topbar-sub">Student Dropout Early Prediction System</div>
  </div>
  <div class="badge">🔬 3-Stage Framework · AUC 0.9424 · Stage B</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "👤  Single Student Analysis",
    "📂  Bulk CSV Upload",
    "📊  Model Overview"
])

# ══════════════════════════════════════════════
# TAB 1 — form LEFT, results RIGHT (no sidebar!)
# ══════════════════════════════════════════════
with tab1:
    form_col, result_col = st.columns([1, 1.7], gap="large")

    # ── LEFT: INPUT FORM ──
    with form_col:
        st.markdown('<div class="form-panel">', unsafe_allow_html=True)
        st.markdown('<div class="form-section-title">🎓 Academic — 1st Semester</div>', unsafe_allow_html=True)
        cu_approved    = st.slider("Units Approved",       0, 26,  5)
        cu_grade       = st.slider("Grade (0–20)",         0.0, 20.0, 12.0, step=0.1)
        cu_enrolled    = st.slider("Units Enrolled",       0, 26,  6)
        cu_evaluations = st.slider("Units Evaluations",    0, 45,  8)
        cu_credited    = st.slider("Units Credited",       0, 20,  0)
        cu_no_eval     = st.slider("Without Evaluations",  0, 12,  0)

        st.markdown('<div class="form-section-title">💰 Financial</div>', unsafe_allow_html=True)
        tuition     = st.selectbox("Tuition Fees Up to Date?", [1, 0],
                                    format_func=lambda x: "✅ Yes" if x==1 else "❌ No")
        debtor      = st.selectbox("Is Debtor?", [0, 1],
                                    format_func=lambda x: "Yes" if x==1 else "No")
        scholarship = st.selectbox("Scholarship Holder?", [0, 1],
                                    format_func=lambda x: "Yes" if x==1 else "No")

        st.markdown('<div class="form-section-title">👤 Personal</div>', unsafe_allow_html=True)
        age    = st.number_input("Age at Enrollment", min_value=17, max_value=70, value=20)
        gender = st.radio("Gender", [1, 0],
                           format_func=lambda x: "Male" if x==1 else "Female",
                           horizontal=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍  Analyze Student Risk", use_container_width=True)

        if analyze_btn:
            st.session_state['analyzed'] = True
            st.session_state['vals'] = {
                'cu_approved': cu_approved, 'cu_grade': cu_grade,
                'cu_enrolled': cu_enrolled, 'cu_evaluations': cu_evaluations,
                'cu_credited': cu_credited, 'cu_no_eval': cu_no_eval,
                'tuition': tuition, 'debtor': debtor,
                'scholarship': scholarship, 'age': age, 'gender': gender
            }

    # ── RIGHT: RESULTS ──
    with result_col:
        if not st.session_state['analyzed']:
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;
                        justify-content:center;height:480px;text-align:center;">
              <div style="font-size:80px;opacity:0.12;margin-bottom:24px">🎓</div>
              <div style="font-family:Syne;font-size:22px;font-weight:700;color:#1e3a5f">
                Fill the form and click Analyze
              </div>
              <div style="font-size:13px;color:#0f2044;margin-top:12px;
                          max-width:300px;line-height:1.7;">
                Enter the student details in the left panel,<br>
                then press <b style="color:#1e3a5f">Analyze Student Risk</b>
              </div>
              <div style="margin-top:32px;display:flex;gap:16px;flex-wrap:wrap;justify-content:center;">
                <div style="background:#071a12;border:1px solid #064e3b;border-radius:10px;
                            padding:12px 20px;font-size:12px;color:#34d399;">
                  🟢 &lt;40% = Low Risk
                </div>
                <div style="background:#1c1003;border:1px solid #78350f;border-radius:10px;
                            padding:12px 20px;font-size:12px;color:#fcd34d;">
                  🟡 40–65% = Medium Risk
                </div>
                <div style="background:#1a0808;border:1px solid #7f1d1d;border-radius:10px;
                            padding:12px 20px;font-size:12px;color:#fca5a5;">
                  🔴 &gt;65% = High Risk
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            v = st.session_state.get('vals', {
                'cu_approved':5,'cu_grade':12.0,'cu_enrolled':6,
                'cu_evaluations':8,'cu_credited':0,'cu_no_eval':0,
                'tuition':1,'debtor':0,'scholarship':0,'age':20,'gender':1
            })

            inp = {f: 0 for f in FEATURES}
            inp['Curricular units 1st sem (approved)']            = v['cu_approved']
            inp['Curricular units 1st sem (grade)']               = v['cu_grade']
            inp['Curricular units 1st sem (enrolled)']            = v['cu_enrolled']
            inp['Curricular units 1st sem (evaluations)']         = v['cu_evaluations']
            inp['Curricular units 1st sem (credited)']            = v['cu_credited']
            inp['Curricular units 1st sem (without evaluations)'] = v['cu_no_eval']
            inp['Tuition fees up to date']                        = v['tuition']
            inp['Debtor']                                         = v['debtor']
            inp['Scholarship holder']                             = v['scholarship']
            inp['Age at enrollment']                              = v['age']
            inp['Gender']                                         = v['gender']

            prob = model.predict_proba(pd.DataFrame([inp]))[0][1]
            pct  = prob * 100

            if pct < 40:
                banner_cls, icon, title, desc = (
                    "safe","✅","Low Dropout Risk",
                    "Student shows strong indicators for graduation."
                )
            elif pct < 65:
                banner_cls, icon, title, desc = (
                    "medium","⚠️","Moderate Risk — Monitor Closely",
                    "Some risk factors detected. Proactive support recommended."
                )
            else:
                banner_cls, icon, title, desc = (
                    "risk","🚨","High Dropout Risk — Urgent Action Needed",
                    "Multiple risk factors identified. Immediate intervention required."
                )

            st.markdown(f"""
            <div class="result-banner {banner_cls}">
              <div class="result-icon">{icon}</div>
              <div>
                <div class="result-title">{title}</div>
                <div class="result-desc">{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            col_g, col_m = st.columns([1, 1.1])
            with col_g:
                st.plotly_chart(gauge(prob), use_container_width=True)
            with col_m:
                st.markdown(f"""
                <div class="metric-row">
                  <div class="metric-card red">
                    <div class="metric-label">Dropout Risk</div>
                    <div class="metric-value red">{pct:.1f}%</div>
                    <div class="metric-sub">Model confidence</div>
                  </div>
                  <div class="metric-card green">
                    <div class="metric-label">Graduate Prob</div>
                    <div class="metric-value green">{100-pct:.1f}%</div>
                    <div class="metric-sub">Likely outcome</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="section-label">Key Indicators</div>', unsafe_allow_html=True)
                for lbl, val, mx, clr in [
                    ("Units Approved", v['cu_approved'], 26, "#3b82f6"),
                    ("1st Sem Grade",  v['cu_grade'],    20, "#10b981"),
                    ("Units Enrolled", v['cu_enrolled'], 26, "#8b5cf6"),
                ]:
                    c1, c2 = st.columns([1.8, 1])
                    with c1:
                        st.plotly_chart(mini_bar(lbl, val, mx, clr), use_container_width=True)
                    with c2:
                        st.markdown(
                            f"<div style='font-size:17px;font-weight:700;"
                            f"color:{clr};font-family:Syne;padding-top:10px'>{val}</div>",
                            unsafe_allow_html=True
                        )

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Strategic Recommendations</div>', unsafe_allow_html=True)

            recs = []
            if v['cu_approved'] < 3:
                recs.append(("danger","🔴","Critical: Very low approved units — immediate academic counseling required"))
            if v['cu_grade'] < 10:
                recs.append(("danger","🔴","Low grade average — connect student with tutoring program"))
            if v['debtor'] == 1:
                recs.append(("danger","🔴","Student is a debtor — refer to financial aid office"))
            if v['tuition'] == 0:
                recs.append(("warn","🟡","Tuition not up to date — verify fee status and offer payment plan"))
            if v['scholarship'] == 0 and pct > 40:
                recs.append(("warn","🟡","No scholarship — explore financial support opportunities"))
            if v['cu_enrolled'] > v['cu_approved'] + 5:
                recs.append(("warn","🟡","Low unit completion rate — review student's course load"))
            if not recs:
                recs.append(("ok","🟢","Student is performing well — maintain regular check-ins"))

            rc1, rc2 = st.columns(2)
            for i, (cls, ico, msg) in enumerate(recs):
                with (rc1 if i % 2 == 0 else rc2):
                    st.markdown(f'<div class="rec-card {cls}">{ico}&nbsp;&nbsp;{msg}</div>',
                                unsafe_allow_html=True)

            # ══════════════════════════════════════════════
            # WHY EXPLANATION — SHAP
            # ══════════════════════════════════════════════
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-label">🧠 Why This Prediction?</div>',
                        unsafe_allow_html=True)

            try:
                import shap

                @st.cache_resource
                def load_explainer():
                    # shap_explainer.pkl available ho toh use karo
                    # warna model se hi TreeExplainer banao
                    try:
                        return joblib.load('shap_explainer.pkl')
                    except Exception:
                        return shap.TreeExplainer(model)

                explainer_shap = load_explainer()
                X_inp     = pd.DataFrame([inp])

                # SHAP 0.40+ new API — returns Explanation object
                shap_exp = explainer_shap(X_inp)
                raw      = shap_exp.values  # (1, n_features) or (1, n_features, n_classes)

                if raw.ndim == 3:
                    sv = raw[0, :, 1]   # multiclass → class 1 (Dropout)
                elif raw.ndim == 2:
                    sv = raw[0, :]      # binary XGBoost
                else:
                    sv = raw[0]

                # Top 8 features by absolute SHAP impact
                indices      = np.argsort(np.abs(sv))[::-1][:8]
                top_features = [FEATURES[i] for i in indices]
                top_values   = [sv[i] for i in indices]
                top_actual   = [X_inp.iloc[0][FEATURES[i]] for i in indices]

                short_names = {
                    'Curricular units 1st sem (approved)':            '1st Sem Units Approved',
                    'Curricular units 1st sem (grade)':               '1st Sem Grade',
                    'Curricular units 1st sem (enrolled)':            'Units Enrolled',
                    'Curricular units 1st sem (evaluations)':         'Units Evaluations',
                    'Curricular units 1st sem (credited)':            'Units Credited',
                    'Curricular units 1st sem (without evaluations)': 'Without Evaluations',
                    'Tuition fees up to date':                        'Tuition Fees Paid',
                    'Debtor':                                         'Is Debtor',
                    'Scholarship holder':                             'Scholarship Holder',
                    'Age at enrollment':                              'Age at Enrollment',
                    'Gender':                                         'Gender',
                }
                display_names = [short_names.get(f, f) for f in top_features]
                colors        = ['#ef4444' if val > 0 else '#10b981' for val in top_values]

                # ── SHAP Bar Chart ──
                fig_why = go.Figure()
                fig_why.add_trace(go.Bar(
                    x=top_values[::-1],
                    y=display_names[::-1],
                    orientation='h',
                    marker_color=colors[::-1],
                    marker_line_width=0,
                    text=[
                        f"+{val:.3f} ↑ Risk" if val > 0 else f"{val:.3f} ↓ Safe"
                        for val in top_values[::-1]
                    ],
                    textposition='outside',
                    textfont=dict(color='#64748b', size=11),
                ))
                fig_why.add_vline(x=0, line_color='#334155', line_width=1.5)
                fig_why.update_layout(
                    height=320,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(
                        title="← Reduces Dropout Risk    |    Increases Dropout Risk →",
                        title_font=dict(color='#475569', size=11),
                        color='#334155', gridcolor='#0f1f35', zeroline=False,
                    ),
                    yaxis=dict(color='#94a3b8', gridcolor='rgba(0,0,0,0)'),
                    margin=dict(t=10, b=50, l=10, r=110),
                    font_color='#64748b',
                )
                st.plotly_chart(fig_why, use_container_width=True)

                # ── Plain Language Cards ──
                st.markdown('<div class="section-label">Plain Language Explanation</div>',
                            unsafe_allow_html=True)
                exp_col1, exp_col2 = st.columns(2)
                shown = 0
                for feat, val, actual in zip(top_features, top_values, top_actual):
                    if abs(val) < 0.005 or shown >= 6:
                        continue
                    color = "#fca5a5" if val > 0 else "#6ee7b7"
                    arrow = "⬆️ increases dropout risk" if val > 0 else "⬇️ reduces dropout risk"

                    if feat == 'Curricular units 1st sem (approved)':
                        msg = f"Only <b>{int(actual)}</b> units approved — {arrow}"
                    elif feat == 'Curricular units 1st sem (grade)':
                        msg = f"Grade <b>{actual:.1f}/20</b> — {arrow}"
                    elif feat == 'Curricular units 1st sem (enrolled)':
                        msg = f"<b>{int(actual)}</b> units enrolled — {arrow}"
                    elif feat == 'Tuition fees up to date':
                        s = "Paid ✅" if actual == 1 else "Not Paid ❌"
                        msg = f"Tuition: <b>{s}</b> — {arrow}"
                    elif feat == 'Debtor':
                        s = "Yes ❌" if actual == 1 else "No ✅"
                        msg = f"Debtor status: <b>{s}</b> — {arrow}"
                    elif feat == 'Scholarship holder':
                        s = "Yes ✅" if actual == 1 else "No"
                        msg = f"Scholarship: <b>{s}</b> — {arrow}"
                    elif feat == 'Age at enrollment':
                        msg = f"Enrolled at age <b>{int(actual)}</b> — {arrow}"
                    elif feat == 'Gender':
                        s = "Male" if actual == 1 else "Female"
                        msg = f"Gender: <b>{s}</b> — {arrow}"
                    else:
                        msg = f"<b>{short_names.get(feat, feat)}</b>: {actual} — {arrow}"

                    col = exp_col1 if shown % 2 == 0 else exp_col2
                    with col:
                        st.markdown(f"""
                        <div style="background:#0d1220;border:1px solid #1a2540;
                                    border-left:3px solid {color};border-radius:8px;
                                    padding:10px 14px;margin:5px 0;
                                    font-size:12px;color:#94a3b8;line-height:1.6;">
                            {msg}
                        </div>
                        """, unsafe_allow_html=True)
                    shown += 1

            except Exception as e:
                st.markdown(f"""
                <div style="background:#1a0808;border:1px solid #7f1d1d;border-radius:10px;
                            padding:14px 18px;font-size:12px;color:#fca5a5;">
                    ⚠️ SHAP explanation unavailable: {str(e)}<br>
                    <span style="color:#475569;margin-top:4px;display:block;">
                    Install SHAP: <code>pip install shap</code> and ensure
                    <code>shap_explainer.pkl</code> or model is accessible.
                    </span>
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — BULK UPLOAD
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-label">Bulk Student Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#475569;font-size:13px;margin-bottom:20px;">
    Upload a CSV file containing student records. Columns must match the 25 Stage-B features.
    </p>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df_up = pd.read_csv(uploaded)
        missing = [f for f in FEATURES if f not in df_up.columns]

        if missing:
            st.error(f"❌ Missing columns: {missing[:5]}{'...' if len(missing)>5 else ''}")
        else:
            with st.spinner("Analyzing students..."):
                probs = model.predict_proba(df_up[FEATURES])[:, 1]

            df_up['Dropout Risk %'] = (probs * 100).round(1)
            df_up['Risk Level'] = pd.cut(
                probs, bins=[0, 0.40, 0.65, 1.0],
                labels=['🟢 Low', '🟡 Medium', '🔴 High'],
                include_lowest=True
            )
            df_up['Prediction'] = [
                '🚨 At Risk' if p > 0.65 else
                '⚠️ Monitor'  if p > 0.40 else
                '✅ Graduate'
                for p in probs
            ]

            n_high   = int((probs > 0.65).sum())
            n_medium = int(((probs > 0.40) & (probs <= 0.65)).sum())
            n_low    = int((probs <= 0.40).sum())
            total    = len(df_up)

            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-card blue">
                <div class="metric-label">Total Students</div>
                <div class="metric-value blue">{total}</div>
                <div class="metric-sub">Records analyzed</div>
              </div>
              <div class="metric-card red">
                <div class="metric-label">High Risk 🔴</div>
                <div class="metric-value red">{n_high}</div>
                <div class="metric-sub">Immediate action needed</div>
              </div>
              <div class="metric-card gold">
                <div class="metric-label">Medium Risk 🟡</div>
                <div class="metric-value gold">{n_medium}</div>
                <div class="metric-sub">Monitor closely</div>
              </div>
              <div class="metric-card green">
                <div class="metric-label">Low Risk 🟢</div>
                <div class="metric-value green">{n_low}</div>
                <div class="metric-sub">On track to graduate</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            ch1, ch2 = st.columns(2)
            with ch1:
                st.markdown('<div class="section-label">Risk Score Distribution</div>', unsafe_allow_html=True)
                fig_hist = px.histogram(df_up, x='Dropout Risk %', nbins=20,
                                         color_discrete_sequence=['#3b82f6'])
                fig_hist.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#94a3b8', height=280,
                    margin=dict(t=10,b=10,l=0,r=0),
                    xaxis=dict(showgrid=False, color='#334155'),
                    yaxis=dict(showgrid=False, color='#334155'),
                    bargap=0.1,
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with ch2:
                st.markdown('<div class="section-label">Risk Category Breakdown</div>', unsafe_allow_html=True)
                fig_pie = go.Figure(go.Pie(
                    labels=['🟢 Low', '🟡 Medium', '🔴 High'],
                    values=[n_low, n_medium, n_high],
                    hole=0.55,
                    marker_colors=['#10b981','#f59e0b','#ef4444'],
                    textfont_color='#94a3b8',
                ))
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8', height=280,
                    margin=dict(t=10,b=10,l=0,r=0),
                    legend=dict(font_color='#64748b', bgcolor='rgba(0,0,0,0)'),
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Student Results (sorted by risk)</div>', unsafe_allow_html=True)

            show_cols = ['Dropout Risk %', 'Risk Level', 'Prediction',
                         'Curricular units 1st sem (approved)',
                         'Curricular units 1st sem (grade)',
                         'Tuition fees up to date', 'Debtor', 'Age at enrollment']
            show_cols = [c for c in show_cols if c in df_up.columns]

            st.dataframe(
                df_up[show_cols].sort_values('Dropout Risk %', ascending=False).reset_index(drop=True),
                use_container_width=True, height=400
            )

            csv_out = df_up.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️  Download Full Results CSV", csv_out,
                               "eduguard_results.csv", "text/csv")

# ══════════════════════════════════════════════
# TAB 3 — MODEL OVERVIEW
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-label">Model Performance</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-row">
      <div class="metric-card green">
        <div class="metric-label">AUC-ROC Score</div>
        <div class="metric-value green">0.9424</div>
        <div class="metric-sub">Stage B — best stage</div>
      </div>
      <div class="metric-card blue">
        <div class="metric-label">Features Used</div>
        <div class="metric-value blue">25</div>
        <div class="metric-sub">Stage B feature set</div>
      </div>
      <div class="metric-card gold">
        <div class="metric-label">Framework</div>
        <div class="metric-value gold" style="font-size:18px;padding-top:4px">3-Stage</div>
        <div class="metric-sub">Early → Mid → Late</div>
      </div>
      <div class="metric-card red">
        <div class="metric-label">Risk Thresholds</div>
        <div class="metric-value red" style="font-size:18px;padding-top:4px">40 / 65</div>
        <div class="metric-sub">Low · Medium · High</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-label">Feature Importance (Approximate)</div>', unsafe_allow_html=True)
        feat_imp = {
            'Units Approved (1st Sem)': 0.28,
            'Tuition Fees Up to Date':  0.18,
            'Grade (1st Sem)':          0.15,
            'Units Enrolled':           0.10,
            'Debtor Status':            0.09,
            'Age at Enrollment':        0.07,
            'Scholarship Holder':       0.06,
            'Gender':                   0.04,
            'Units Credited':           0.03,
        }
        fig_imp = go.Figure(go.Bar(
            x=list(feat_imp.values()), y=list(feat_imp.keys()),
            orientation='h',
            marker=dict(color=list(feat_imp.values()),
                        colorscale=[[0,'#1e3a5f'],[1,'#3b82f6']], showscale=False),
            text=[f"{v*100:.0f}%" for v in feat_imp.values()],
            textposition='outside', textfont_color='#64748b',
        ))
        fig_imp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8', height=340,
            margin=dict(l=0,r=50,t=10,b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, tickfont=dict(size=11, color='#64748b')),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-label">3-Stage AUC Comparison</div>', unsafe_allow_html=True)
        fig_stages = go.Figure(go.Bar(
            x=['Stage A\n(Early)', 'Stage B\n(Mid)', 'Stage C\n(Late)'],
            y=[0.88, 0.9424, 0.91],
            marker_color=['#334155', '#3b82f6', '#334155'],
            text=['0.8800', '0.9424 ★', '0.9100'],
            textposition='outside', textfont_color='#64748b',
        ))
        fig_stages.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8', height=340,
            margin=dict(l=0,r=0,t=10,b=10),
            xaxis=dict(showgrid=False, tickfont=dict(size=12, color='#64748b')),
            yaxis=dict(showgrid=False, range=[0.80, 0.97],
                       tickfont=dict(size=10, color='#334155'), title='AUC-ROC'),
        )
        st.plotly_chart(fig_stages, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">About This Model</div>', unsafe_allow_html=True)

    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown("""
        <div class="info-box">
          <h4>🤖 Algorithm</h4>
          <p>XGBoost Classifier trained on a Portuguese higher-education dataset.
          Gradient boosting with early stopping for optimal generalization.</p>
        </div>
        """, unsafe_allow_html=True)
    with i2:
        st.markdown("""
        <div class="info-box">
          <h4>📊 Training Data</h4>
          <p>4,424 students across multiple degree programs. Balanced using SMOTE.
          Stratified 80/20 train-test split with 5-fold cross-validation.</p>
        </div>
        """, unsafe_allow_html=True)
    with i3:
        st.markdown("""
        <div class="info-box">
          <h4>🎯 Stage B Focus</h4>
          <p>Stage B uses 1st semester academic + financial data,
          achieving AUC 0.9424 — highest of all three stages.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;padding:28px 0 8px 0;color:#1e3a5f;font-size:11px;letter-spacing:0.5px;">
      EduGuard AI · 3-Stage Dropout Prediction Framework · <span style="color:#334155">Stage B Active</span>
    </div>
    """, unsafe_allow_html=True)
