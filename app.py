"""
AutoQA — Browser Testing Agent
Streamlit UI — LangGraph + Playwright + Gemini Vision
"""

import os
import base64
import streamlit as st

st.set_page_config(
    page_title="AutoQA — Browser Testing Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main, .stApp { background-color: #0f1117; }
.hero { font-size:2.5rem; font-weight:800;
  background:linear-gradient(135deg,#00c853,#6c63ff);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.sub { color:#888; font-size:1rem; margin-bottom:1.5rem; }
.card { background:#1e2130; border:1px solid #2d3148; border-radius:12px; padding:20px 24px; margin:10px 0; }
.pass-box { background:#1a2e1a; border:2px solid #00c853; border-radius:12px; padding:18px 22px; }
.fail-box { background:#2e1a1a; border:2px solid #ef5350; border-radius:12px; padding:18px 22px; }
.step-pass { color:#00c853; font-size:.9rem; margin:4px 0; }
.step-fail { color:#ef5350; font-size:.9rem; margin:4px 0; }
.step-error { color:#f9a825; font-size:.8rem; margin:2px 0 8px 16px; }
div[data-testid="stSidebar"] { background:#151825; }
.stButton>button { background:linear-gradient(135deg,#00c853,#6c63ff);
  color:#fff; border:none; border-radius:8px; padding:12px 28px; font-weight:700; width:100%; }
.stTextInput input, .stTextArea textarea { background:#1e2130 !important;
  color:#e0e0e0 !important; border:1px solid #2d3148 !important; }
.safety-note { background:#1a1a2e; border:1px solid #f9a82555;
  border-radius:8px; padding:10px 14px; color:#f9a825; font-size:.85rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 AutoQA")
    st.markdown("---")
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    gemini_key = st.text_input(
        "Gemini API Key (optional)", type="password", placeholder="AIza...",
        help="For visual screenshot verification. Get free at ai.google.dev"
    )
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key

    st.markdown("---")
    st.markdown("### 🔗 Pipeline")
    for step in ["🧠 Parse plain English", "📋 Generate test steps", "▶️ Drive real browser", "📸 Screenshot each step", "👁️ Gemini Vision verify", "📝 Generate report"]:
        st.markdown(step)

    st.markdown("---")
    st.markdown("**Stack:** LangGraph · Playwright · Gemini Vision · Groq")

    st.markdown("---")
    st.markdown('<div class="safety-note">🛡️ <b>Safety:</b> Only public URLs allowed. Local/private IPs are blocked. Max 30 actions per test.</div>', unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero">🤖 AutoQA</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Write browser tests in plain English — AI drives a real browser, takes screenshots, and verifies with Gemini Vision</div>', unsafe_allow_html=True)

# ── Examples ───────────────────────────────────────────────────────────────────
examples = [
    ("Wikipedia Search", "https://www.wikipedia.org", "Go to Wikipedia, search for 'Artificial Intelligence', verify the page title contains 'Artificial intelligence' and the article loads"),
    ("GitHub Repo Check", "https://github.com", "Navigate to GitHub, verify the page title contains 'GitHub', check that a Sign in button is visible"),
    ("Hacker News", "https://news.ycombinator.com", "Go to Hacker News, verify the page loads with news items, check the title says 'Hacker News'"),
    ("Python Docs", "https://docs.python.org", "Navigate to Python docs, verify the page loads, check that navigation links are visible"),
    ("DuckDuckGo Search", "https://duckduckgo.com", "Go to DuckDuckGo, search for 'LangGraph', verify results appear on the page"),
]

st.markdown("**💡 Try an example:**")
cols = st.columns(3)
for i, (name, url, desc) in enumerate(examples[:3]):
    with cols[i]:
        if st.button(f"🔗 {name}", key=f"ex{i}"):
            st.session_state["ex_url"] = url
            st.session_state["ex_desc"] = desc

st.markdown("---")

# ── Input ──────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])
with col1:
    test_desc = st.text_area(
        "📝 Describe your test in plain English",
        value=st.session_state.get("ex_desc", ""),
        height=110,
        placeholder="e.g. Go to the homepage, click the login button, verify the login form appears with email and password fields...",
    )
with col2:
    target_url = st.text_input(
        "🌐 Target URL",
        value=st.session_state.get("ex_url", ""),
        placeholder="https://example.com",
    )
    st.markdown('<div class="safety-note">Only public URLs</div>', unsafe_allow_html=True)

# Clear example state after use
for k in ["ex_url", "ex_desc"]:
    if k in st.session_state:
        del st.session_state[k]

run = st.button("🚀 Run Test", use_container_width=True)

# ── Run ────────────────────────────────────────────────────────────────────────
if run:
    if not test_desc.strip():
        st.error("Please describe your test.")
        st.stop()
    if not target_url.strip():
        st.error("Please enter a target URL.")
        st.stop()
    if not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ Add your Groq API key in the sidebar.")
        st.stop()

    from src.tools.browser import is_url_safe
    if not is_url_safe(target_url):
        st.error("🛡️ URL blocked by safety policy. Only public HTTP/HTTPS URLs are allowed.")
        st.stop()

    progress = st.progress(0)
    status_ph = st.empty()

    with st.spinner("🤖 AutoQA is running your test..."):
        try:
            status_ph.info("🧠 Planning test steps from your description...")
            progress.progress(20)

            from src.agents.autoqa_agent import run_test
            state = run_test(test_desc, target_url)

            progress.progress(100)
            status_ph.empty()

            # ── Results ────────────────────────────────────────────────────────
            passed = state.get("passed", False)
            plan = state.get("test_plan", {})
            steps = state.get("steps_executed", [])
            screenshots = state.get("screenshots", [])
            vision_results = state.get("vision_results", [])

            if state.get("status") == "blocked":
                st.error(f"🛡️ {state['fail_reason']}")
                st.stop()

            # Verdict banner
            if passed:
                st.markdown(f"""<div class="pass-box">
                <h2>✅ TEST PASSED</h2>
                <p>{plan.get('test_name', 'Browser Test')}</p>
                <p style="color:#888">{sum(1 for s in steps if s.get('success'))}/{len(steps)} steps succeeded</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="fail-box">
                <h2>❌ TEST FAILED</h2>
                <p>{plan.get('test_name', 'Browser Test')}</p>
                <p style="color:#888">Reason: {state.get('fail_reason', 'Unknown')}</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")

            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Steps Passed", f"{sum(1 for s in steps if s.get('success'))}/{len(steps)}")
            c2.metric("Screenshots", len(screenshots))
            c3.metric("Vision Checks", len(vision_results))
            c4.metric("Result", "✅ PASS" if passed else "❌ FAIL")

            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Test Plan", "🔢 Steps", "📸 Screenshots", "📝 Report"])

            with tab1:
                st.markdown(f"### {plan.get('test_name', 'Test')}")
                st.markdown(f"**Objective:** {plan.get('objective', '')}")
                st.markdown(f"**Success Criteria:** {plan.get('success_criteria', '')}")
                st.markdown(f"**Steps planned:** {len(plan.get('steps', []))}")
                st.markdown("#### Planned Steps")
                for s in plan.get("steps", []):
                    st.markdown(f"**{s['step_number']}.** {s.get('description', '')} `{s['action']}` → `{s.get('target','')}`")

            with tab2:
                st.markdown("### Step Results")
                for s in steps:
                    icon = "✅" if s.get("success") else "❌"
                    cls = "step-pass" if s.get("success") else "step-fail"
                    st.markdown(f'<div class="{cls}">{icon} Step {s.get("step_number","?")}: {s.get("step_description", s.get("action",""))}</div>', unsafe_allow_html=True)
                    if not s.get("success") and s.get("error"):
                        st.markdown(f'<div class="step-error">↳ {s["error"]}</div>', unsafe_allow_html=True)

                if vision_results:
                    st.markdown("### 👁️ Vision Verification")
                    for v in vision_results:
                        icon = "✅" if v.get("verified") else "❌" if v.get("verified") is False else "⚠️"
                        with st.expander(f"{icon} Step {v.get('step_number')}: {v.get('step_description','')}"):
                            st.markdown(f"**What Gemini sees:** {v.get('description','')}")
                            st.markdown(f"**Match:** {v.get('matches','')}")
                            st.markdown(f"**Confidence:** {v.get('confidence','N/A')}")
                            if not v.get("verified") and v.get("suggestion") != "N/A":
                                st.warning(f"💡 {v.get('suggestion','')}")

            with tab3:
                st.markdown("### Screenshots")
                if screenshots:
                    cols_ss = st.columns(2)
                    for i, ss in enumerate(screenshots):
                        with cols_ss[i % 2]:
                            if ss.get("path"):
                                try:
                                    st.image(ss["path"], caption=ss.get("label", f"Step {i+1}"), use_column_width=True)
                                except Exception:
                                    st.caption(f"Screenshot {i+1}: {ss.get('label','')}")
                else:
                    st.info("No screenshots available")

            with tab4:
                report = state.get("report", "No report generated")
                st.markdown(report)
                st.download_button(
                    "⬇️ Download Report",
                    data=report,
                    file_name=f"autoqa_report_{plan.get('test_name','test').replace(' ','_')}.md",
                    mime="text/markdown",
                )

            if state.get("errors"):
                with st.expander("⚠️ Errors"):
                    for e in state["errors"]:
                        st.warning(e)

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)

else:
    # Landing
    st.markdown("### ⚡ How it works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="card"><b>📝 Write in Plain English</b><br>
        <span style="color:#888">"Go to the homepage, click Sign In, verify the login form appears" — no code needed</span></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="card"><b>📸 Every Step Captured</b><br>
        <span style="color:#888">Screenshots taken at each action — full visual audit trail of what happened</span></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card"><b>🤖 AI Plans the Steps</b><br>
        <span style="color:#888">Llama 3 converts your description into precise Playwright browser actions</span></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="card"><b>👁️ Gemini Verifies</b><br>
        <span style="color:#888">Gemini Vision looks at screenshots and confirms what's actually on screen</span></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="card"><b>🌐 Real Browser</b><br>
        <span style="color:#888">Playwright drives a real Chromium browser — not a simulation</span></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="card"><b>🛡️ Safety First</b><br>
        <span style="color:#888">Only public URLs allowed. Local IPs blocked. Max 30 actions per test.</span></div>""", unsafe_allow_html=True)
