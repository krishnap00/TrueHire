from model_utils import predict_job
import streamlit as st
import time

st.set_page_config(page_title="TrueHire", layout="wide")

# ---------------- CSS ---------------- #
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(to right, #0f172a, #1e3a8a);
    color: white;
}

/* HERO */
.hero-title {
    font-size: 6.5rem;
    font-weight: 900;
}

.hero-sub {
    font-size: 2rem;
    margin-top: 20px;
    color: #cbd5f5;
}

/* Section */
.section-title {
    text-align: center;
    font-size: 2.8rem;
    margin-top: 60px;
}

/* Description */
.desc {
    text-align: center;
    font-size: 1.2rem;
    width: 70%;
    margin: auto;
    margin-top: 20px;
    color: #dbeafe;
}

/* Cards */
.card {
    background: white;
    color: black;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    transition: 0.3s;
    margin: 15px;
}

.card:hover {
    transform: translateY(-10px);
}

/* PREMIUM BUTTON (FORCE CENTERED) */
div.stButton {
    display: flex;
    justify-content: center;
}

div.stButton > button {
    font-size: 1.5rem;
    padding: 16px 50px;
    border-radius: 14px;
    background: white;
    color: black;
    border: none;
    font-weight: 600;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.25);
    transition: all 0.2s ease;
    margin-top: 25px; /* Adds space below the title */
}

/* HOVER EFFECT */
div.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0px 12px 25px rgba(0,0,0,0.35);
    background: #f3f4f6;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 60px;
    font-size: 1.2rem;
    color: #cbd5f5;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HERO ---------------- #
col1, col2 = st.columns([1,1])

with col1:
    st.markdown("<div class='hero-title'>TrueHire</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Your Job Partner </div>", unsafe_allow_html=True)

with col2:
    st.image("https://images.unsplash.com/photo-1521737604893-d14cc237f11d", use_container_width=True)

# ---------------- WHY SECTION ---------------- #
st.markdown("<div class='section-title'>Why a Fake Job Detector?</div>", unsafe_allow_html=True)

st.markdown("""
<div class='desc'>
Fake job postings and online recruitment scams have become a major issue in recent years. 
With the rise of job portals and social platforms, scammers exploit job seekers through 
misleading offers and fake opportunities. Detecting fake jobs early helps prevent serious risks.
</div>
""", unsafe_allow_html=True)

# ---------------- PROBLEMS ---------------- #
row1 = st.columns(3)

problems = [
    ("💸 Money Loss", "Scammers may ask for registration or training fees."),
    ("🔐 Data Theft", "Personal data can be stolen and misused."),
    ("⏳ Time Waste", "You may waste time on fake interviews."),
]

for i, col in enumerate(row1):
    with col:
        st.markdown(f"<div class='card'><h3>{problems[i][0]}</h3><p>{problems[i][1]}</p></div>", unsafe_allow_html=True)

# ---------------- FEATURES ---------------- #
st.markdown("<div class='section-title'>Why Choose Us?</div>", unsafe_allow_html=True)

row1 = st.columns(3)
row2 = st.columns(3)

features = [
    ("🧠 AI Analysis", "Smart fraud detection"),
    ("⚡ Fast Results", "Instant verification"),
    ("🔒 Secure", "Safe and private"),
    ("📊 Insights", "Clear analysis"),
    ("🌐 Easy UI", "Simple interface"),
    ("🚀 Reliable", "Accurate predictions")
]

for i, col in enumerate(row1 + row2):
    with col:
        st.markdown(f"<div class='card'><h3>{features[i][0]}</h3><p>{features[i][1]}</p></div>", unsafe_allow_html=True)

# ---------------- CTA ---------------- #
st.markdown("<div class='section-title'>Start Analyzing Jobs Instantly</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    start = st.button("🚀 CLICK HERE TO START")

# STATE
if "show_input" not in st.session_state:
    st.session_state.show_input = False

if start:
    st.session_state.show_input = True

# ---------------- INPUT ---------------- #
if st.session_state.show_input:

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        job_text = st.text_area(
            "",
            placeholder="Enter job details here...",
            height=200
        )

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        analyze = st.button("🔍 START ANALYSIS")

    # ---------------- ANALYSIS ---------------- #
    if analyze:
        if job_text.strip() == "":
            st.error("Please enter job description")
        else:
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.015)
                progress.progress(i + 1)

            # 🔥 CALL YOUR ML MODEL
            real, fake, label = predict_job(job_text)

            # ---------------- RESULT ---------------- #
            st.markdown(f"""
            <div style="margin-top:30px; font-size:1.3rem;">
                <b>Real: {real}%</b> &nbsp;&nbsp;&nbsp;&nbsp;
                <b>Fake: {fake}%</b>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"<h1 style='color:white;'>{label}</h1>", unsafe_allow_html=True)

            # Simple reason logic (optional)
            if fake > 70:
                reason = "High probability of scam patterns detected."
            elif fake > 40:
                reason = "Some suspicious patterns found."
            else:
                reason = "Looks safe based on analysis."

            st.markdown(f"""
            <div style="font-size:1.2rem; margin-top:10px;">
                <b>Reason:</b> {reason}
            </div>
            """, unsafe_allow_html=True)

# ---------------- FOOTER ---------------- #
st.markdown("""
<div class='footer'>
<h3>Trusted by Students & Job Seekers</h3>
✔ Helps avoid scams &nbsp;&nbsp;
✔ Saves time &nbsp;&nbsp;
✔ Increases trust in job search
</div>
""", unsafe_allow_html=True)