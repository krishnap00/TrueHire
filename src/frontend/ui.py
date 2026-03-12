import streamlit as st
import time

# Page Configuration
st.set_page_config(page_title="TrueHire | Fake Job Detector", page_icon="🛡️")

# Custom CSS
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #e8f5e9;
}

/* Title */
.main-title {
    color: #1b5e20;
    text-align: center;
    font-size: 3rem;
    font-weight: 700;
    margin-top: 60px;
}

/* Subtitle */
.sub-title {
    color: #43a047;
    text-align: center;
    font-size: 1.3rem;
    margin-bottom: 50px;
}

/* Main Button */
div[data-testid="stColumn"] button {
    display: block;
    margin: auto;
    background-color: #2e7d32;
    color: white !important;
    border-radius: 30px;
    padding: 16px 45px;
    font-size: 1.2rem;
    font-weight: 600;
    border: none;
    box-shadow: 0 6px 10px rgba(0,0,0,0.15);

    white-space: nowrap;   /* prevents text from breaking */
    min-width: 160px;      /* ensures enough width */
}

div[data-testid="stColumn"] button:hover {
    background-color: #1b5e20;
}

/* Text Area */
textarea {
    border-radius: 10px !important;
}

/* Result Cards */
.result-card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-title'>Welcome to TrueHire</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Your Job Partner</p>", unsafe_allow_html=True)

# Session state
if "show_input" not in st.session_state:
    st.session_state.show_input = False

def show_input():
    st.session_state.show_input = True

# Center Button
col1, col2, col3 = st.columns([1,1,1])

with col2:
    if not st.session_state.show_input:
        st.button("Add Job Details", on_click=show_input)

# Input Area
if st.session_state.show_input:

    st.write("")
    st.subheader("Paste Job Description")

    job_text = st.text_area(
        "Enter job posting text here",
        height=250,
        placeholder="Paste the job description..."
    )

    st.write("")

    c1, c2, c3 = st.columns([2,1,2])

    with c2:
        check = st.button("CHECK")

    if check:

        if job_text.strip() == "":
            st.error("Please enter job description first.")
        else:

            # Progress Animation
            progress = st.progress(0)

            for i in range(100):
                time.sleep(0.02)
                progress.progress(i + 1)

            st.success("Analysis Complete")

            # Example results (until backend ready)
            real_percent = 90
            fake_percent = 10

            st.write("")
            st.markdown("---")
            st.subheader("Detection Result")

            r1, r2 = st.columns(2)

            with r1:
                st.metric("Real Job Probability", f"{real_percent}%")

            with r2:
                st.metric("Fake Job Probability", f"{fake_percent}%")

            if real_percent > fake_percent:
                st.success("Result: This job posting appears REAL")
            else:
                st.error("Result: This job posting appears FAKE")