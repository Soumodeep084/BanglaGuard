import streamlit as st

st.set_page_config(
    page_title="BanglaGuard - Bengali SMS Spam Detection",
    page_icon="🛡️",
    layout="centered",
)

st.title("🛡️ BanglaGuard")
st.subheader("Bengali SMS Spam Detection System")

st.write(
    "Enter a Bengali SMS message below to analyze its content."
)

sms_input = st.text_area(
    "Bengali SMS Text",
    placeholder="এখানে বাংলা এসএমএস টেক্সট লিখুন... (e.g., আপনি লটারি জিতেছেন!)",
    height=140,
)

if st.button("Analyze SMS", type="primary"):
    if sms_input.strip():
        st.info(f"**Entered SMS:** {sms_input.strip()}")
    else:
        st.warning("Please enter an SMS message before analyzing.")
