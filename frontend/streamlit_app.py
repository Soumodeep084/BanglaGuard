import os
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="BanglaGuard - Bengali SMS Spam Detection",
    page_icon="🛡️",
    layout="centered",
)

def check_backend_health(url: str):
    try:
        res = requests.get(f"{url}/health", timeout=2)
        if res.status_code == 200:
            return True, res.json().get("status", "healthy")
        return False, f"HTTP {res.status_code}"
    except requests.exceptions.RequestException:
        return False, "Offline"

def send_sms_to_backend(url: str, text: str):
    try:
        res = requests.post(f"{url}/test-sms", json={"text": text}, timeout=5)
        if res.status_code == 200:
            return True, res.json()
        return False, f"Backend returned HTTP {res.status_code}: {res.text}"
    except requests.exceptions.ConnectionError:
        return False, "Could not connect to FastAPI backend. Ensure the server is running on http://127.0.0.1:8000."
    except requests.exceptions.Timeout:
        return False, "Request timed out while contacting backend."
    except requests.exceptions.RequestException as e:
        return False, f"An error occurred: {e}"

st.title("🛡️ BanglaGuard")
st.subheader("Bengali SMS Spam Detection System")

is_healthy, health_msg = check_backend_health(API_BASE_URL)
if is_healthy:
    st.success(f"🟢 Backend API: Connected ({API_BASE_URL})")
else:
    st.warning(f"🔴 Backend API: Unavailable ({API_BASE_URL}). Run: `uv run uvicorn app.main:app --reload`")

st.write("Enter a Bengali SMS message below to test communication with the backend.")

sms_input = st.text_area(
    "Bengali SMS Text",
    placeholder="এখানে বাংলা এসএমএস টেক্সট লিখুন... (e.g., প্রিয় গ্রাহক, আপনি জিতেছেন নগদ পুরস্কার!)",
    height=140,
)

if st.button("Analyze SMS", type="primary"):
    if not sms_input.strip():
        st.warning("Please enter an SMS message before analyzing.")
    else:
        with st.spinner("Communicating with backend..."):
            success, response_data = send_sms_to_backend(API_BASE_URL, sms_input.strip())

        if success:
            st.success("Response received from FastAPI backend:")
            st.json(response_data)
        else:
            st.error(f"Failed to communicate with backend: {response_data}")
