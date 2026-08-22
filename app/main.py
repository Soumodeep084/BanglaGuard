from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="BanglaGuard API", version="0.1.0")


class SMSRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "BanglaGuard API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/test-sms")
def test_sms(payload: SMSRequest):
    return {"status": "success", "message": "SMS received successfully by BanglaGuard API", "received_text": payload.text}
