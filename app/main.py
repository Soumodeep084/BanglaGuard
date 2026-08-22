from fastapi import FastAPI

app = FastAPI(title="BanglaGuard API", version="0.1.0")


@app.get("/")
def root():
    return {"message": "BanglaGuard API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}
