from fastapi import FastAPI

app = FastAPI(title="Fraud Detection API")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": False}