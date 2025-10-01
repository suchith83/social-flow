from fastapi import FastAPI

app = FastAPI()

@app.post("/pay")
def process_payment(amount: float):
    # TODO: Process with Stripe
    return {"status": "success"}
