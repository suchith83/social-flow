from fastapi import FastAPI

app = FastAPI()

@app.get("/ads/{video_id}")
def get_ads(video_id: str):
    # TODO: Return ads
    return {"ads": []}
