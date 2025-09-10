from fastapi import FastAPI
import redis

app = FastAPI()

r = redis.Redis(host='localhost', port=6379)

@app.post("/view/{video_id}")
def increment_view(video_id: str):
    r.incr(f'view:{video_id}')
    return {"status": "incremented"}

@app.get("/view/{video_id}")
def get_view(video_id: str):
    return {"views": r.get(f'view:{video_id}')}
