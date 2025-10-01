from fastapi import FastAPI
from elasticsearch import Elasticsearch

app = FastAPI()

es = Elasticsearch(['http://localhost:9200'])

@app.get("/search")
def search(query: str):
    res = es.search(index="videos", body={"query": {"match": {"title": query}}})
    return res

@app.get("/autocomplete")
def autocomplete(partial: str):
    # TODO: Autocomplete
    return []

@app.get("/hashtag/{tag}")
def get_hashtag(tag: str):
    # TODO: Hashtag search
    return {}
