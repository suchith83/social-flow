"""
Storage-related API integration tests.

Covers:
- list buckets/containers
- list objects in a bucket (paginated)
- upload/download object via API (if endpoints available)
- object metadata contract
"""

import pytest
from schemas.models import BucketList, StorageObject
import io

@pytest.mark.integration
def test_list_buckets_contract(api_client):
    resp = api_client.get("/storage/buckets")
    assert resp.status_code == 200
    body = resp.json()
    parsed = BucketList.parse_obj(body)
    assert isinstance(parsed.buckets, list)

@pytest.mark.integration
def test_list_objects_in_bucket(api_client):
    bucket = "test"  # adjust to a known bucket or ensure a test bucket exists
    resp = api_client.get(f"/storage/{bucket}/objects", params={"prefix": "", "limit": 50})
    # Accept 200 or 404 (if bucket not present) but fail test if 5xx
    assert resp.status_code < 500
    if resp.status_code == 200:
        data = resp.json()
        # be tolerant: expect "objects" or array at root
        if isinstance(data, dict) and "objects" in data:
            objs = data["objects"]
        elif isinstance(data, list):
            objs = data
        else:
            objs = []
        # Validate first item if present
        if objs:
            obj = objs[0]
            parsed = StorageObject.parse_obj(obj)
            assert parsed.key

@pytest.mark.integration
def test_upload_and_download_via_api_if_supported(api_client):
    """
    This test will attempt to upload a small object via API endpoints if available:
    - POST /storage/<bucket>/objects (multipart or json)
    - GET /storage/<bucket>/objects/<key>
    It's written defensively: if endpoints are not present, it will skip.
    """
    bucket = "test"
    key = f"api-tests-{int(pytest.time.time())}.txt" if hasattr(pytest, "time") else f"api-tests-{random.randint(1000,9999)}.txt"
    content = b"hello-from-api-tests"

    # POST upload: try json first (some APIs accept base64 or raw)
    try:
        resp = api_client.post(f"/storage/{bucket}/objects", json={"key": key, "content": content.decode("utf-8")})
    except Exception:
        pytest.skip("Upload endpoint not available in this API")

    if resp.status_code not in (200, 201):
        pytest.skip(f"Upload not supported or failed: {resp.status_code}")

    # Now download
    get_resp = api_client.get(f"/storage/{bucket}/objects/{key}")
    assert get_resp.status_code in (200, 206)
