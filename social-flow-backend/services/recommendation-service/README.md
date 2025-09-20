# Recommendation Service (Python)

Lightweight recommendation microservice used for prototyping and local
development. It exposes a small FastAPI app with endpoints:

- GET /health
- GET /recommendations/{user_id}
- GET /trending
- POST /feedback

Quick start (Windows PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install -r runtime-requirements.txt
python -m uvicorn src.main:app --port 8003
```

Notes:
- This service uses a placeholder inference implementation. Replace with
  calls to your model (SageMaker/Bedrock) or a real inference service for
  production.
- For local development, install the monorepo `common` libraries in editable
  mode so the service can import shared helpers:

```powershell
cd ..\..\common\libraries\python
python -m pip install -e .
```

  After installing `social_flow_common` editable, you can run the service as
  described above without modifying PYTHONPATH.
