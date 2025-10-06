import pytest
import asyncio
from app.ml_pipelines.orchestrator import get_orchestrator, PipelineType

@pytest.mark.asyncio
async def test_orchestrator_submit_and_status():
    orch = await get_orchestrator()
    # Submit a lightweight cache warming task
    task_id = await orch.submit_task(
        pipeline_type=PipelineType.CACHE_WARMING,
        name="warm-recs",
        config={"cache_types": ["recommendations"], "limit": 1},
        priority=9,
    )
    assert isinstance(task_id, str)
    # Give the executor loop a moment
    await asyncio.sleep(0.5)
    status = await orch.get_task_status(task_id)
    assert status is not None
    queue = await orch.get_queue_status()
    assert "pending_tasks" in queue and "running_tasks" in queue

@pytest.mark.asyncio
async def test_orchestrator_queue_status_shape():
    orch = await get_orchestrator()
    qs = await orch.get_queue_status()
    for field in ["pending_tasks", "running_tasks", "completed_tasks", "max_concurrent", "is_running"]:
        assert field in qs