import io
import pytest
from app.infrastructure.storage.manager import get_storage_manager

@pytest.mark.skip(reason="Requires valid AWS credentials - should use mocked S3")
@pytest.mark.asyncio
async def test_storage_manager_basic_upload_download(tmp_path):
    sm = get_storage_manager()
    data = b"hello-world"
    f = io.BytesIO(data)
    meta = await sm.upload_file(f, key="test/hello.txt")
    assert meta.key.endswith("hello.txt")
    fetched = await sm.download_file("test/hello.txt")
    assert fetched == data
    exists = await sm.file_exists("test/hello.txt")
    assert exists is True

@pytest.mark.skip(reason="Requires valid AWS credentials - should use mocked S3")
@pytest.mark.asyncio
async def test_storage_manager_list_and_delete():
    sm = get_storage_manager()
    f = io.BytesIO(b"abc")
    await sm.upload_file(f, key="list/sample.txt")
    files = await sm.list_files(prefix="list/")
    assert any(m.key.endswith("sample.txt") for m in files)
    deleted = await sm.delete_file("list/sample.txt")
    assert deleted is True