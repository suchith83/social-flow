import tempfile
from thumbnails.dedupe import compute_phash

def test_phash_on_sample_image(tmp_path):
    img_path = tmp_path / "img.jpg"
    # create a small blank jpeg via Pillow
    from PIL import Image
    img = Image.new("RGB", (64,64), color=(255,0,0))
    img.save(img_path)
    ph = compute_phash(str(img_path))
    assert ph is not None
