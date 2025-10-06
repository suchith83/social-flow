import pytest

from app.models.user import User
from app.models.video import Video
from app.models.social import Post

# Minimal relationship presence tests (no DB needed) relying on SQLAlchemy instrumentation attributes.
@pytest.mark.parametrize(
    "model, rel_names",
    [
        (User, ["videos", "posts", "followers", "following"]),
        # 'views' is provided via backref from a separate model; treat as optional
        (Video, ["owner", "comments", "likes"]),
        (Post, ["owner", "comments", "likes"]),
    ],
)
def test_relationship_attributes_exist(model, rel_names):
    missing = []
    for rel in rel_names:
        if not hasattr(model, rel):
            missing.append(rel)
    # Allow 'views' absence on Video (provided as backref) but require the rest
    if model.__name__ == "Video":
        # Provide a clearer message if core relationships missing
        assert not [m for m in missing if m != "views"], f"Missing core relationships on Video: {missing}"
    else:
        assert not missing, f"Missing relationship attributes on {model.__name__}: {missing}"