import pytest
from hypothesis import given, strategies as st

def slugify(text: str) -> str:
    """Utility function that converts text into URL-friendly slugs."""
    return text.lower().replace(" ", "-")

def test_slugify_basic():
    assert slugify("Hello World") == "hello-world"

@given(st.text())
def test_slugify_never_crashes(random_text):
    """
    Property-based test: slugify should never throw exceptions,
    regardless of input.
    """
    slugify(random_text)

@pytest.mark.parametrize("inp,expected", [
    ("Video Title", "video-title"),
    ("Already-Slug", "already-slug"),
    ("  Trim  Me  ", "--trim--me--"),
])
def test_slugify_parametrized(inp, expected):
    assert slugify(inp) == expected
