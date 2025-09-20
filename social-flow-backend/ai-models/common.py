from typing import Any, Dict, List


class ModelBase:
    """Minimal model interface used by local stubs."""

    def predict(self, inputs: Any) -> Any:
        raise NotImplementedError


class DummyModel(ModelBase):
    def __init__(self, name: str = "dummy"):
        self.name = name

    def predict(self, inputs: Any) -> Dict[str, Any]:
        return {"model": self.name, "result": inputs}


def load_dummy_model(name: str = "dummy", **_cfg) -> DummyModel:
    """Return a simple DummyModel instance. Accepts arbitrary config for compatibility."""
    return DummyModel(name=name)
