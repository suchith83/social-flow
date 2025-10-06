IMPORTANT_IMPORTS = [
    ("app.main", None),
    ("app.core.config", "settings"),
    ("app.core.database", "get_engine"),
    ("app.api.v1.router", "api_router"),
    ("app.models.user", "User"),
    ("app.videos.services.video_service", "VideoService"),
    ("app.services.recommendation_service", "RecommendationService"),
]


def test_core_imports_subset():
    for module_name, attr in IMPORTANT_IMPORTS:
        mod = __import__(module_name, fromlist=[attr] if attr else [])
        if attr:
            getattr(mod, attr)
