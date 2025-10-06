from app.core.config import settings


def test_basic_settings_values_present():
    assert settings.PROJECT_NAME
    assert settings.VERSION
    assert settings.API_V1_STR.startswith("/api/")
    assert isinstance(settings.FEATURE_ML_ENABLED, bool)


def test_feature_flags_exist():
    flags = [
        settings.FEATURE_S3_ENABLED,
        settings.FEATURE_REDIS_ENABLED,
        settings.FEATURE_ML_ENABLED,
        settings.FEATURE_CELERY_ENABLED,
    ]
    assert all(isinstance(f, bool) for f in flags)
