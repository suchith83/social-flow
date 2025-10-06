# Prevent pytest from trying to treat generated text result artifacts as tests.
def pytest_ignore_collect(path, config):  # type: ignore[override]
    """Prevent pytest from collecting persistent test artifact files.

    Some CI or local quality gate runs archive prior full-suite outputs like
    test_results_final.txt that may contain non-UTF8 bytes (e.g. 0xFF) and
    would cause collection-time decode errors if pytest attempts to read
    them as source files. We explicitly ignore those here.
    """
    p = str(path)
    artifact_names = (
        'test_results.txt',
        'test_results_latest.txt',
        'test_output.txt',
        'test_results_final.txt',  # newly added to stop UnicodeDecodeError
        'final_unit_results.txt',
        'advanced_test_results.json',
    )
    return any(p.endswith(name) for name in artifact_names)
