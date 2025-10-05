# Prevent pytest from trying to treat generated text result artifacts as tests.
def pytest_ignore_collect(path, config):  # type: ignore[override]
    p = str(path)
    if p.endswith('test_results.txt') or p.endswith('test_results_latest.txt') or p.endswith('test_output.txt'):
        return True
    return False
