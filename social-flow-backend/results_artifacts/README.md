# Test Results Artifacts

This directory stores archived test result artifacts that should not be collected by pytest.

Files here use extensions like `.txt` or `.json` but remain outside the configured `testpaths = tests` in `pytest.ini` so they are ignored by collection.

Previously, `test_results_final.txt` in the repository root caused a pytest collection error on Windows due to a non-UTF8 byte. That file has been removed and its contents are archived as `test_results_final_archive.txt` in this folder.

Do not place test artifact files at repository root to avoid accidental collection.# Test Result Artifacts

This directory stores historical pytest result artifacts (moved from project root) to prevent pytest from
attempting to collect them as test modules. Files here are ignored by test discovery.
