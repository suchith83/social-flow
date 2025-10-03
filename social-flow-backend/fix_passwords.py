#!/usr/bin/env python3
"""Fix password in video test file."""

with open('tests/integration/api/test_video_endpoints.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all instances of TestPassword123! with TestPassword123
content = content.replace('"password": "TestPassword123!"', '"password": "TestPassword123"')

with open('tests/integration/api/test_video_endpoints.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Successfully fixed password references in video tests")
