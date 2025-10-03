#!/usr/bin/env python3
"""Fix password inconsistency in video tests."""

# Read the file
with open('tests/integration/api/test_video_endpoints.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace password used when creating users
content = content.replace('password="TestPassword123!",', 'password="TestPassword123",')

# Write back
with open('tests/integration/api/test_video_endpoints.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Successfully fixed password inconsistencies in video tests!")
