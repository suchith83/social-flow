#!/usr/bin/env python3
"""Fix is_creator to use role field instead."""

import re

# Read the test file
with open('tests/integration/api/test_video_endpoints.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace is_creator = True with role = UserRole.CREATOR
content = content.replace('test_user.is_creator = True', 'test_user.role = UserRole.CREATOR')
content = content.replace('other_user.is_creator = True', 'other_user.role = UserRole.CREATOR')

# Make sure UserRole is imported
if 'from app.models.user import User, UserRole' not in content:
    # Find the import line for User
    content = content.replace(
        'from app.models.user import User',
        'from app.models.user import User, UserRole'
    )

# Write back
with open('tests/integration/api/test_video_endpoints.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Successfully fixed creator role references in video tests!")
