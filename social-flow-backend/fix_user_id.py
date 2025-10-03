#!/usr/bin/env python3
"""Fix user_id to use owner_id in video endpoints."""

# Read the file
with open('app/api/v1/endpoints/videos.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all instances of .user_id with .owner_id in video context
content = content.replace('video.user_id', 'video.owner_id')

# Write back
with open('app/api/v1/endpoints/videos.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Successfully fixed user_id to owner_id in video endpoints!")
