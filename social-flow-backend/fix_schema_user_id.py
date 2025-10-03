#!/usr/bin/env python3
"""Fix user_id to use owner_id in video schemas."""

# Read the file
with open('app/schemas/video.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace user_id with owner_id in schema definitions
content = content.replace('user_id: UUID', 'owner_id: UUID')

# Write back
with open('app/schemas/video.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Successfully fixed user_id to owner_id in video schemas!")
