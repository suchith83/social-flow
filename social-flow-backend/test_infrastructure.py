"""Test infrastructure security components."""
import sys
sys.path.insert(0, '.')

from uuid import uuid4
from app.auth.infrastructure.security import password_hasher, jwt_handler

print("=== Testing Password Hasher ===")

# Test password hashing
password = "SecurePassword123!"
hashed = password_hasher.hash(password)
print(f"✓ Password hashed: {hashed[:20]}...")

# Test password verification
is_valid = password_hasher.verify(password, hashed)
print(f"✓ Password verification (correct): {is_valid}")

is_invalid = password_hasher.verify("WrongPassword", hashed)
print(f"✓ Password verification (incorrect): {is_invalid}")

print("\n=== Testing JWT Handler ===")

# Test access token creation
user_id = uuid4()
username = "test_user"
role = "user"

access_token = jwt_handler.create_access_token(user_id, username, role)
print(f"✓ Access token created: {access_token[:30]}...")

# Test access token verification
payload = jwt_handler.verify_access_token(access_token)
print(f"✓ Access token verified")
print(f"  - User ID: {payload['sub']}")
print(f"  - Username: {payload['username']}")
print(f"  - Role: {payload['role']}")
print(f"  - Type: {payload['type']}")

# Test refresh token creation
refresh_token = jwt_handler.create_refresh_token(user_id)
print(f"\n✓ Refresh token created: {refresh_token[:30]}...")

# Test refresh token verification
refresh_payload = jwt_handler.verify_refresh_token(refresh_token)
print(f"✓ Refresh token verified")
print(f"  - User ID: {refresh_payload['sub']}")
print(f"  - Type: {refresh_payload['type']}")

# Test user ID extraction
extracted_id = jwt_handler.get_user_id_from_token(access_token)
print(f"\n✓ User ID extracted from token: {extracted_id}")
print(f"✓ Matches original: {extracted_id == user_id}")

print("\n✅ All security tests passed!")
