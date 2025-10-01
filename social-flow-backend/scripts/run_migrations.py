"""
Database Migration Runner

This script helps run Alembic migrations for the Social Flow backend.
"""

import sys
import subprocess
from pathlib import Path

def run_migrations():
    """Run all pending database migrations."""
    print("🔄 Running database migrations...")
    
    try:
        # Run alembic upgrade head
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            check=True
        )
        
        print("✅ Migrations completed successfully!")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print("❌ Migration failed!")
        print(f"Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ Alembic not found. Please install it:")
        print("   pip install alembic")
        return False

def check_migration_status():
    """Check the current migration status."""
    print("📊 Checking migration status...")
    
    try:
        result = subprocess.run(
            ["alembic", "current"],
            capture_output=True,
            text=True,
            check=True
        )
        
        print("Current migration:")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return False

def create_migration(message: str):
    """Create a new migration."""
    print(f"📝 Creating new migration: {message}")
    
    try:
        result = subprocess.run(
            ["alembic", "revision", "--autogenerate", "-m", message],
            capture_output=True,
            text=True,
            check=True
        )
        
        print("✅ Migration created successfully!")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create migration: {e.stderr}")
        return False

def rollback_migration(steps: int = 1):
    """Rollback migrations."""
    print(f"⏪ Rolling back {steps} migration(s)...")
    
    try:
        result = subprocess.run(
            ["alembic", "downgrade", f"-{steps}"],
            capture_output=True,
            text=True,
            check=True
        )
        
        print("✅ Rollback completed successfully!")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Rollback failed: {e.stderr}")
        return False

def show_migration_history():
    """Show migration history."""
    print("📜 Migration history:")
    
    try:
        result = subprocess.run(
            ["alembic", "history"],
            capture_output=True,
            text=True,
            check=True
        )
        
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run_migrations.py upgrade    - Run all pending migrations")
        print("  python run_migrations.py status     - Check migration status")
        print("  python run_migrations.py history    - Show migration history")
        print("  python run_migrations.py create <message> - Create new migration")
        print("  python run_migrations.py rollback <steps> - Rollback migrations")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "upgrade":
        success = run_migrations()
    elif command == "status":
        success = check_migration_status()
    elif command == "history":
        success = show_migration_history()
    elif command == "create":
        if len(sys.argv) < 3:
            print("Error: Please provide a migration message")
            sys.exit(1)
        message = " ".join(sys.argv[2:])
        success = create_migration(message)
    elif command == "rollback":
        steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        success = rollback_migration(steps)
    else:
        print(f"Unknown command: {command}")
        success = False
    
    sys.exit(0 if success else 1)
