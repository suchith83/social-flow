#!/usr/bin/env python3
"""
Test runner script for the Social Flow backend.

This script provides a convenient way to run different types of tests
with various configurations and options.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {description}:")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def run_unit_tests(verbose=False, coverage=False):
    """Run unit tests."""
    command = "python -m pytest tests/unit/"
    
    if verbose:
        command += " -v"
    
    if coverage:
        command += " --cov=app --cov-report=html --cov-report=term"
    
    return run_command(command, "Unit Tests")


def run_integration_tests(verbose=False, coverage=False):
    """Run integration tests."""
    command = "python -m pytest tests/integration/"
    
    if verbose:
        command += " -v"
    
    if coverage:
        command += " --cov=app --cov-report=html --cov-report=term"
    
    return run_command(command, "Integration Tests")


def run_performance_tests(verbose=False):
    """Run performance tests."""
    command = "python -m pytest tests/performance/ -m performance"
    
    if verbose:
        command += " -v"
    
    return run_command(command, "Performance Tests")


def run_security_tests(verbose=False):
    """Run security tests."""
    command = "python -m pytest tests/security/ -m security"
    
    if verbose:
        command += " -v"
    
    return run_command(command, "Security Tests")


def run_all_tests(verbose=False, coverage=False):
    """Run all tests."""
    command = "python -m pytest tests/"
    
    if verbose:
        command += " -v"
    
    if coverage:
        command += " --cov=app --cov-report=html --cov-report=term"
    
    return run_command(command, "All Tests")


def run_specific_test(test_path, verbose=False):
    """Run a specific test file or test function."""
    command = f"python -m pytest {test_path}"
    
    if verbose:
        command += " -v"
    
    return run_command(command, f"Specific Test: {test_path}")


def run_linting():
    """Run code linting."""
    commands = [
        ("python -m flake8 app/", "Flake8 Linting"),
        ("python -m mypy app/", "MyPy Type Checking"),
        ("python -m bandit -r app/", "Bandit Security Linting"),
    ]
    
    results = []
    for command, description in commands:
        results.append(run_command(command, description))
    
    return all(results)


def run_formatting():
    """Run code formatting."""
    commands = [
        ("python -m black app/", "Black Code Formatting"),
        ("python -m isort app/", "Import Sorting"),
    ]
    
    results = []
    for command, description in commands:
        results.append(run_command(command, description))
    
    return all(results)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test runner for Social Flow backend")
    parser.add_argument("--type", choices=["unit", "integration", "performance", "security", "all"], 
                       default="all", help="Type of tests to run")
    parser.add_argument("--test", help="Specific test file or function to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", help="Generate coverage report")
    parser.add_argument("--lint", action="store_true", help="Run linting")
    parser.add_argument("--format", action="store_true", help="Run code formatting")
    parser.add_argument("--all-checks", action="store_true", help="Run all checks (tests, linting, formatting)")
    
    args = parser.parse_args()
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    success = True
    
    if args.all_checks:
        print("Running all checks...")
        success &= run_formatting()
        success &= run_linting()
        success &= run_all_tests(verbose=args.verbose, coverage=args.coverage)
    elif args.lint:
        success &= run_linting()
    elif args.format:
        success &= run_formatting()
    elif args.test:
        success &= run_specific_test(args.test, verbose=args.verbose)
    else:
        if args.type == "unit":
            success &= run_unit_tests(verbose=args.verbose, coverage=args.coverage)
        elif args.type == "integration":
            success &= run_integration_tests(verbose=args.verbose, coverage=args.coverage)
        elif args.type == "performance":
            success &= run_performance_tests(verbose=args.verbose)
        elif args.type == "security":
            success &= run_security_tests(verbose=args.verbose)
        elif args.type == "all":
            success &= run_all_tests(verbose=args.verbose, coverage=args.coverage)
    
    if success:
        print("\n✅ All checks passed!")
        sys.exit(0)
    else:
        print("\n❌ Some checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
