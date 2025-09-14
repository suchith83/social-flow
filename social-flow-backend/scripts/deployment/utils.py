# scripts/deployment/utils.py
import subprocess
import logging


def run_command(cmd, cwd=None):
    """
    Runs a system command with logging and error handling.
    """
    logging.info(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, check=True
        )
        logging.info(result.stdout.strip())
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {e.stderr}")
        raise
