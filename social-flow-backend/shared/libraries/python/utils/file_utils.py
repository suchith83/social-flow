# file_utils.py
import os
import shutil
from pathlib import Path


class FileUtils:
    """
    File system utilities.
    """

    @staticmethod
    def ensure_dir(path: str):
        Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def copy(src: str, dst: str):
        shutil.copy(src, dst)

    @staticmethod
    def move(src: str, dst: str):
        shutil.move(src, dst)

    @staticmethod
    def delete(path: str):
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
