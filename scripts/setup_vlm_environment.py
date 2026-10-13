"""Isolated VLM setup; no shared packages are changed."""
from bootstrap import environment
if __name__ == "__main__":
    print(environment(".venv-vlm", "requirements-vlm.txt", True))
