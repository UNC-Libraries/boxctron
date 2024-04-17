from datetime import datetime
from pathlib import Path

_all__ = ["log"]

def log(message):
  print(f'{datetime.now().isoformat()} {message}')
