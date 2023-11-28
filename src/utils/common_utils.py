from datetime import datetime

_all__ = ["log"]

def log(message):
  print(f'{datetime.now().isoformat()} {message}')