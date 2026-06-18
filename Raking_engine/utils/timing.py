import time

class Timer:
    def __init__(self, name=""):
        self.name = name
        self.start = None
    
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        print(f"[{self.name}] took {elapsed:.2f}s")
