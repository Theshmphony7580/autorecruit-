import heapq

class TopKTracker:
    def __init__(self, k: int = 100):
        self.k = k
        self.heap = []
    
    def push(self, candidate_id: str, score: float, candidate_data: dict):
        entry = (-score, candidate_id, candidate_data)
        
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, entry)
        elif score > -self.heap[0][0]:
            heapq.heapreplace(self.heap, entry)
            
    def get_ranked(self) -> list:
        return sorted(self.heap, key=lambda x: -x[0])
