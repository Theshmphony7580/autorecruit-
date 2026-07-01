import heapq

class TopKTracker:
    def __init__(self, k: int = 100):
        self.k = k
        self.heap = []
    
    def push(self, candidate_id: str, score: float, candidate_data: dict):
        # Push positive score so the smallest score is at the root of the min-heap
        entry = (score, candidate_id, candidate_data)
        
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, entry)
        elif score > self.heap[0][0]:
            heapq.heapreplace(self.heap, entry)
            
    def get_ranked(self) -> list:
        # Sort by score descending (-x[0]), and tie-break by candidate_id ascending (x[1])
        return sorted(self.heap, key=lambda x: (-x[0], x[1]))

