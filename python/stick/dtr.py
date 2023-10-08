# Reference: https://openreview.net/pdf?id=Vfs_2RnOD0H

import sys, time, math
import heapq
from typing import Any

ENABLE_DTR = False


class enable_dtr:
    def __init__(self, enable) -> None:
        self.enable = enable

    def __enter__(self) -> None:
        global ENABLE_DTR
        self.prev = ENABLE_DTR
        ENABLE_DTR = self.enable

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global ENABLE_DTR
        ENABLE_DTR = self.prev
    

def get_limit():
    # TODO: read the config file
    return 0

class Dtr:
    tensors = {}
    mem_cnt = 0
    mem_limit = get_limit()

    @staticmethod
    def del_tensor(tensor):
        (ts, mem, cost) = Dtr.tensors[tensor]
        Dtr.mem_cnt -= mem
        tensor.outputs = None
        del Dtr.tensors[tensor]
    
    @staticmethod
    def search_evict():
        assert Dtr.mem_cnt >= Dtr.mem_limit
        assert len(Dtr.tensors) > 0

        min_score = sys.maxsize
        min_tensor = None
        now = time.perf_counter()

        for tensor, (ts, mem, cost) in Dtr.tensors.items():
            score = cost / ((now - ts) * mem)
            if score < min_score:
                min_score = score
                min_tensor = tensor
        
        Dtr.del_tensor(min_tensor)
    
    @staticmethod
    def search_evict_by_sampling():
        assert Dtr.mem_cnt >= Dtr.mem_limit
        assert len(Dtr.tensors) > 0

        min_score = sys.maxsize
        min_tensor = None
        now = time.perf_counter()
        n = int(math.sqrt(len(Dtr.tensors)))

        for i, (tensor, (ts, mem, cost)) in enumerate(Dtr.tensors.items()):
            if i % n != 0:
                continue
            score = cost / ((now - ts) * mem)
            if score < min_score:
                min_score = score
                min_tensor = tensor
        
        Dtr.del_tensor(min_tensor)
    
    @staticmethod
    def search_evict_by_top_n():
        """
        This function has no reference, so we might be the first proposer and it may not work.
        But it's very fast in our test and may have theoritical bound, so I set it as default.
        """

        # evict 1% everytime
        n = max(1, len(Dtr.tensors) // 100)
        now = time.perf_counter()
        items = list(Dtr.tensors.items())
        sorted_items = heapq.nsmallest(n, items, key=lambda x: x[1][2] / ((now - x[1][0]) * x[1][1]))
        for i in sorted_items:
            Dtr.del_tensor(i[0])
        
    @staticmethod
    def add(tensor, ts, mem, cost):
        assert tensor not in Dtr.tensors

        # ts: timestamp
        Dtr.tensors[tensor] = (ts, mem, cost)
        Dtr.mem_cnt += mem
        while Dtr.mem_cnt >= Dtr.mem_limit:
            Dtr.search_evict()
            # Dtr.search_evict_by_sampling()
            # Dtr.search_evict_by_top_n()

    @staticmethod
    def get_obj(tensor):
        if tensor.outputs is None:
            return None
        if tensor in Dtr.tensors:
            (_ts, mem, cost) = Dtr.tensors[tensor]
            Dtr.tensors[tensor] = (time.perf_counter(), mem, cost)
        return tensor.outputs
