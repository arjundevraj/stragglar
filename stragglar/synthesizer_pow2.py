import sys
import numpy as np
import time
import math

class GPU:
    def __init__(self, rank, n):
        self.rank = rank
        self.chunks = np.zeros(n-1)
        self.world_size = n
        self.last_received_chunk = -1
    
    def is_fully_reduced(self):
        return np.sum(self.chunks) == self.world_size - 1
    
    def available_chunks(self):
        return np.where(self.chunks == 1)
    
    def get_highest_chunk(self):
        for i in range(len(self.chunks)-1, -1, -1):
            if self.chunks[i] == 1:
                return i
        return None
    
    def compute_similarity(self, other_gpu):
        my_chunk = self.get_highest_chunk()
        other_chunk = other_gpu.get_highest_chunk()

        if my_chunk is None and other_chunk is None:
            return (0, 0)
        if my_chunk is None:
            return (0, 1)
        if other_chunk is None:
            return (1, 0)
        
        if self.chunks[other_chunk] == 0 and other_gpu.chunks[my_chunk] == 0:
            return (1,1)
        elif self.chunks[other_chunk] == 1 and other_gpu.chunks[my_chunk] == 0:
            return (1,0)
        elif self.chunks[other_chunk] == 0 and other_gpu.chunks[my_chunk] == 1:
            return (0,1)
        else:
            return (0,0)

class OneWayMatching:
    def __init__(self, send_gpu, recv_gpu, chunk_id):
        assert send_gpu.chunks[chunk_id] == 1
        self.send_gpu = send_gpu
        self.recv_gpu = recv_gpu
        self.chunk_id = chunk_id
        recv_gpu.chunks[chunk_id] = 1
        recv_gpu.last_received_chunk = chunk_id
    def __str__(self):
        return f"OneWayMatching: {self.send_gpu.rank} -> {self.recv_gpu.rank}, chunk_id: {self.chunk_id}"

class TwoWayMatching:
    def __init__(self, gpu1, gpu2, chunk_id1, chunk_id2):
        assert gpu1.chunks[chunk_id1] == 1
        assert gpu2.chunks[chunk_id2] == 1
        self.gpu1 = gpu1
        self.gpu2 = gpu2
        self.chunk_id1 = chunk_id1
        self.chunk_id2 = chunk_id2

        gpu1.chunks[chunk_id2] = 1
        gpu1.last_received_chunk = chunk_id2
        gpu2.chunks[chunk_id1] = 1
        gpu2.last_received_chunk = chunk_id1
    
    def __str__(self):
        return f"TwoWayMatching: {self.gpu1.rank} -> {self.gpu2.rank}, chunk_id: {self.chunk_id1}; {self.gpu2.rank} -> {self.gpu1.rank}, chunk_id: {self.chunk_id2}"
    
class StragglerMatching:
    def __init__(self, curr_gpu, straggler_gpu, chunk_id):
        self.gpu = curr_gpu
        self.straggler_gpu = straggler_gpu
        self.chunk_id = chunk_id
        straggler_gpu.chunks[chunk_id] = 1
        curr_gpu.chunks[chunk_id] = 1
        straggler_gpu.last_received_chunk = chunk_id
        curr_gpu.last_received_chunk = chunk_id
    
    def __str__(self):
        return f"StragglerMatching: {self.gpu.rank} <-> {self.straggler_gpu.rank}, chunk_id: {self.chunk_id}"

def valid_round(round_schedule):
    send_gpus = set()
    recv_gpus = set()
    for matching in round_schedule:
        if type(matching) == OneWayMatching:
            assert matching.send_gpu.rank not in send_gpus, "GPU already sent in this round"
            assert matching.recv_gpu.rank not in recv_gpus, "GPU already received in this round"
            send_gpus.add(matching.send_gpu.rank)
            recv_gpus.add(matching.recv_gpu.rank)
        elif type(matching) == TwoWayMatching:
            assert matching.gpu1.rank not in send_gpus, "GPU already sent in this round"
            assert matching.gpu2.rank not in recv_gpus, "GPU already received in this round"
            send_gpus.add(matching.gpu1.rank)
            recv_gpus.add(matching.gpu2.rank)
            assert matching.gpu2.rank not in send_gpus, "GPU already sent in this round"
            assert matching.gpu1.rank not in recv_gpus, "GPU already received in this round"
            send_gpus.add(matching.gpu2.rank)
            recv_gpus.add(matching.gpu1.rank)

def get_active_chunks(gpus):
    active_chunks = {}
    for chunk_id in range(len(gpus) - 1):
        if gpus[len(gpus) - 1].chunks[chunk_id] == 0:
            continue
        if all(gpu.chunks[chunk_id] == 1 for gpu in gpus):
            continue
        active_chunks[chunk_id] = 0
        for gpu in gpus[:len(gpus) - 1]:
            if gpu.chunks[chunk_id] == 1:
                active_chunks[chunk_id] += 1
    return active_chunks

def get_active_chunk_sets(gpus, round_id):
    logn = math.ceil(np.log2(len(gpus)))
    n = len(gpus)
    i = min(round_id - logn, n - 2)
    j = min(round_id, n - 1)
    active_chunks = {i: [] for i in range(i, j)}
    for gpu in gpus:
        if round_id < n-1 and (gpu.rank == len(gpus) - 1 or gpu.rank == round_id):
            continue
        if round_id >= n - 1 and gpu.rank == n - 1:
            active_chunks[n-2].append(gpu.rank)
            continue
        for chunk_id in range(i, j):
            if gpu.chunks[chunk_id] == 1:
                active_chunks[chunk_id].append(gpu.rank)
                continue
    return active_chunks

def find_matching_opt(gpus, available, round_id):
    assert sum(available) > 0, "available_gpus must not be empty"
    assert sum(available) % 2 == 0, "available_gpus must be even"
    n = len(gpus)
    logn = math.ceil(np.log2(n))
    available_gpus = [gpu for gpu in gpus if available[gpu.rank] == 1]
    matchings = []
    if round_id < logn:
        have_chunks = [gpu.rank for gpu in available_gpus if sum(gpu.chunks) > 0]
        lack_chunks = [gpu.rank for gpu in available_gpus if gpu.rank > 2 * (logn - 1) and sum(gpu.chunks) == 0]
        # assert that no GPU in have_chunks is in lack_chunks
        assert len(set(have_chunks).intersection(set(lack_chunks))) == 0, "GPU in have_chunks is also in lack_chunks"
        # assert that no GPU in lack_chunks is in have_chunks
        assert len(set(lack_chunks).intersection(set(have_chunks))) == 0, "GPU in lack_chunks is also in have_chunks"

        # arbitrarily match GPUs with chunks to GPUs without chunks
        for have, lack in zip(have_chunks, lack_chunks):
            matchings.append((have, lack))
    
    elif round_id < n - 1:
        active_chunks = get_active_chunk_sets(gpus, round_id)
        for i in range(round_id+1, min(round_id+logn, n-1)):
            # get the min active chunk that is not gpu i's active chunk
            # find my active chunk and remove me from its list
            for chunk_id in range(round_id - logn, round_id): 
                if gpus[i].chunks[chunk_id] == 1:
                    active_chunks[chunk_id].remove(i)
                    if len(active_chunks[chunk_id]) == 0:
                        del active_chunks[chunk_id]
                    break
        for i in range(round_id+1, min(round_id+logn, n-1)):
            for chunk_id in range(round_id - logn, round_id): 
                if gpus[i].chunks[chunk_id] == 0:
                    partner = active_chunks[chunk_id].pop(0)
                    # if active_Chunks[chunk_id] is empty, remove it from the dict
                    if len(active_chunks[chunk_id]) == 0:
                        del active_chunks[chunk_id]
                    matchings.append((i, partner))
                    break
        min_active_chunk = min(active_chunks.keys())
        have_chunks = active_chunks[min_active_chunk]
        lack_chunks = []
        for chunk_id in range(min_active_chunk + 1, round_id):
            if chunk_id in active_chunks:
                lack_chunks.extend(active_chunks[chunk_id])
          # assert that no GPU in have_chunks is in lack_chunks
        assert len(set(have_chunks).intersection(set(lack_chunks))) == 0, "GPU in have_chunks is also in lack_chunks"
        # assert that no GPU in lack_chunks is in have_chunks
        assert len(set(lack_chunks).intersection(set(have_chunks))) == 0, "GPU in lack_chunks is also in have_chunks"
        assert len(have_chunks) == len(lack_chunks), "have and lack must be the same length"
        for have, lack in zip(have_chunks, lack_chunks):
            matchings.append((have, lack))
    else:
        active_chunks = get_active_chunk_sets(gpus, round_id)
        min_active_chunk = min(active_chunks.keys())
        have_chunks = active_chunks[min_active_chunk]
        lack_chunks = []
        for chunk_id in range(min_active_chunk + 1, min(round_id, n-1)):
            if chunk_id in active_chunks:
                lack_chunks.extend(active_chunks[chunk_id])
          # assert that no GPU in have_chunks is in lack_chunks
        assert len(set(have_chunks).intersection(set(lack_chunks))) == 0, "GPU in have_chunks is also in lack_chunks"
        # assert that no GPU in lack_chunks is in have_chunks
        assert len(set(lack_chunks).intersection(set(have_chunks))) == 0, "GPU in lack_chunks is also in have_chunks"
        assert len(have_chunks) == len(lack_chunks), "have and lack must be the same length"
        for have, lack in zip(have_chunks, lack_chunks):
            matchings.append((have, lack))
       
    return matchings

def construct_schedule(gpus, print_result=False):
    # schedule is a list. Each index corresponds to a round. The round contains a list of tuples specifying (GPU_id, chunk_id) for the GPU to be paired with and the chunk idx to send
    schedule = []
    curr_round = 0
    straggler_rank = len(gpus) - 1
    logn = math.ceil(np.log2(len(gpus)))

    while not all(gpu.is_fully_reduced() for gpu in gpus):
        round_schedule = []
        available = np.ones(len(gpus))
        # linear matching with the straggler
        if curr_round < len(gpus) - 1:
            round_schedule.append(StragglerMatching(gpus[curr_round], gpus[straggler_rank], curr_round))
            available[straggler_rank] = 0
            available[curr_round] = 0
        # find matching for the rest of the GPUs
        if curr_round > 0 and curr_round < logn:
            last_picked = curr_round - 1
            round_schedule.append(OneWayMatching(gpus[last_picked], gpus[last_picked + logn], last_picked))
            available[last_picked] = 0
            available[last_picked + logn] = 0
        
        matching = find_matching_opt(gpus, available, curr_round)
        for pair in matching:
            if logn == np.log2(len(gpus)):
                sim = gpus[pair[0]].compute_similarity(gpus[pair[1]])
                if sum(sim) == 1:
                    if sim[0] == 1:
                        round_schedule.append(OneWayMatching(gpus[pair[0]], gpus[pair[1]], gpus[pair[0]].get_highest_chunk()))
                    else:
                        round_schedule.append(OneWayMatching(gpus[pair[1]], gpus[pair[0]], gpus[pair[1]].get_highest_chunk()))
                elif sum(sim) == 2:
                    round_schedule.append(TwoWayMatching(gpus[pair[0]], gpus[pair[1]], gpus[pair[0]].get_highest_chunk(), gpus[pair[1]].get_highest_chunk()))

        valid_round(round_schedule)
        if print_result:
            print("Round", curr_round, flush=True)
            print(len(round_schedule), [str(matching) for matching in round_schedule], flush=True)
            for gpu in gpus:
                print("GPU", gpu.rank, "chunks", gpu.chunks, flush=True)
            print("Active chunks", get_active_chunks(gpus), flush=True)
            print("----------------------------------------------", flush=True)
        curr_round += 1
        schedule.append(round_schedule)

    return schedule

# Test the function with n and straggler rank as command line arguments
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python synthesizer_pow2.py <n>")
        sys.exit(1)

    n = int(sys.argv[1])

    assert n > 0, "n must be positive"

    assert n & (n - 1) == 0, "n must be a power of 2"
    
    gpus = []
    for i in range(n):
        gpus.append(GPU(i, n))

    time.start = time.time()
    schedule = construct_schedule(gpus, print_result=True)
    time.end = time.time()
    print("Time taken:", time.end - time.start)

    print("Actual number of rounds", len(schedule))
    print("Optimal number of rounds", n - 2 + np.log2(n))