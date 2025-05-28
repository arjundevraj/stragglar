import numpy as np

# === Alpha-beta model parameters ===
a = 3 / 1000  # alpha (ms)
b = 1000 / (450 * 1024 ** 3)  # beta (ms/byte) assuming 450 GiB/s bandwidth
s = 1 * 1024**3  # 1 GiB buffer size in bytes
SAR_cofficients = {192: 208, 96: 108, 48: 57, 24: 27, 12: 15, 6: 7}  # obtained by running synthesizer.py

# === Cost and speedup computation ===
def compute_costs(N):
    stragglar, rhd, ring, direct = [], [], [], []
    for n in N:
        # StragglAR
        # not power of two
        if n in SAR_cofficients:
            c = SAR_cofficients[n]
            stragglar.append(c * a + (c / (n - 1)) * s * b)
        # power of two
        else:
            stragglar.append((n - 2 + np.log2(n)) * a + ((n - 2 + np.log2(n)) / (n - 1)) * s * b)

        # RH/D
        # not power of two, based on 3-2 elimination technique proposed by  Rabenseifner & Traff, 2004
        if n in SAR_cofficients:
            ceil = np.ceil(np.log2(n))
            floor = np.floor(np.log2(n))
            rhd.append((2 * ceil * a) + (2 * s * b) * (1.5 - (1 / (2 ** floor))))
        # power of two
        else:
            rhd.append((2 * np.log2(n)) * a + ((2 * (n - 1)) / n) * s * b)

        # Ring
        ring.append((2 * (n - 1)) * a + ((2 * (n - 1)) / n) * s * b)

        # Direct
        direct.append(np.log2(n) * a + np.log2(n) * s * b)

    return stragglar, rhd, ring, direct

def compute_speedups(N):
    stragglar, rhd, ring, direct = compute_costs(N)
    return (
        [r / s for r, s in zip(ring, stragglar)],
        [r / h for r, h in zip(ring, rhd)],
        [r / d for r, d in zip(ring, direct)],
        [1.0 for _ in ring]
    )

# === Datasets ===
powers_of_two = [4, 8, 16, 32, 64, 128, 256]
inbetween = [6, 12, 24, 48, 96, 192]

# === Compute values ===
sp_stragglar_pow2, sp_rhd_pow2, sp_direct_pow2, sp_ring_pow2 = compute_speedups(powers_of_two)
sp_stragglar_ib, sp_rhd_ib, sp_direct_ib, sp_ring_ib = compute_speedups(inbetween)

print("Speedups for powers of two:", sp_stragglar_pow2, sp_rhd_pow2, sp_direct_pow2, sp_ring_pow2)
print("Speedups for non-power-of-two values:", sp_stragglar_ib, sp_rhd_ib, sp_direct_ib, sp_ring_ib)