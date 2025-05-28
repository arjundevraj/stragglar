#!/usr/bin/env python3
"""
Pad a buffer so that each GPU's chunk is the smallest multiple of
1 KiB *strictly larger* than the raw chunk size (needed for StragglAR)

Usage
-----
$ python pad_buffers.py <num_gpus>

Example
-------
$ python pad_buffers.py 500160 8
"""

import sys

KIB = 1024

BUFFER_SIZES = [
  262144,        # 256 KiB
  1048576,       #   1 MiB
  4194304,       #   4 MiB
  16777216,      #  16 MiB
  67108864,      #  64 MiB
  134217728,     # 128 MiB
  268435456,     # 256 MiB
  536870912,   # 512 MiB
  1073741824   #   1 GiB
]

def next_multiple_greater(x: int, base: int) -> int:
    return ((x // base) + 1) * base

def main():
    if len(sys.argv) != 2:
        print("Need exactly one argument: <num_gpus>")
        sys.exit(1)

    n = int(sys.argv[1])

    padded_sizes = []

    for buf_bytes in BUFFER_SIZES:
        raw_chunk = buf_bytes // (n -1)

        # Compute candidate padded chunk sizes and pick the smallest
        padded_chunk = next_multiple_greater(raw_chunk, 1 * KIB)
        padded_buf = padded_chunk * (n-1)
        assert padded_buf >= buf_bytes
        padded_sizes.append(padded_buf)

    print("Original buffer sizes: ", " ".join([str(x) for x in BUFFER_SIZES]))
    print("Padded buffer sizes: ", " ".join([str(x) for x in padded_sizes]))

if __name__ == "__main__":
    main()
