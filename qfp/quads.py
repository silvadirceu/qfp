# This Python file uses the following encoding: utf-8
from __future__ import division

from bisect import bisect_left, bisect_right
from collections import namedtuple
from itertools import combinations


# def find_quads(peaks, r, c):
#     """
#     Returns list of valid/strong quads for list of peaks
#     """
#     quads = []
#     for root in peaks:
#         quads += _root_quads(root, peaks, r, c)
#     return quads


# def _root_quads(root, peaks, r, c):
#     """
#     finds valid quads for given root
#     """
#     quads = []
#     filtered = _filter_peaks(root, peaks, r, c)
#     if filtered is None:
#         return []
#     found = _valid_quads(root, filtered)
#     if found is not None:
#         quads += found
#     return quads


# def _filter_peaks(root, peaks, r, c):
#     """
#     returns peaks inside window of Ax + c ± (r / 2)
#     """
#     lastPeak = peaks[-1].x
#     windowStart = root.x + c - (r / 2)
#     if windowStart > lastPeak:
#         return None
#     windowEnd = windowStart + r
#     idx_start = bisect_left(peaks, (windowStart, 0))
#     idx_end = bisect_right(peaks, (windowEnd, 0))
#     filtered = peaks[idx_start:idx_end]
#     if len(filtered) < 3:
#         return None
#     return filtered


# def _valid_quads(root, filtered):
#     """
#     returns list of validated quads for given root (A)
#     """
#     Quad = namedtuple('Quad', ['A', 'C', 'D', 'B'])
#     validQuads = []
#     for comb in combinations(filtered, 3):
#         quad = Quad(root, comb[0], comb[1], comb[2])
#         if _valid_quad(quad):
#             validQuads.append(quad)
#     if len(validQuads) == 0:
#         return None
#     else:
#         return validQuads


# def _valid_quad(q):
#     """
#     Evaluates:

#           Ay < By
#       Ax < Cx <= Dx <= Bx
#       Ay < Cy ,  Dy <= By

#     !! NOTE: assumes combinations are sorted by x value
#     (default behavior of itertools.combinations)
#     """
#     if q.A.y < q.C.y < q.B.y and q.A.y < q.D.y <= q.B.y:
#         return True
#     else:
#         return False

Quad = namedtuple('Quad', ['A', 'C', 'D', 'B'])
def find_quads_stream_v2(peaks, r, c, spec, q_per_partition, partition_len=250):
    """
    Implementação completa e eficiente.
    """
    from bisect import bisect_left, bisect_right
    from heapq import heappush, heappushpop

    heaps = {}
    counter = 0
    lastPeakX = peaks[-1].x if peaks else 0

    for root in peaks:
        windowStart = root.x + c - (r / 2)
        if windowStart > lastPeakX:
            continue
        windowEnd = windowStart + r
        idx_start = bisect_left(peaks, (windowStart, 0))
        idx_end = bisect_right(peaks, (windowEnd, 0))
        filtered = peaks[idx_start:idx_end]
        if len(filtered) < 3:
            continue

        # Precompute arrays for filtered to speed up scans
        xs = [p.x for p in filtered]
        ys = [p.y for p in filtered]
        # find leftmost index where x > root.x (Ax < Cx)
        # if no such index, continue
        import bisect
        left_idx = bisect.bisect_right(xs, root.x - 1)  # first with x > root.x
        if left_idx >= len(filtered):
            continue

        # We'll iterate B over indices b_idx in [left_idx .. len(filtered)-1]
        for b_idx in range(left_idx, len(filtered)):
            B = filtered[b_idx]
            # By must be > Ay (root.y)
            if not (B.y > root.y):
                continue
            # For this B, valid C and D must satisfy:
            # - Ax < Cx <= Bx, Ay < Cy < By
            # - Ax < Dx <= Bx, Ay < Dy <= By
            # So consider candidates between left_idx and b_idx inclusive
            # Build list of indices in [left_idx .. b_idx] that satisfy y constraints for C and D
            # Because y constraints differ slightly between C and D, we can compute two index lists
            c_indices = []
            d_indices = []
            for i in range(left_idx, b_idx + 1):
                yi = ys[i]
                if yi > root.y and yi < B.y:
                    c_indices.append(i)
                if yi > root.y and yi <= B.y:
                    d_indices.append(i)
            if not c_indices or not d_indices:
                continue

            # Now produce pairs (C_idx, D_idx) with C_idx <= D_idx (i.e., x(C) <= x(D))
            # c_indices and d_indices are in ascending x order, so we can produce pairs efficiently:
            # for each c_idx, we can find in d_indices the first position >= c_idx
            # use bisect on d_indices (which contains indices in original filtered space)
            import bisect as _bis
            # create an array for binary-searchable d_indices
            # iterate over c_indices
            for c_idx in c_indices:
                # find first d_pos in d_indices where value >= c_idx
                pos = _bis.bisect_left(d_indices, c_idx)
                if pos >= len(d_indices):
                    continue
                for d_pos in range(pos, len(d_indices)):
                    d_idx = d_indices[d_pos]
                    C = filtered[c_idx]
                    D = filtered[d_idx]
                    # extra safety check: C.x <= D.x <= B.x (holds because indices in range and xs sorted)
                    # Build quad
                    quad = Quad(root, C, D, B)
                    # compute strength using spec (C.x, C.y) and (D.x, D.y)
                    try:
                        strength = spec[C.x][C.y] + spec[D.x][D.y]
                    except Exception:
                        # Caso índice inválido por qualquer razão — pular
                        continue

                    # push into heap for partition
                    part_idx = root.x // partition_len
                    heap = heaps.get(part_idx)
                    entry = (strength, counter, quad)
                    counter += 1
                    if heap is None:
                        # initialize with capacity q_per_partition
                        heaps[part_idx] = [entry]
                    else:
                        # maintain min-heap of up to q_per_partition strongest; smallest strength at root
                        # we want to keep largest strengths; use heappushpop to maintain size
                        if len(heap) < q_per_partition:
                            import heapq
                            heapq.heappush(heap, entry)
                        else:
                            import heapq
                            # if entry stronger than smallest, replace
                            if entry[0] > heap[0][0]:
                                heapq.heapreplace(heap, entry)

    # flatten heaps in order of partition index -> return list of quads (strongest per partition)
    result = []
    for pidx in sorted(heaps.keys()):
        # heaps[pidx] is min-heap; extract entries and sort descending by strength for determinism
        heap = heaps[pidx]
        entries = sorted(heap, key=lambda e: (-e[0], e[1]))
        result.extend([e[2] for e in entries])
    return result