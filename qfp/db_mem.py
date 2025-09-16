from __future__ import division, print_function
from collections import defaultdict, namedtuple
import numpy as np
import os
import math
import operator
import faiss  # pip install faiss-cpu (ou faiss-gpu)

try:
    from itertools import izip
except ImportError:
    izip = zip
    xrange = range


class InMemoryQfpDB:
    """
    Implementação em memória das estruturas:
     - fidindex -> dict (title -> metadata)
     - peakfile -> numpy arrays concatenados (peaks_x, peaks_y) com offsets por record
     - refrecords -> quads armazenados em numpy arrays (Ax,Ay,Cx,Cy,Dx,Dy,Bx,By) + recordid
     - searchtree -> FAISS index sobre hashes (4-dim float32)
    """

    def __init__(self, faiss_metric='L2'):
        # metadata index: title -> dict {recordid, num_peaks, peak_start, peak_end, num_quads, quad_start, quad_end}
        self.fidindex = {}
        # counters
        self._next_recordid = 1

        # peakfile: 1D arrays for X (time) and Y (freq) (dtype=int32)
        self.peaks_x = np.empty((0,), dtype=np.int32)
        self.peaks_y = np.empty((0,), dtype=np.int32)
        self.peak_offsets = {}  # recordid -> (start, end)

        # refrecords / quads: store quad coordinates as int32 columns
        self.quad_Ax = np.empty((0,), dtype=np.int32)
        self.quad_Ay = np.empty((0,), dtype=np.int32)
        self.quad_Cx = np.empty((0,), dtype=np.int32)
        self.quad_Cy = np.empty((0,), dtype=np.int32)
        self.quad_Dx = np.empty((0,), dtype=np.int32)
        self.quad_Dy = np.empty((0,), dtype=np.int32)
        self.quad_Bx = np.empty((0,), dtype=np.int32)
        self.quad_By = np.empty((0,), dtype=np.int32)
        self.quad_recordid = np.empty((0,), dtype=np.int32)

        # hashes array (float32 Nx4) in same order as quads
        self.hashes = np.empty((0, 4), dtype=np.float32)

        # FAISS index (will be created lazily). We keep index_flat for simplicity.
        self.faiss_index = None
        self.faiss_ids_offset = 0  # corresponds 1-to-1 with row indices in self.hashes

        # namedtuples (compat)
        self.Peak = namedtuple('Peak', ['x', 'y'])
        self.Quad = namedtuple('Quad', ['A', 'C', 'D', 'B'])
        mcNames = ['recordid', 'offset', 'num_matches', 'sTime', 'sFreq']
        self.MatchCandidate = namedtuple('MatchCandidate', mcNames)
        self.Match = namedtuple('Match', ['record', 'offset', 'vScore'])

    # --------------------
    # Utilities: internal
    # --------------------

    def _ensure_faiss_index(self, use_gpu=False):
        """
        Builds a FAISS IndexFlatL2 over self.hashes if not exists.
        For large DBs prefer IVF,PQ or HNSW depending on memory/speed tradeoffs.
        """
        if self.faiss_index is not None:
            return
        d = 4
        # index that stores vectors and supports range_search
        index = faiss.IndexFlatL2(d)
        # convert to float32 contiguous
        if self.hashes.shape[0] > 0:
            index.add(self.hashes)
        self.faiss_index = index

    # --------------------
    # STORING FUNCTIONS
    # --------------------

    def store(self, fp, title):
        """
        Store a ReferenceFingerprint (in memory).
        """
        if fp.fp_type != fpType.Reference:
            raise TypeError("May only store reference fingerprints in db")

        if title in self.fidindex:
            print(f"record already exists: {title}")
            return

        recordid = self._next_recordid
        self._next_recordid += 1

        # 1) store peaks: fp.peaks is list of namedtuples Peak(x,y)
        peaks_arr_x = np.array([int(p.x) for p in fp.peaks], dtype=np.int32)
        peaks_arr_y = np.array([int(p.y) for p in fp.peaks], dtype=np.int32)
        start = len(self.peaks_x)
        self.peaks_x = np.concatenate([self.peaks_x, peaks_arr_x]) if peaks_arr_x.size > 0 else self.peaks_x
        self.peaks_y = np.concatenate([self.peaks_y, peaks_arr_y]) if peaks_arr_y.size > 0 else self.peaks_y
        end = len(self.peaks_x)
        self.peak_offsets[recordid] = (start, end)

        # 2) store quads (strongest) and hashes
        # Each quad is namedtuple with A,C,D,B where each .x,.y may be numpy scalars -> cast to int
        n_quads = len(fp.strongest)
        if n_quads > 0:
            Ax = np.array([int(q.A.x) for q in fp.strongest], dtype=np.int32)
            Ay = np.array([int(q.A.y) for q in fp.strongest], dtype=np.int32)
            Cx = np.array([int(q.C.x) for q in fp.strongest], dtype=np.int32)
            Cy = np.array([int(q.C.y) for q in fp.strongest], dtype=np.int32)
            Dx = np.array([int(q.D.x) for q in fp.strongest], dtype=np.int32)
            Dy = np.array([int(q.D.y) for q in fp.strongest], dtype=np.int32)
            Bx = np.array([int(q.B.x) for q in fp.strongest], dtype=np.int32)
            By = np.array([int(q.B.y) for q in fp.strongest], dtype=np.int32)
            recids = np.full((n_quads,), recordid, dtype=np.int32)

            self.quad_Ax = np.concatenate([self.quad_Ax, Ax]) if Ax.size > 0 else self.quad_Ax
            self.quad_Ay = np.concatenate([self.quad_Ay, Ay]) if Ay.size > 0 else self.quad_Ay
            self.quad_Cx = np.concatenate([self.quad_Cx, Cx]) if Cx.size > 0 else self.quad_Cx
            self.quad_Cy = np.concatenate([self.quad_Cy, Cy]) if Cy.size > 0 else self.quad_Cy
            self.quad_Dx = np.concatenate([self.quad_Dx, Dx]) if Dx.size > 0 else self.quad_Dx
            self.quad_Dy = np.concatenate([self.quad_Dy, Dy]) if Dy.size > 0 else self.quad_Dy
            self.quad_Bx = np.concatenate([self.quad_Bx, Bx]) if Bx.size > 0 else self.quad_Bx
            self.quad_By = np.concatenate([self.quad_By, By]) if By.size > 0 else self.quad_By
            self.quad_recordid = np.concatenate([self.quad_recordid, recids]) if recids.size > 0 else self.quad_recordid

        # 3) store hashes (fp.hashes assumed iterable of 4-tuples or np.array)
        if len(fp.hashes) > 0:
            h = np.array(fp.hashes, dtype=np.float32).reshape(-1, 4)
            self.hashes = np.vstack([self.hashes, h]) if self.hashes.size else h

        # 4) update fidindex metadata
        self.fidindex[title] = {
            'recordid': recordid,
            'title': title,
            'num_peaks': len(fp.peaks),
            'peak_start': start,
            'peak_end': end,
            'num_quads': n_quads,
            'quad_start': len(self.hashes) - n_quads if n_quads > 0 else 0,
            'quad_end': len(self.hashes)
        }

        # 5) invalidate/rebuild faiss (lazy)
        # simplest approach: discard index; build on next query
        if self.faiss_index is not None:
            self.faiss_index = None

        print(f"Stored record '{title}' with recordid {recordid}, peaks {len(fp.peaks)}, quads {n_quads}")

    def store_from_pickle(self, pickle_path, title=None):
        fp = ReferenceFingerprint.load_from_pickle(pickle_path)
        if title is None:
            title = os.path.splitext(os.path.basename(fp.path))[0]
        self.store(fp, title)

    def store_all_pickles_from_directory(self, pickle_dir="fingerprints"):
        if not os.path.exists(pickle_dir):
            print(f"Directory {pickle_dir} does not exist")
            return
        pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith('.pkl') or f.endswith('.pkl') or f.endswith('.pickle') or f.endswith('.pkl')]
        if not pickle_files:
            print(f"No pickle files found in {pickle_dir}")
            return
        print(f"Found {len(pickle_files)} pickle files to process...")
        for pickle_file in pickle_files:
            try:
                self.store_from_pickle(os.path.join(pickle_dir, pickle_file))
            except Exception as e:
                print(f"Error processing {pickle_file}: {e}")

    # --------------------
    # QUERY / SEARCH FLOW
    # --------------------

    def query(self, fp, vThreshold=0.5, e=0.125, radius_l2=None):
        """
        Query the in-memory DB with a QueryFingerprint.

        Args:
            fp: QueryFingerprint (fp.fp_type must be fpType.Query)
            vThreshold: validation threshold for vScore
            e: per-dimension tolerance used in original implementation
            radius_l2: override radius (Euclidean). If None, computed from e as L_inf->L2: 2*e
        """
        if fp.fp_type != fpType.Query:
            raise TypeError("May only query db with query fingerprints")

        # prepare query peaks sorted by x for verification stage
        qPeaks = [(int(p.x), int(p.y)) for p in fp.peaks]
        qPeaks.sort(key=lambda p: p[0])  # sort by time X
        fp._qPeaks_sorted = qPeaks  # attach for use in verify

        # ensure faiss index
        self._ensure_faiss_index()

        if radius_l2 is None:
            # map L_inf epsilon to L2 radius (4 dims): L2_radius = sqrt(d) * e  (worst-case)
            radius = math.sqrt(4) * e
        else:
            radius = float(radius_l2)

        # For each query hash -> range search
        filtered = defaultdict(list)  # recordid -> list of (offset, (sTime,sFreq))
        for qHash, qQuad in zip(fp.hashes, fp.strongest):
            qvec = np.array(qHash, dtype=np.float32).reshape(1, -1)
            # faiss range_search expects squared radius (L2)
            lims, D, I = self.faiss_index.range_search(qvec, radius * radius)
            # I contains indices of neighbors; D squared L2 distances; lims delimit results
            if I.size == 0:
                continue
            # iterate matched indices
            for idx in I:
                # retrieve quad coords and recordid
                cQuad, recordid = self._lookup_quad_by_index(int(idx))
                # Now perform identical checks as original _filter_candidates
                # safe-cast to float to avoid int division
                try:
                    # rough pitch coherence:
                    if not 1 / (1 + e) <= (float(qQuad.A.y) / float(cQuad.A.y)) <= 1 / (1 - e):
                        continue
                    # sTime
                    denom = (cQuad.B.x - cQuad.A.x)
                    if denom == 0:
                        continue
                    sTime = (qQuad.B.x - qQuad.A.x) / denom
                    if not 1 / (1 + e) <= sTime <= 1 / (1 - e):
                        continue
                    denom2 = (cQuad.B.y - cQuad.A.y)
                    if denom2 == 0:
                        continue
                    sFreq = (qQuad.B.y - qQuad.A.y) / denom2
                    if not 1 / (1 + e) <= sFreq <= 1 / (1 - e):
                        continue
                    # fine pitch coherence
                    if not abs(qQuad.A.y - (cQuad.A.y * sFreq)) <= 1.8:
                        continue
                    offset = cQuad.A.x - (qQuad.A.x / sTime)
                    filtered[recordid].append((offset, (sTime, sFreq)))
                except Exception:
                    # numeric issues -> skip candidate
                    continue
        # print(filtered)
        # bin times and produce match candidates
        binned = {k: self._bin_times(v) for k, v in filtered.items()}
        # print("binned: ", binned)
        results = {k: self._scales(v) for k, v in binned.items() if len(v) >= 4}
        # print("results: ", results)
        mc = [self.MatchCandidate(k, a[0], a[1], a[2][0], a[2][1])
              for k, v in results.items() for a in v]
        # print("mc: ", mc)

        # validate each candidate
        matches = []
        for m in mc:
            vScore = self._validate_match(m, fp)
            if vScore >= vThreshold:
                
                title = self._lookup_record_title(m.recordid)
                matches.append(self.Match(title, m.offset, vScore))
        
        fp.match_candidates = mc
        fp.matches = matches
        return fp.matches

    # --------------------
    # Helper methods mirroring original behaviour
    # --------------------

    def _lookup_quad_by_index(self, idx):
        """
        Given a row index in self.hashes / quads, returns Quad(namedtuple) and recordid
        """
        A = self.Peak(int(self.quad_Ax[idx]), int(self.quad_Ay[idx]))
        C = self.Peak(int(self.quad_Cx[idx]), int(self.quad_Cy[idx]))
        D = self.Peak(int(self.quad_Dx[idx]), int(self.quad_Dy[idx]))
        B = self.Peak(int(self.quad_Bx[idx]), int(self.quad_By[idx]))
        recordid = int(self.quad_recordid[idx])
        return self.Quad(A, C, D, B), recordid

    def _bin_times(self, l, binwidth=20, ts=4):
        d = defaultdict(list)
        for offset, (sTime, sFreq) in l:
            binname = int(math.floor(offset / binwidth) * binwidth)
            d[binname].append((sTime, sFreq))
        return {k: v for k, v in d.items() if len(v) >= ts}

    
    def _outlier_removal(self, d):
        means = np.mean(d, axis=0)
        stds = np.std(d, axis=0)
        d = [v for v in d if
             (means[0] - 2 * stds[0] <= v[0] <= means[0] + 2 * stds[0]) and
             (means[1] - 2 * stds[1] <= v[1] <= means[1] + 2 * stds[1])]
        return d

    def _scales(self, d):
        o_rm = {k: self._outlier_removal(v) for k, v in d.items()}
        res = [(i[0], len(i[1]), np.mean(i[1], axis=0))
               for i in o_rm.items() if len(i[1]) >= 4]
        sorted_mc = sorted(res, key=operator.itemgetter(1), reverse=True)
        return sorted_mc

    def _lookup_peak_range(self, recordid, offset, e=3750):
        """
        Return peaks for given recordid in range [offset, offset+e]
        """
        if recordid not in self.peak_offsets:
            return []
        start, end = self.peak_offsets[recordid]
        xs = self.peaks_x[start:end]
        ys = self.peaks_y[start:end]
        # filter by X range
        mask = (xs >= offset) & (xs <= offset + e)
        sel_x = xs[mask]
        sel_y = ys[mask]
        return [self.Peak(int(x), int(y)) for x, y in zip(sel_x, sel_y)]

    def _verify_peaks(self, mc, rPeaks, qPeaks, eX=18, eY=12):
        """
        very similar to original: use bisect on qPeaks (sorted by x)
        qPeaks: list of tuples (x,y)
        rPeaks: list of Peak namedtuples (x,y)
        """
        validated = 0
        if len(rPeaks) == 0:
            return 0.0
        # qPeaks sorted by x
        xs = [p[0] for p in qPeaks]
        from bisect import bisect_left, bisect_right
        for rPeak in rPeaks:
            rPeak_adj_x = rPeak.x - mc.offset
            rPeakScaled_x = rPeak_adj_x / mc.sFreq
            rPeakScaled_y = rPeak.y / mc.sTime
            lBound = bisect_left(xs, (rPeakScaled_x - eX))
            rBound = bisect_right(xs, (rPeakScaled_x + eX))
            for i in range(lBound, rBound):
                if not (rPeakScaled_y - eY <= qPeaks[i][1] <= rPeakScaled_y + eY):
                    continue
                else:
                    validated += 1
        vScore = (float(validated) / len(rPeaks))
        return vScore

    def _validate_match(self, mc, fp):
        """
        mc is MatchCandidate(recordid, offset, num_matches, sTime, sFreq)
        """
        rPeaks = self._lookup_peak_range(mc.recordid, mc.offset)
        # Ensure qPeaks sorted stored in fp._qPeaks_sorted
        qPeaks = fp._qPeaks_sorted
        # create a simple object with fields used by _verify_peaks: mc has recordid, offset, sTime, sFreq
        # For compatibility, create a small namedtuple
        M = namedtuple('M', ['recordid', 'offset', 'sTime', 'sFreq'])
        mm = M(mc.recordid, mc.offset, mc.sTime, mc.sFreq)
        vScore = self._verify_peaks(mm, rPeaks, qPeaks)
        title = self._lookup_record_title(mc.recordid)
        return vScore

    def _lookup_record_title(self, recordid):
        # reverse lookup in fidindex
        for title, meta in self.fidindex.items():
            if meta['recordid'] == recordid:
                return title
        return None