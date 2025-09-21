from __future__ import division, print_function
from collections import defaultdict, namedtuple
import numpy as np
import os
import math
import operator
import faiss  # pip install faiss-cpu (ou faiss-gpu)
import time
from numba import njit, prange
from numba.typed import List as NumbaList
from qfp.fingerprint import fpType, ReferenceFingerprint

try:
    from itertools import izip
except ImportError:
    izip = zip
    xrange = range


# --------------------------
# Núcleo numba (sem usar listas Python)
# --------------------------
@njit(cache=True, debug=True)
def _filter_candidates_core(qQuads_arr, lims, I,
                            quad_Ax, quad_Ay, quad_Bx, quad_By,
                            quad_recordid, e_tolerance):
    # Usar numba.typed.List para append dentro do njit
    recordids = NumbaList()
    offsets = NumbaList()
    sTimes = NumbaList()
    sFreqs = NumbaList()

    n_queries = qQuads_arr.shape[0]

    for qi in range(n_queries):
        # qQuads_arr assumed float64: Ax,Ay,Bx,By,Cx,Cy,Dx,Dy
        qAx = qQuads_arr[qi, 0]
        qAy = qQuads_arr[qi, 1]
        qBx = qQuads_arr[qi, 2]
        qBy = qQuads_arr[qi, 3]

        start = lims[qi]
        end = lims[qi + 1]

        # iterate indices in I[start:end]
        for k in range(start, end):
            idx = I[k]

            # recupera cQuad dos arrays; todos inteiros 
            cAx = quad_Ax[idx]
            cAy = quad_Ay[idx]
            cBx = quad_Bx[idx]
            cBy = quad_By[idx]
            recordid = quad_recordid[idx]

            # Rough pitch coherence:
            #   1/(1+e) <= queAy/canAy <= 1/(1-e)
            if cAy == 0:
                continue
            ratio = qAy / cAy
            if not (1.0 / (1.0 + e_tolerance) <= ratio <= 1.0 / (1.0 - e_tolerance)):
                continue

            # X transformation tolerance check:
            #   sTime = (queBx-queAx)/(canBx-canAx)
            denom = (cBx - cAx)
            if denom == 0:
                continue
            sTime = (qBx - qAx) / denom
            if not (1.0 / (1.0 + e_tolerance) <= sTime <= 1.0 / (1.0 - e_tolerance)):
                continue

            # Y transformation tolerance check:
            #   sFreq = (queBy-queAy)/(canBy-canAy)
            denom2 = (cBy - cAy)
            if denom2 == 0:
                continue
            sFreq = (qBy - qAy) / denom2
            if not (1.0 / (1.0 + e_tolerance) <= sFreq <= 1.0 / (1.0 - e_tolerance)):
                continue

            # Fine pitch coherence:
            #   |queAy-canAy*sFreq| <= eFine
            # Obs: qAy e cAy são floats/integer; operação segura
            if abs(qAy - (cAy * sFreq)) > 1.8:
                continue

            # offset
            offset = cAx - (qAx / sTime)

            # append em typed lists
            recordids.append(recordid)
            offsets.append(offset)
            sTimes.append(sTime)
            sFreqs.append(sFreq)
    return recordids, offsets, sTimes, sFreqs

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


        # FAISS index. We keep index_flat for simplicity.
        self.faiss_index = faiss.IndexFlatL2(4)

        # namedtuples
        self.Peak = namedtuple('Peak', ['x', 'y'])
        self.Quad = namedtuple('Quad', ['A', 'C', 'D', 'B'])
        mcNames = ['recordid', 'offset', 'num_matches', 'sTime', 'sFreq']
        self.MatchCandidate = namedtuple('MatchCandidate', mcNames)
        self.Match = namedtuple('Match', ['record', 'offset', 'vScore'])

    # --------------------
    # Utilities: internal
    # --------------------

    def _create_faiss_index(self, d=4):
        """
        Builds a FAISS IndexFlatL2.
        For large DBs prefer IVF,PQ or HNSW depending on memory/speed tradeoffs.
        """
        self.faiss_index = faiss.IndexFlatL2(d)

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

        # 1) store peaks
        new_peaks_x = np.array([p.x for p in fp.peaks], dtype=np.int32)
        new_peaks_y = np.array([p.y for p in fp.peaks], dtype=np.int32)
        start_peak = len(self.peaks_x)
        self.peaks_x = np.concatenate([self.peaks_x, new_peaks_x])
        self.peaks_y = np.concatenate([self.peaks_y, new_peaks_y])
        end_peak = len(self.peaks_x)
        self.peak_offsets[recordid] = (start_peak, end_peak)

        # 2) store quads
        n_quads = len(fp.strongest)
        
        new_quad_Ax = np.array([q.A.x for q in fp.strongest], dtype=np.int32)
        new_quad_Ay = np.array([q.A.y for q in fp.strongest], dtype=np.int32)
        new_quad_Cx = np.array([q.C.x for q in fp.strongest], dtype=np.int32)
        new_quad_Cy = np.array([q.C.y for q in fp.strongest], dtype=np.int32)
        new_quad_Dx = np.array([q.D.x for q in fp.strongest], dtype=np.int32)
        new_quad_Dy = np.array([q.D.y for q in fp.strongest], dtype=np.int32)
        new_quad_Bx = np.array([q.B.x for q in fp.strongest], dtype=np.int32)
        new_quad_By = np.array([q.B.y for q in fp.strongest], dtype=np.int32)
        new_quad_recordid = np.full(n_quads, recordid, dtype=np.int32)

        self.quad_Ax = np.concatenate([self.quad_Ax, new_quad_Ax])
        self.quad_Ay = np.concatenate([self.quad_Ay, new_quad_Ay])
        self.quad_Cx = np.concatenate([self.quad_Cx, new_quad_Cx])
        self.quad_Cy = np.concatenate([self.quad_Cy, new_quad_Cy])
        self.quad_Dx = np.concatenate([self.quad_Dx, new_quad_Dx])
        self.quad_Dy = np.concatenate([self.quad_Dy, new_quad_Dy])
        self.quad_Bx = np.concatenate([self.quad_Bx, new_quad_Bx])
        self.quad_By = np.concatenate([self.quad_By, new_quad_By])
        self.quad_recordid = np.concatenate([self.quad_recordid, new_quad_recordid])


        # 3) store hashes
        new_hashes = np.array(fp.hashes, dtype=np.float32).reshape(-1, 4)
        self.hashes = np.concatenate([self.hashes, new_hashes])
        self.faiss_index.add(new_hashes)
        start_hash = self.hashes.shape[0] - n_quads


        # 4) update fidindex
        self.fidindex[title] = {
            'recordid': recordid,
            'title': title,
            'num_peaks': len(fp.peaks),
            'peak_start': start_peak,
            'peak_end': end_peak,
            'num_quads': n_quads,
            'quad_start': start_hash,
            'quad_end': start_hash + n_quads
        }

        # print(f"Stored record '{title}' with recordid {recordid}, peaks {len(fp.peaks)}, quads {n_quads}")


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

    def _faiss_batch_search(self, qHashes, radius):
        """
        Executa uma busca em batch no FAISS para todos os qHashes.

        Args:
            qHashes: np.array (N, 4) com os hashes da query.
            radius: raio L2 (não quadrado).

        Returns:
            lims, D, I -> saída bruta do faiss.range_search
        """
        print("qHashes type: ", type(qHashes))
        print("qHashes len: ", len(qHashes), " type: ", type(qHashes[0]))
        # start_ensure = time.time()
        # self._ensure_faiss_index()
        # end_ensure = time.time()
        # print("Faiss index ensure time: ", end_ensure - start_ensure)
        start_numpy = time.time()
        qmat = np.ascontiguousarray(qHashes, dtype=np.float32)
        end_numpy = time.time()
        print("qHashes conversion to numpy time: ", end_numpy - start_numpy)
        lims, D, I = self.faiss_index.range_search(qmat, radius * radius)
        print("Faiss I: ", I)
        print("Faiss I type: ", type(I))
        print("Faiss I shape: ", I.shape)
        print("Faiss lims: ", lims)
        print("Faiss lims type: ", type(lims))
        print("Faiss lims shape: ", lims.shape)
        return lims, D, I


    # ==========================
    # Função wrapper híbrida
    # ==========================
    def filter_candidates_hybrid(self, qHashes, qQuads, lims, I, e_tolerance):
        """
        wrapper: converte qQuads para ndarray, valida dtypes e chama o núcleo numba.
        Retorna filtered dict {recordid: [(offset, (sTime, sFreq)), ...]}
        """

        # 1) converter qQuads (lista de Quad(namedtuple)) -> numpy float64 (n,8)
        n = len(qQuads)
        qQuads_arr = np.zeros((n, 8), dtype=np.float64)
        for i, quad in enumerate(qQuads):
            qQuads_arr[i, 0] = float(quad.A.x)
            qQuads_arr[i, 1] = float(quad.A.y)
            qQuads_arr[i, 2] = float(quad.B.x)
            qQuads_arr[i, 3] = float(quad.B.y)
            qQuads_arr[i, 4] = float(quad.C.x)
            qQuads_arr[i, 5] = float(quad.C.y)
            qQuads_arr[i, 6] = float(quad.D.x)
            qQuads_arr[i, 7] = float(quad.D.y)

        # 2) garantir que lims e I são numpy arrays de inteiros (contíguos)
        lims_arr = np.ascontiguousarray(lims)   # geralmente int64
        I_arr = np.ascontiguousarray(I)         # geralmente int64

        # 3) garantir arrays da classe são contíguos e do dtype adequado
        # quad_* podem ser int32; numba aceita int32/int64 para indexação.
        quad_Ax = np.ascontiguousarray(self.quad_Ax)
        quad_Ay = np.ascontiguousarray(self.quad_Ay)
        quad_Bx = np.ascontiguousarray(self.quad_Bx)
        quad_By = np.ascontiguousarray(self.quad_By)
        quad_recordid = np.ascontiguousarray(self.quad_recordid)

        # 4) chama núcleo numba
        rec_list, off_list, st_list, sf_list = _filter_candidates_core(
            qQuads_arr, lims_arr, I_arr,
            quad_Ax, quad_Ay, quad_Bx, quad_By,
            quad_recordid, float(e_tolerance)
        )

        # 5) converte numba.typed.List para numpy arrays em Python
        # rec_list é um numba.typed.List — iterável como lista normal
        recordids = np.array(list(rec_list), dtype=np.int64)
        offsets = np.array(list(off_list), dtype=np.float64)
        sTimes = np.array(list(st_list), dtype=np.float64)
        sFreqs = np.array(list(sf_list), dtype=np.float64)

        # 6) agrupa no formato original (defaultdict(list))
        filtered = defaultdict(list)
        for rid, off, st, sf in zip(recordids, offsets, sTimes, sFreqs):
            filtered[int(rid)].append((float(off), (float(st), float(sf))))

        return filtered


    def _filter_candidates(self, qHashes, qQuads, lims, I, e_tolerance):
        return self.filter_candidates_hybrid(qHashes, qQuads, lims, I, e_tolerance)


    def query(self, fp, vThreshold=0.5, e_radius=0.1, radius_l2=None):
        if fp.fp_type != fpType.Query:
            raise TypeError("May only query db with query fingerprints")

        # preparar query peaks
        qPeaks = [(int(p.x), int(p.y)) for p in fp.peaks]
        qPeaks.sort(key=lambda p: p[0])
        fp._qPeaks_sorted = qPeaks

        if radius_l2 is None:
            radius = math.sqrt(4) * e_radius
        else:
            radius = float(radius_l2)

        # 1. FAISS batch search
        faiss_search_start = time.time()
        lims, D, I = self._faiss_batch_search(fp.hashes, radius)
        faiss_search_end = time.time()

        # 2. Aplicar filtros nos resultados
        filter_start = time.time()
        filtered = self._filter_candidates(fp.hashes, fp.strongest, lims, I, e_tolerance=0.2)
        filter_end = time.time()

        # 3. Bin times + scales
        bin_start = time.time()
        binned = {k: self._bin_times(v) for k, v in filtered.items()}
        bin_end = time.time()
        results_start = time.time()
        results = {k: self._scales(v) for k, v in binned.items() if len(v) >= 4}
        results_end = time.time()
        matches_start = time.time()
        mc = [self.MatchCandidate(k, a[0], a[1], a[2][0], a[2][1])
              for k, v in results.items() for a in v]

        # 4. Validação
        matches = []
        print("Recordid: ", mc[0].recordid)
        print("Recordid type: ", type(mc[0].recordid))
        for m in mc:
            vScore = self._validate_match(m, fp)
            if vScore >= vThreshold:
                title = self._lookup_record_title(m.recordid)
                matches.append(self.Match(title, m.offset, vScore))
        matches_end = time.time()
        fp.match_candidates = mc
        fp.matches = matches

        print(f"FAISS search time: {faiss_search_end - faiss_search_start:.3f}s")
        print(f"Filtering time: {filter_end - filter_start:.3f}s")
        print(f"Bin times time: {bin_end - bin_start:.3f}s")
        print(f"Results time: {results_end - results_start:.3f}s")
        print(f"Total matches time: {matches_end - matches_start:.3f}s")
        return fp.matches

    # --------------------
    # Helper methods for QFP thecnique
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
        return vScore

    def _lookup_record_title(self, recordid):
        # reverse lookup in fidindex
        for title, meta in self.fidindex.items():
            if meta['recordid'] == recordid:
                return title
        return None