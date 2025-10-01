from __future__ import division, print_function
from collections import defaultdict, namedtuple
import numpy as np
import os
import math
import operator
import faiss  # pip install faiss-cpu (ou faiss-gpu)
import time
from qfp.fingerprint import fpType, ReferenceFingerprint
from qfp.cython_test.cython_filter import _new_filter_candidates_cy

try:
    from itertools import izip
except ImportError:
    izip = zip
    xrange = range


def _is_pitch_coherent(qAy: float, cAy, e_tolerance: float) -> bool:
    """Verifica a coerência de pitch (rough pitch coherence)."""
    if cAy == 0:
        return False
    ratio = qAy / cAy
    return (1.0 / (1.0 + e_tolerance)) <= ratio <= (1.0 / (1.0 - e_tolerance))


def _is_x_transform_valid(qAx: float, qBx: float, cAx, cBx, e_tolerance: float):
    """Verifica a transformação no eixo X e retorna sTime se válido."""
    denom = cBx - cAx
    if denom == 0:
        return None
    sTime = (qBx - qAx) / denom
    if not (1.0 / (1.0 + e_tolerance)) <= sTime <= (1.0 / (1.0 - e_tolerance)):
        return None
    return sTime


def _is_y_transform_valid(qAy: float, qBy: float, cAy, cBy, e_tolerance: float):
    """Verifica a transformação no eixo Y e retorna sFreq se válido."""
    denom = cBy - cAy
    if denom == 0:
        return None
    sFreq = (qBy - qAy) / denom
    if not (1.0 / (1.0 + e_tolerance)) <= sFreq <= (1.0 / (1.0 - e_tolerance)):
        return None
    return sFreq


def _is_fine_pitch_coherent(qAy: float, cAy, sFreq: float, threshold: float = 1.8) -> bool:
    """Verifica a coerência fina de pitch."""
    return abs(qAy - (cAy * sFreq)) <= threshold


class InMemoryQfpDB:
    """
    Implementação em memória das estruturas:
     - fidindex -> dict (title -> metadata)
     - peakfile -> numpy arrays concatenados (peaks_x, peaks_y) com offsets por record
     - refrecords -> quads armazenados em numpy arrays (Ax,Ay,Cx,Cy,Dx,Dy,Bx,By) + phonogram_code
     - searchtree -> FAISS index sobre hashes (4-dim float32)
    """

class InMemoryQfpDB:
    """
    Implementação em memória das estruturas:
     - fidindex -> dict (title -> metadata)
     - peakfile -> numpy arrays concatenados (peaks_x, peaks_y) com offsets por record
     - refrecords -> quads armazenados em numpy arrays (Ax,Ay,Cx,Cy,Dx,Dy,Bx,By) + phonogram_code
     - searchtree -> FAISS index sobre hashes (4-dim float32)
    """

    def __init__(self):
        # fingerprint reference database
        # fonogram_code -> {'peaks', 'strongest', 'hashes}
        self.fingerprints = {}
        
        # list of start indices of each fingerprint record in the FAISS index
        self.border_list = [] 

        # list of phonogram_code corresponding to each fingerprint record in the FAISS index
        self.phonogram_code_list = []

        # dict to store the histograms o query filtering process
        self.histogram_dict = {}

        # FAISS index. We keep index_flat for simplicity.
        self.faiss_index = self._create_faiss_index()

        # namedtuples
        mcNames = ['phonogram_code', 'offset', 'num_matches', 'sTime', 'sFreq']
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
        # return faiss.IndexFlatL2(d)
        index = faiss.IndexHNSWFlat(d, 48)  # 32 = grau do grafo
        index.hnsw.efSearch = 1024            # controla tradeoff velocidade/precisão
        index.hnsw.efConstruction = 200     # custo de construção
        return index

    
    # --------------------
    # STORING FUNCTIONS
    # --------------------

    def store(self, fp, phonogram_code):
        """
        Store a ReferenceFingerprint (in memory).
        """
        if fp.fp_type != fpType.Reference:
            raise TypeError("May only store reference fingerprints in db")

        if phonogram_code in self.fingerprints:
            print(f"record already exists: {phonogram_code}")
            return

        self.fingerprints[phonogram_code] = {
            'peaks': fp.peaks,
            'strongest': fp.strongest,
            'hashes': fp.hashes
        }

        n_vectors = self.faiss_index.ntotal
        start = n_vectors

        self.faiss_index.add(fp.hashes)

        self.border_list.append(start)
        self.phonogram_code_list.append(phonogram_code)

        # print(f"Stored record '{title}' with phonogram_code {phonogram_code}, peaks {len(fp.peaks)}, quads {n_quads}")


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

    def _faiss_batch_search(self, query_hashes, radius):
        """
        Executa uma busca em batch no FAISS para todos os query_hashes.

        Args:
            query_hashes: np.array (N, 4) com os hashes da query.
            radius: raio L2 (não quadrado).

        Returns:
            lims, D, I -> saída bruta do faiss.range_search
        """
        qmat = np.ascontiguousarray(query_hashes, dtype=np.float32)
        lims, D, I = self.faiss_index.range_search(qmat, radius * radius)
        return lims, D, I


    def query(self, fp_query, vThreshold=0.5, e_radius=0.1, radius_l2=None):
        if fp_query.fp_type != fpType.Query:
            raise TypeError("May only query db with query fingerprints")

        # preparar query peaks
        fp_query._query_Peaks_sorted = fp_query.peaks[fp_query.peaks[:, 0].argsort()]


        if radius_l2 is None:
            radius = math.sqrt(4) * e_radius
        else:
            radius = float(radius_l2)

        # 1. FAISS batch search
        faiss_search_start = time.time()
        lims, D, I = self._faiss_batch_search(fp_query.hashes, radius)
        faiss_search_end = time.time()
            
        filter_start = time.time()
        self.histogram_dict = self._new_filter_candidates(I, lims, fp_query.strongest)
        filter_end = time.time()
        start_process_histogram = time.time()
        results = self._process_histogram(self.histogram_dict, min_matches=4)
        end_process_histogram = time.time()


        # 4. Validação
        matches_start = time.time()
        matches = []
        for phonogram_code, bins in results.items():
            for avg_offset, match_count, (sTime, sFreq) in bins:
                vScore = self._validate_match(self.MatchCandidate(phonogram_code, avg_offset, match_count, sTime, sFreq), fp_query)
                if vScore >= vThreshold:
                    matches.append(self.Match(phonogram_code, avg_offset, vScore))

        matches_end = time.time()
        
        fp_query.matches = matches

        faiss_search_time = faiss_search_end - faiss_search_start
        filter_time = filter_end - filter_start
        process_histogram_time = end_process_histogram - start_process_histogram
        matches_time = matches_end - matches_start
        return fp_query.matches, faiss_search_time, filter_time, process_histogram_time, matches_time

    # --------------------
    # Helper methods for QFP thecnique
    # --------------------

    def _lookup_quad_by_index(self, idx):
        """
        Given a row index in self.hashes / quads, returns Quad(namedtuple) and phonogram_code
        """
        A = self.Peak(int(self.quad_Ax[idx]), int(self.quad_Ay[idx]))
        C = self.Peak(int(self.quad_Cx[idx]), int(self.quad_Cy[idx]))
        D = self.Peak(int(self.quad_Dx[idx]), int(self.quad_Dy[idx]))
        B = self.Peak(int(self.quad_Bx[idx]), int(self.quad_By[idx]))
        phonogram_code = int(self.quad_phonogram_code[idx])
        return self.Quad(A, C, D, B), phonogram_code

    def _create_histogram(self, 
                          phonogram_code, 
                          offset, 
                          sTime, 
                          sFreq, 
                          histogram_dict, 
                          binwidth=20, 
                          ts=4):
        bin_idx = int(math.floor(offset / binwidth) * binwidth)

        # Se a música ainda não está no histograma, adiciona
        if phonogram_code not in histogram_dict:
            histogram_dict[phonogram_code] = {}
        
        # Se o bin ainda não existe para esta música, cria
        if bin_idx not in histogram_dict[phonogram_code]:
            histogram_dict[phonogram_code][bin_idx] = {
                'offsets': [],      # Lista de offsets brutos
                'transforms': [],   # Lista de transformações (sTime, sFreq)
                'count': 0          # Contador de matches
            }
        
        # Adiciona o candidato ao bin
        histogram_dict[phonogram_code][bin_idx]['offsets'].append(offset)
        histogram_dict[phonogram_code][bin_idx]['transforms'].append((sTime, sFreq))
        histogram_dict[phonogram_code][bin_idx]['count'] += 1
        return histogram_dict


    def _process_histogram(self, histogram_dict, min_matches=4):
        """
        Processa o histograma para encontrar matches válidos, aplicando remoção de outliers.
        Equivalente às funções _scales e _outlier_removal do código original.
        """
        results = {}
        
        for phonogram_code, bins in histogram_dict.items():
            valid_bins = []
            
            for bin_idx, bin_data in bins.items():
                # Só processa bins com mínimo de matches (equivalente ao if len(v) >= 4)
                if bin_data['count'] >= min_matches:
                    # Pega as transformações deste bin
                    transforms = bin_data['transforms']  # Lista de (sTime, sFreq)
                    
                    # APLICA REMOÇÃO DE OUTLIERS 
                    cleaned_transforms = self._remove_outliers(transforms)
                    
                    # Só considera se após limpeza ainda tiver min_matches
                    if len(cleaned_transforms) >= min_matches:
                        # Calcula offset médio (equivalente ao bin_idx * 20 original)
                        offsets = bin_data['offsets']
                        avg_offset = np.mean(offsets)  # Ou poderia ser bin_idx * 20
                        
                        # Calcula médias das transformações
                        avg_sTime, avg_sFreq = np.mean(cleaned_transforms, axis=0)
                        match_count = len(cleaned_transforms)
                        
                        valid_bins.append((avg_offset, match_count, (avg_sTime, avg_sFreq)))
            
            # Ordena por número de matches (decrescente) - equivalente ao sorted original
            valid_bins.sort(key=lambda x: x[1], reverse=True)
            results[phonogram_code] = valid_bins
        
        return results


    def _remove_outliers(self, transforms):
        """
        Versão adaptada da _outlier_removal original para trabalhar com a nova estrutura.
        Remove pontos que estão além de 2 desvios padrão da média.
        """
        if len(transforms) == 0:
            return []
        
        # Converte para array NumPy para cálculos vetorizados
        transforms_arr = np.array(transforms)
        
        # Calcula média e desvio padrão para sTime (coluna 0) e sFreq (coluna 1)
        means = np.mean(transforms_arr, axis=0)
        stds = np.std(transforms_arr, axis=0)
        
        # Filtra pontos dentro de 2 desvios padrão
        filtered = []
        for sTime, sFreq in transforms:
            if (means[0] - 2 * stds[0] <= sTime <= means[0] + 2 * stds[0] and
                means[1] - 2 * stds[1] <= sFreq <= means[1] + 2 * stds[1]):
                filtered.append((sTime, sFreq))
        
        return filtered

    def _lookup_peak_range(self, phonogram_code, offset, e=3750):
        """
        Return peaks for given phonogram_code in range [offset, offset+e]
        """
        if phonogram_code not in self.fingerprints:
            return np.empty((0, 2), dtype=int)

        peaks = self.fingerprints[phonogram_code].get("peaks")
        if peaks is None or len(peaks) == 0:
            return np.empty((0, 2), dtype=int)

        mask = (peaks[:, 0] >= offset) & (peaks[:, 0] <= offset + e)
        return peaks[mask]


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
            rPeak_adj_x = rPeak[0] - mc.offset
            rPeakScaled_x = rPeak_adj_x / mc.sFreq
            rPeakScaled_y = rPeak[1] / mc.sTime
            lBound = bisect_left(xs, (rPeakScaled_x - eX))
            rBound = bisect_right(xs, (rPeakScaled_x + eX))
            for i in range(lBound, rBound):
                if not (rPeakScaled_y - eY <= qPeaks[i][1] <= rPeakScaled_y + eY):
                    continue
                else:
                    validated += 1
        vScore = (float(validated) / len(rPeaks))
        return vScore

    def _validate_match(self, match_candidate, fp_query):
        """
        match_candidate is MatchCandidate(phonogram_code, offset, num_matches, sTime, sFreq)
        """
        rPeaks = self._lookup_peak_range(match_candidate.phonogram_code, match_candidate.offset)
        # Ensure qPeaks sorted stored in fp_query._query_Peaks_sorted
        qPeaks = fp_query._query_Peaks_sorted
        # create a simple object with fields used by _verify_peaks: match_candidate has phonogram_code, offset, sTime, sFreq
        # For compatibility, create a small namedtuple
        M = namedtuple('M', ['phonogram_code', 'offset', 'sTime', 'sFreq'])
        mm = M(match_candidate.phonogram_code, match_candidate.offset, match_candidate.sTime, match_candidate.sFreq)
        vScore = self._verify_peaks(mm, rPeaks, qPeaks)
        return vScore

    def _lookup_record_title(self, phonogram_code):
        # reverse lookup in fidindex
        for title, meta in self.fidindex.items():
            if meta['phonogram_code'] == phonogram_code:
                return title
        return None

    def _new_filter_candidates(self, I, lims, quads):
    # Converter para memoryviews compatíveis
        I_mv = np.asarray(I, dtype=np.int64)
        lims_mv = np.asarray(lims, dtype=np.uint64)
        query_strongest_mv = np.asarray(quads, dtype=np.int64)
        
        return _new_filter_candidates_cy(
            self.fingerprints,
            self.border_list, 
            self.phonogram_code_list,
            I_mv,
            lims_mv,
            query_strongest_mv,
            e_tolerance=0.75
        )
    
    # def _new_filter_candidates(self, fp_query, I, lims):
    #     for qi, (start, end) in enumerate(zip(lims[:-1], lims[1:])):
    #         idx_faiss = I[start:end]
    #         # 1) Encontrar os intervalos de cada índice
    #         interval_idx = np.searchsorted(self.border_list, idx_faiss, side='right') - 1

    #         # 2) Pegar o phonogram_code correspondente a cada índice
    #         # list_phonogram_code_ref = self.phonogram_code_list[interval_idx]
    #         list_phonogram_code_candidate = [self.phonogram_code_list[i] for i in interval_idx]

    #         # 3) Pegar a posição relativa dentro do intervalo
    #         # list_musicid_index_candidate = idx_faiss - self.border_list[interval_idx]
    #         list_phonogram_code_index_candidate = [i - self.border_list[j] for i, j in zip(idx_faiss, interval_idx)]

    #         # list_quads_candidate = self.fingerprints[list_phonogram_code_candidate]['strongest'][list_phonogram_code_index_candidate]
    #         list_quads_candidate = [
    #             self.fingerprints[pc]['strongest'][idx]
    #             for pc, idx in zip(list_phonogram_code_candidate, list_phonogram_code_index_candidate)
    #         ]

    #         # query_quads values to use in filter calculations
    #         qAx = float(fp_query.strongest[qi, 0])
    #         qAy = float(fp_query.strongest[qi, 1])
    #         qBx = float(fp_query.strongest[qi, 6])
    #         qBy = float(fp_query.strongest[qi, 7])
    #         for idx in range(len(list_quads_candidate)): 
    #             cAx = list_quads_candidate[idx][0]
    #             cAy = list_quads_candidate[idx][1]
    #             cBx = list_quads_candidate[idx][6]
    #             cBy = list_quads_candidate[idx][7]

    #             if not _is_pitch_coherent(qAy, cAy, e_tolerance=0.2):
    #                 continue

    #             sTime = _is_x_transform_valid(qAx, qBx, cAx, cBx, e_tolerance=0.2)
    #             if sTime is None:
    #                 continue

    #             sFreq = _is_y_transform_valid(qAy, qBy, cAy, cBy, e_tolerance=0.2)
    #             if sFreq is None:
    #                 continue

    #             if not _is_fine_pitch_coherent(qAy, cAy, sFreq):
    #                 continue

    #             offset = cAx - (qAx / sTime)

    #             self.histogram_dict = self._create_histogram(list_phonogram_code_candidate[idx], offset, sTime, sFreq, self.histogram_dict)
    #     return self.histogram_dict