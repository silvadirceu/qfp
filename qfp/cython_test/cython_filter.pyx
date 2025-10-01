# cython_filter.pyx
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import cython
import numpy as np
cimport numpy as np

# Declaração das funções de validação como "cdef inline" para eficiência
cdef inline bint _is_pitch_coherent(double qAy, double cAy, double e_tolerance) nogil:
    if cAy == 0:
        return False
    cdef double ratio = qAy / cAy
    cdef double lower_bound = 1.0 / (1.0 + e_tolerance)
    cdef double upper_bound = 1.0 / (1.0 - e_tolerance)
    return lower_bound <= ratio <= upper_bound

cdef inline double _is_x_transform_valid(double qAx, double qBx, double cAx, double cBx, double e_tolerance) nogil:
    cdef double denom = cBx - cAx
    if denom == 0:
        return -1.0
    cdef double sTime = (qBx - qAx) / denom
    cdef double lower_bound = 1.0 / (1.0 + e_tolerance)
    cdef double upper_bound = 1.0 / (1.0 - e_tolerance)
    if lower_bound <= sTime <= upper_bound:
        return sTime
    return -1.0

cdef inline double _is_y_transform_valid(double qAy, double qBy, double cAy, double cBy, double e_tolerance) nogil:
    cdef double denom = cBy - cAy
    if denom == 0:
        return -1.0
    cdef double sFreq = (qBy - qAy) / denom
    cdef double lower_bound = 1.0 / (1.0 + e_tolerance)
    cdef double upper_bound = 1.0 / (1.0 - e_tolerance)
    if lower_bound <= sFreq <= upper_bound:
        return sFreq
    return -1.0

cdef inline bint _is_fine_pitch_coherent(double qAy, double cAy, double sFreq, double threshold=1.8) nogil:
    return abs(qAy - (cAy * sFreq)) <= threshold

def _new_filter_candidates_cy(
    dict fingerprints,
    list border_list,
    list phonogram_code_list,
    long[:] I,
    unsigned long[:] lims,  
    long[:,:] query_strongest,
    double e_tolerance=0.2
):
    cdef dict histogram_dict = {}
    cdef int n_queries = lims.shape[0] - 1
    cdef long[:] border_arr = np.array(border_list, dtype=np.int64)
    cdef list code_list = phonogram_code_list
    
    # Declarar TODAS as variáveis no início da função
    cdef int qi, start, end, i, idx_faiss, interval_idx, local_idx, bin_idx, count
    cdef double qAx, qAy, qBx, qBy, cAx, cAy, cBx, cBy, sTime, sFreq, offset
    cdef long[:] idx_faiss_slice
    cdef np.ndarray[np.int64_t] interval_indices_arr
    cdef long[:] interval_indices_mv
    cdef np.ndarray[np.int64_t, ndim=2] strongest_arr
    cdef dict bin_data
    cdef list offsets_list, transforms_list
    count = 0
    for qi in range(n_queries):
        # Código executável - sem novas declarações cdef aqui
        start = <int>lims[qi]
        end = <int>lims[qi + 1]
        idx_faiss_slice = I[start:end]
        
        if idx_faiss_slice.shape[0] == 0:
            continue
            
        # Esta linha agora deve compilar sem erro
        interval_indices_arr = np.searchsorted(
            np.asarray(border_arr), 
            np.asarray(idx_faiss_slice), 
            side='right'
        ) - 1
        interval_indices_mv = interval_indices_arr
        
        # Valores da query uma única vez
        qAx = <double>query_strongest[qi, 0]
        qAy = <double>query_strongest[qi, 1]
        qBx = <double>query_strongest[qi, 6]
        qBy = <double>query_strongest[qi, 7]
        
        for i in range(idx_faiss_slice.shape[0]):
            count += 1
            idx_faiss = idx_faiss_slice[i]
            
            # Acesso direto via memoryview - sem conversão problemática
            interval_idx = <int>interval_indices_mv[i]
            
            # Debug simplificado
            # print(f"DEBUG: i={i}, interval_idx={interval_idx}, len(code_list)={len(code_list)}")
            
            if interval_idx < 0 or interval_idx >= len(code_list):
                continue
                
            phonogram_code = code_list[interval_idx]
            local_idx = idx_faiss - border_arr[interval_idx]
            
            # Acesso ao dicionário uma única vez por candidato
            if phonogram_code not in fingerprints:
                continue
                
            strongest_arr = fingerprints[phonogram_code]['strongest']
            
            if local_idx < 0 or local_idx >= strongest_arr.shape[0]:
                continue
                
            # Acesso direto aos valores do quad
            cAx = <double>strongest_arr[local_idx, 0]
            cAy = <double>strongest_arr[local_idx, 1]
            cBx = <double>strongest_arr[local_idx, 6]
            cBy = <double>strongest_arr[local_idx, 7]
            
            # Aplica as validações
            if not _is_pitch_coherent(qAy, cAy, e_tolerance):
                continue
                
            sTime = _is_x_transform_valid(qAx, qBx, cAx, cBx, e_tolerance)
            if sTime == -1.0:
                continue
                
            sFreq = _is_y_transform_valid(qAy, qBy, cAy, cBy, e_tolerance)
            if sFreq == -1.0:
                continue
                
            if not _is_fine_pitch_coherent(qAy, cAy, sFreq):
                continue
                
            offset = cAx - (qAx / sTime)
            
            # Cálculo do bin_idx
            bin_idx = <int>(offset / 20.0) * 20

            # Atualiza o histograma
            if phonogram_code not in histogram_dict:
                histogram_dict[phonogram_code] = {}
            
            bin_data = histogram_dict[phonogram_code]
            
            if bin_idx not in bin_data:
                # Cria nova entrada no histograma
                histogram_dict[phonogram_code][bin_idx] = {
                    'offsets': [offset],
                    'transforms': [(sTime, sFreq)],
                    'count': 1
                }
            else:
                # Atualiza entrada existente
                offsets_list = bin_data[bin_idx]['offsets']
                transforms_list = bin_data[bin_idx]['transforms']
                
                offsets_list.append(offset)
                transforms_list.append((sTime, sFreq))
                bin_data[bin_idx]['count'] += 1
    print("total candidates: ", count)      
    return histogram_dict