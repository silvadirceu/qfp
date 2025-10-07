# query_window_fast.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport cython
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def _process_faiss_results_cy(
    list border_list,
    list phonogram_code_list,
    long[:] I,
    unsigned long[:] lims
):
    cdef Py_ssize_t qi, j, n, start, end
    cdef long idx_value, left, right, mid
    cdef long[:] border_arr = np.array(border_list, dtype=np.int64)
    cdef list code_list = phonogram_code_list
    cdef Py_ssize_t len_border = border_arr.shape[0]
    cdef list result = []

    n = lims.shape[0] - 1

    for qi in range(n):
        start = lims[qi]
        end = lims[qi + 1]

        for j in range(start, end):
            idx_value = I[j]

            # Busca binária manual (side='right')
            left = 0
            right = len_border
            while left < right:
                mid = (left + right) // 2
                if border_arr[mid] <= idx_value:
                    left = mid + 1
                else:
                    right = mid

            # Corrige índices fora do range
            if left == 0:
                continue  # idx menor que o primeiro limite
            elif left > len_border:
                left = len_border

            # Usa left-1 de forma segura
            result.append(code_list[left - 1])

    return result


# @cython.boundscheck(False)
# @cython.wraparound(False)
# def _process_faiss_results_cy(
#     list border_list,
#     list phonogram_code_list,
#     long[:] I,
#     unsigned long[:] lims,  
# ):
#     cdef int n_queries = lims.shape[0] - 1
#     cdef long[:] border_arr = np.array(border_list, dtype=np.int64)
#     cdef list code_list = phonogram_code_list
    
#     # Declarar TODAS as variáveis no início da função
#     cdef int qi, start, end, i, idx_faiss, interval_idx
#     cdef long[:] idx_faiss_slice
#     cdef np.ndarray[np.int64_t] interval_indices_arr
#     cdef long[:] interval_indices_mv
#     cdef list phonogram_list
#     phonogram_list = []
#     for qi in range(n_queries):
#         # Código executável - sem novas declarações cdef aqui
#         start = <int>lims[qi]
#         end = <int>lims[qi + 1]
#         idx_faiss_slice = I[start:end]
        
#         if idx_faiss_slice.shape[0] == 0:
#             continue
            
#         # Esta linha agora deve compilar sem erro
#         interval_indices_arr = np.searchsorted(
#             np.asarray(border_arr), 
#             np.asarray(idx_faiss_slice), 
#             side='right'
#         ) - 1
#         interval_indices_mv = interval_indices_arr
        
#         for i in range(idx_faiss_slice.shape[0]):
#             idx_faiss = idx_faiss_slice[i]
            
#             # Acesso direto via memoryview - sem conversão problemática
#             interval_idx = <int>interval_indices_mv[i]
            
#             if interval_idx < 0 or interval_idx >= len(code_list):
#                 continue
                
#             phonogram_list.append(code_list[interval_idx])
        
#     return phonogram_list