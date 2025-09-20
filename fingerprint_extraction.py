from qfp import ReferenceFingerprint
from qfp.db import QfpDB
import glob
import os
import time
from qfp import QueryFingerprint
from qfp.db_mem import InMemoryQfpDB
import json
from pathlib import Path


# filenames = glob.glob("/mnt/disk1/BAF/audio/references/*.wav")
# inicio = 388
# start = time.time()
# total = len(filenames[inicio:447])

# for idx, file in enumerate(filenames[inicio:447], start=inicio):
#     filename = os.path.splitext(os.path.basename(file))[0]
#     print(f"[{idx}/{447}] {filename}")
#     fp_r = ReferenceFingerprint(file)
#     fp_r.create("/mnt/disk1/BAF/qfp_features/references")


# end = time.time()     # marca o tempo final
# print(f"Tempo de execução: {end - start:.4f} segundos")



# ---------------QUERY------------------

db = InMemoryQfpDB()

# armazenar vários
start_index = time.time()
db.store_all_pickles_from_directory(r"/mnt/disk1/BAF/qfp_features/references")
end_index = time.time()

query_path = "/mnt/disk1/BAF/audio/queries/query_0001.wav"
query_fp = QueryFingerprint(query_path)
filename = os.path.splitext(os.path.basename(query_path))[0]

extraction_start = time.time()
query_fp.create()  # gera hashes, peaks, strongest
extraction_end = time.time()
search_start = time.time()
matches = db.query(query_fp, vThreshold=0.05, e_radius=0.09)
search_end = time.time()


sorted_matches = sorted(matches, key=lambda m: m.vScore, reverse=True)

indexing_time = end_index - start_index
extraction_time = extraction_end - extraction_start
# faiss_search_time = search_end - search_start  # se você quiser separar FAISS de "busca total"
search_time = search_end - search_start

# montar dicionário final
output = {
    "total_matches": len(sorted_matches),
    "indexing_time": indexing_time,
    "extraction_time": extraction_time,
    # "faiss_search_time": faiss_search_time,
    "search_time": search_time,
    "matches": [m._asdict() for m in sorted_matches]
}
with open(f"{filename}.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4)





# db = InMemoryQfpDB()

# # armazenar vários
# start_index = time.time()
# db.store_all_pickles_from_directory("/mnt/disk1/BAF/qfp_features/references") 
# end_index = time.time()

# # diretório onde estão os áudios de query
# query_dir = Path("/mnt/disk1/BAF/audio/queries")


# # diretório para salvar os resultados
# output_dir = Path("data")
# output_dir.mkdir(parents=True, exist_ok=True)

# # procurar todos arquivos que começam com "query_" e terminar em .wav
# query_files = sorted(query_dir.glob("query_*.wav"))

# for query_path in query_files:
#     filename = query_path.stem  # ex: "query_0002"

#     query_fp = QueryFingerprint(str(query_path))

#     extraction_start = time.time()
#     query_fp.create()  # gera hashes, peaks, strongest
#     extraction_end = time.time()

#     search_start = time.time()
#     matches = db.query(query_fp, vThreshold=0.05, e_radius=0.09)
#     search_end = time.time()

#     sorted_matches = sorted(matches, key=lambda m: m.vScore, reverse=True)

#     indexing_time = end_index - start_index
#     extraction_time = extraction_end - extraction_start
#     search_time = search_end - search_start

#     # montar dicionário final
#     output = {
#         "total_matches": len(sorted_matches),
#         "indexing_time": indexing_time,
#         "extraction_time": extraction_time,
#         "search_time": search_time,
#         "matches": [m._asdict() for m in sorted_matches]
#     }

#     # salvar com o mesmo nome do arquivo de query
#     json_path = output_dir / f"{filename}.json"
#     with open(json_path, "w", encoding="utf-8") as f:
#         json.dump(output, f, indent=4)

#     print(f"Processed {query_path} -> {filename}.json")


# query_path = "/mnt/disk1/BAF/audio/queries/query_0001.wav"
# query_fp = QueryFingerprint(query_path)

# query_fp.create(pickle_dir="/mnt/disk1/BAF/qfp_features/queries")
# print(query_fp.subfingerprints)




def test_pickle_loading(db_path, fingerprints_dir):
    """Testa o carregamento de fingerprints do pickle"""
    print("\n=== Testando carregamento de pickles ===")
    
    db = QfpDB(db_path=db_path)
    
    # Carregar todos os pickles do diretório padrão
    db.store_all_pickles_from_directory(fingerprints_dir)






# test peaks plot
# filename = os.path.splitext(os.path.basename(filenames[0]))[0]
# fp_r = ReferenceFingerprint("/home/luiz/repositories/qfp_original2/qfp/data/cutted/references/1160248.ogg")
# fp_r.create("fingerprints_2")
# print(filename)
# db.store(fp_r, "1160248")





# ------------------- query

# db = QfpDB(db_path="data/cutted_cia_av.db")
# for file in filenames[:10]:
#     fp_q = QueryFingerprint(file)
#     fp_q.create()
#     start = time.time()
#     db.query(fp_q)
#     end = time.time() 
#     print(file, fp_q.matches)
#     print(f"Tempo de execução: {end - start:.4f} segundos")


# file = "/home/luiz/repositories/qfp_original2/qfp/data/cutted/queries/11602482.ogg"
# db = QfpDB(db_path="data/cutted_cia_av_test.db")

# fp_q = QueryFingerprint(file)
# fp_q.create()
# start = time.time()
# db.query(fp_q)
# end = time.time() 
# print(file, fp_q.matches)
# print(f"Tempo de execução: {end - start:.4f} segundos")