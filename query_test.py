from collections import namedtuple
import time
Quad = namedtuple('Quad', ['A', 'C', 'D', 'B'])
from qfp.fingerprint import QueryFingerprint
from qfp.db_mem import InMemoryQfpDB

db = InMemoryQfpDB()

# armazenar vários
db.store_all_pickles_from_directory(r"C:\Users\luizf\github\innovox\qfp\data\references_av")

qfp = QueryFingerprint("C:/Users/luizf/github/innovox/references/1836962/cutted/queries/11602482.ogg")
# qfp = QueryFingerprint("C:/Users/luizf/github/innovox/references/1836962/audio/Tarefa/1836962/1836962.ogg")

extraction_start = time.time()
qfp.create()  # gera hashes, peaks, strongest
extraction_end = time.time()
search_start = time.time()
print("antes")
matches = db.query(qfp, vThreshold=0.2, e=0.07)
print("depois")
search_end = time.time()
print(f"Tempo de extração: {extraction_end - extraction_start:.4f} segundos")
print(f"Tempo de busca: {search_end - search_start:.4f} segundos")
print("Matches:", matches)