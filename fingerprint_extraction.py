from qfp import ReferenceFingerprint
from qfp.db import QfpDB
import glob
import os
import time
from qfp import QueryFingerprint



filenames = glob.glob("C:/Users/luizf/github/innovox/references/baf/BAF/audio/references/*.wav")

# db = QfpDB(db_path="baf_data.db")
start = time.time()
for file in filenames[447:]:
    filename = os.path.splitext(os.path.basename(file))[0]
    print(filename)
    fp_r = ReferenceFingerprint(file)
    fp_r.create("C:/Users/luizf/github/innovox/qfp/data/baf_references")
    # db.store(fp_r, filename)  


# filename = os.path.splitext(os.path.basename(filenames[438]))[0]
# fp_r = ReferenceFingerprint(filenames[438])
# fp_r.create()
# print(filename)

end = time.time()     # marca o tempo final
print(f"Tempo de execução: {end - start:.4f} segundos")


# fp_q = QueryFingerprint("kiss_pitched_up.mp3")
# fp_q.create()
# db.query(fp_q)
# print(fp_q.matches)




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