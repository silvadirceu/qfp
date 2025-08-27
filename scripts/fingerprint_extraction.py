from qfp import ReferenceFingerprint
from qfp.db import QfpDB
import glob
import os
import time
from qfp import QueryFingerprint


start = time.time()

filenames = glob.glob("/home/luiz/repositories/qfp/ecad_db/BAF/audio/references/*.wav")

db = QfpDB(db_path="baf_data.db")

for file in filenames[438:]:
    filename = os.path.splitext(os.path.basename(file))[0]
    print(filename)
    fp_r = ReferenceFingerprint(file)
    fp_r.create()
    # db.store(fp_r, filename)
    


# filename = os.path.splitext(os.path.basename(filenames[438]))[0]
# fp_r = ReferenceFingerprint(filenames[438])
# fp_r.create()
# print(filename)

# end = time.time()     # marca o tempo final
# print(f"Tempo de execução: {end - start:.4f} segundos")


# fp_q = QueryFingerprint("kiss_pitched_up.mp3")
# fp_q.create()
# db.query(fp_q)
# print(fp_q.matches)
