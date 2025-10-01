import pickle
import os
from .audio import load_audio
from .utils import stft, find_peaks, generate_hash, n_strongest
from .quads import find_quads_stream_v2
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple

# Recreate namedtuples
Peak = namedtuple('Peak', ['x', 'y'])
Quad = namedtuple('Quad', ['A', 'C', 'D', 'B'])

class fpType:
    """
    Parameters for reference/query fingerprint types
    Presented in order (q, r, c, w, h)

    Tuple is used to ensure immutability

    q = quads to create per root point (A)
    r = width of search window
    c = distance from root point to position window
    w = width of max filter
    h = height of max filter

    based on stft hop-size of 32 samples (4ms):
    ref.r =    800ms / 4ms =  200
    ref.c =   1375ms / 4ms = ~345
    que.r =   1300ms / 4ms =  325
    que.c = 1437.5ms / 4ms = ~360

    query filter height/width are calculated as:
    query.w = ref.w / (1 + .2) = 125
    query.h = ref.h * (1 - .2) = 60

    reference width changed from 151 to 150 so that
    result is an int for epsilon of .2 (20% change in speed/tempo)
    """
    #             Q    R    C    W    H
    Reference = (9, 200, 325, 150,  75)
    Query = (500, 345, 360, 125,  60)


class Fingerprint:

    def __init__(self, path, fp_type):
        self.path = path
        if fp_type is not fpType.Reference and fp_type is not fpType.Query:
            raise TypeError(
                "Fingerprint must be of type 'Reference' or 'Query'")
        else:
            self.params = fp_type
            self.params = fp_type
            self.peaks = np.empty((0, 2), dtype=np.int64)
            self.strongest = np.empty((0, 8), dtype=np.int64)
            self.hashes = np.empty((0, 4), dtype=np.float64)

    def create(self, snip=None):
        """
        Creates quad hashes for a given audio file
        """
        q, r, c, w, h = self.params
        samples = load_audio(self.path, snip=snip)
        spectrogram = stft(samples)
        # TODO: refactor find peaks to return a ndarray of peaks
        self.peaks = find_peaks(spectrogram, w, h)
        # self.save_spectrogram_with_peaks(spectrogram, self.peaks)
        # quads = find_quads(self.peaks, r, c)
        # self.strongest = n_strongest(spectrogram, quads, q)
        # TODO: refactor find_quads_stream_v2 to return a ndarray of quads
        self.strongest = find_quads_stream_v2(self.peaks, r, c, spectrogram, q)
        self.hashes = np.vstack([generate_hash(q) for q in self.strongest])

    def save_spectrogram_with_peaks(self, spectrogram, peaks, out_dir="plots"):
        """
        Salva o espectrograma em escala logarítmica com os picos encontrados.
        
        Args:
            spectrogram (ndarray): matriz do espectrograma
            peaks (list): lista de picos detectados
            out_dir (str): diretório para salvar os plots
        """
        # cria diretório se não existir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        audio_filename = os.path.splitext(os.path.basename(self.path))[0]

        # --- Versão limpa ---
        plt.figure(figsize=(12, 6))
        plt.imshow(
            np.transpose(spectrogram),
            origin="lower",
            aspect="auto",
            cmap="magma"
        )
        plt.colorbar(label="Amplitude (dB)")
        plt.xlabel("Tempo (frames STFT)")
        plt.ylabel("Frequência (bins)")
        plt.title("Spectrograma")
        plt.tight_layout()

        out_path_clean = os.path.join(out_dir, f"{audio_filename}_spectrogram.png")
        plt.savefig(out_path_clean, dpi=150)
        plt.close()
        print(f"Spectrograma (sem picos) salvo em: {out_path_clean}")

        # --- Versão com picos ---
        plt.figure(figsize=(12, 6))
        plt.imshow(
            np.transpose(spectrogram),
            origin="lower",
            aspect="auto",
            cmap="magma"
        )
        x_vals = [p.x for p in peaks]
        y_vals = [p.y for p in peaks]
        plt.scatter(x_vals, y_vals, c="cyan", s=10, marker="x", label="Peaks")
        plt.colorbar(label="Amplitude (dB)")
        plt.xlabel("Tempo (frames STFT)")
        plt.ylabel("Frequência (bins)")
        plt.title("Spectrograma com picos detectados")
        plt.legend()
        plt.tight_layout()

        out_path_peaks = os.path.join(out_dir, f"{audio_filename}_spectrogram_peaks.png")
        plt.savefig(out_path_peaks, dpi=150)
        plt.close()
        print(f"Spectrograma (com picos) salvo em: {out_path_peaks}")




class ReferenceFingerprint(Fingerprint):

    def __init__(self, path):
        self.fp_type = fpType.Reference
        Fingerprint.__init__(self, path, fp_type=self.fp_type)

    def create(self, pickle_dir="fingerprints"):
        """
        Creates fingerprints and saves object attributes to pickle file
        
        Args:
            pickle_dir (str): Directory to save pickle files (default: "fingerprints")
        """
        # Create the fingerprints using parent method
        Fingerprint.create(self)
        
        # Save to pickle file
        self.save_to_pickle(pickle_dir)
    
    def save_to_pickle(self, pickle_dir="fingerprints"):
        """
        Saves the fingerprint attributes to a pickle file
        
        Args:
            pickle_dir (str): Directory to save pickle files
        """
        # Create directory if it doesn't exist
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)
        
        # Generate filename from audio file path
        audio_filename = os.path.splitext(os.path.basename(self.path))[0]
        pickle_filename = f"{audio_filename}_fingerprint.pkl"
        pickle_path = os.path.join(pickle_dir, pickle_filename)
        
        # # Convert namedtuples to simple tuples to avoid pickle issues
        # peaks_data = [(peak.x, peak.y) for peak in self.peaks]
        
        # # Convert strongest quads to serializable format
        # strongest_data = []
        # for quad in self.strongest:
        #     quad_data = {
        #         'A': (quad.A.x, quad.A.y),
        #         'C': (quad.C.x, quad.C.y),
        #         'D': (quad.D.x, quad.D.y),
        #         'B': (quad.B.x, quad.B.y)
        #     }
        #     strongest_data.append(quad_data)
        
        # Prepare data to pickle (all important attributes)
        fingerprint_data = {
            'path': self.path,
            'fp_type': self.fp_type,
            'params': self.params,
            'peaks': self.peaks,
            'strongest': self.strongest,
            'hashes': self.hashes
        }
        
        # Save to pickle file
        with open(pickle_path, 'wb') as f:
            pickle.dump(fingerprint_data, f)
        
        print(f"Fingerprint saved to: {pickle_path}")
    
    @classmethod
    def load_from_pickle(cls, pickle_path):
        """
        Recreates a ReferenceFingerprint object from a pickle file
        
        Args:
            pickle_path (str): Path to the pickle file
            
        Returns:
            ReferenceFingerprint: Recreated fingerprint object
        """
        with open(pickle_path, 'rb') as f:
            fingerprint_data = pickle.load(f)
        
        # Create new instance
        fingerprint = cls(fingerprint_data['path'])
        
        fingerprint.fp_type = fingerprint_data['fp_type']
        fingerprint.params = fingerprint_data['params']
        fingerprint.hashes = np.array(fingerprint_data['hashes'], dtype=np.float64)
        
        fingerprint.peaks = np.array([(p[0], p[1]) for p in fingerprint_data['peaks']], dtype=np.int64)
        
        # Restore strongest quads from dict data
        strongest_list = []
        quads = fingerprint_data['strongest']
        for quad in quads:
            strongest_list.append([
                quad['A'][0], quad['A'][1],
                quad['C'][0], quad['C'][1],
                quad['D'][0], quad['D'][1],
                quad['B'][0], quad['B'][1]
            ])
        
        fingerprint.strongest = np.array(strongest_list, dtype=np.int64)

        return fingerprint

    # ONLY FOR LOAD THE EXTRACTED PICKLES FROM OLD VERSION
    @classmethod
    def load_from_pickle2(cls, pickle_path):
        """
        Recreates a ReferenceFingerprint object from a pickle file
        
        Args:
            pickle_path (str): Path to the pickle file
            
        Returns:
            ReferenceFingerprint: Recreated fingerprint object
        """
        with open(pickle_path, 'rb') as f:
            fingerprint_data = pickle.load(f)
        
        # Create new instance
        fingerprint = cls(fingerprint_data['path'])
        
        fingerprint.fp_type = fingerprint_data['fp_type']
        fingerprint.params = fingerprint_data['params']
        fingerprint.hashes = fingerprint_data['hashes']
        
        fingerprint.peaks = fingerprint_data['peaks']
        
        # Restore strongest quads from dict data
        strongest_list = []
        quads = fingerprint_data['strongest']
        for quad in quads:
            strongest_list.append([
                quad['A'][0], quad['A'][1],
                quad['C'][0], quad['C'][1],
                quad['D'][0], quad['D'][1],
                quad['B'][0], quad['B'][1]
            ])
        
        fingerprint.strongest = np.array(strongest_list, dtype=np.int64)

        return fingerprint

class QueryFingerprint(Fingerprint):

    def __init__(self, path):
        self.fp_type = fpType.Query
        Fingerprint.__init__(self, path, fp_type=self.fp_type)

    def create(self):
        Fingerprint.create(self, snip=15)

# class SubFingerprint:
#     """
#     Representa o fingerprint de uma única janela de áudio.
#     """
#     def __init__(self, segment_index, start_time, end_time, peaks, strongest, hashes):
#         self.segment_index = segment_index
#         self.start_time = start_time
#         self.end_time = end_time
#         self.peaks = peaks              
#         self.strongest = strongest      
#         self.hashes = hashes            


# class QueryFingerprint(Fingerprint):

#     def __init__(self, path):
#         self.fp_type = fpType.Query
#         Fingerprint.__init__(self, path, fp_type=self.fp_type)
#         self.subfingerprints = []

#     # ---------------------------
#     # Métodos auxiliares
#     # ---------------------------

#     def _load_full_audio(self):
#         """Carrega o áudio inteiro (sem cortes)."""
#         return load_audio(self.path, downsample=True, normalize=False, snip=None)

#     def _get_segments(self, total_duration, window_size, step_size):
#         """
#         Gera as janelas de tempo para o processamento.
#         Retorna uma lista de (start_time, end_time).
#         """
#         segments = []
#         start_time = 0
#         while start_time + window_size <= total_duration:
#             end_time = start_time + window_size
#             segments.append((start_time, end_time))
#             start_time += step_size
#         return segments

#     def _process_segment(self, samples, start_time, end_time, segment_idx):
#         """
#         Executa o pipeline de fingerprinting para um segmento de áudio.
#         """
#         q, r, c, w, h = self.params

#         spectrogram = stft(samples)

#         peaks = list(find_peaks(spectrogram, w, h))

#         strongest = find_quads_stream_v2(peaks, r, c, spectrogram, q)

#         hashes = [generate_hash(q) for q in strongest]

#         # transforma para formato leve (compatível com pickle)
#         peaks_out = [(p.x, p.y) for p in peaks]
#         strongest_out = [
#             {"A": (quad.A.x, quad.A.y),
#              "C": (quad.C.x, quad.C.y),
#              "D": (quad.D.x, quad.D.y),
#              "B": (quad.B.x, quad.B.y)} for quad in strongest
#         ]

#         return SubFingerprint(
#             segment_index=segment_idx,
#             start_time=start_time,
#             end_time=end_time,
#             peaks=peaks_out,
#             strongest=strongest,
#             hashes=hashes
#         )

#     def _save_to_pickle(self, pickle_dir="queries"):
#         """
#         Salva todos os subfingerprints em um único .pkl
#         """
#         if not os.path.exists(pickle_dir):
#             os.makedirs(pickle_dir)

#         audio_filename = os.path.splitext(os.path.basename(self.path))[0]
#         pickle_filename = f"{audio_filename}_query.pkl"
#         pickle_path = os.path.join(pickle_dir, pickle_filename)

#         fingerprint_data = {
#             "path": self.path,
#             "fp_type": self.fp_type,
#             "params": self.params,
#             "subfingerprints": self.subfingerprints
#         }

#         with open(pickle_path, "wb") as f:
#             pickle.dump(fingerprint_data, f)

#         print(f"Query fingerprints salvos em: {pickle_path}")

#     # ---------------------------
#     # Método principal
#     # ---------------------------

#     def create(self, window_size=15, step_size=5, pickle_dir="queries"):
#         """
#         Divide o áudio em janelas sobrepostas e cria fingerprints para cada trecho.
#         """
#         # carrega áudio inteiro
#         # carrega áudio inteiro
#         full_audio = self._load_full_audio()
#         total_duration = len(full_audio) / 8000.0  # segundos (downsample=8kHz)

#         # gera lista de janelas
#         segments_info = self._get_segments(total_duration, window_size, step_size)

#         # processa cada janela
#         for idx, (start_time, end_time) in enumerate(segments_info):
#             start_sample = int(start_time * 8000)
#             end_sample = int(end_time * 8000)
#             samples = full_audio[start_sample:end_sample]

#             sfp = self._process_segment(samples, start_time, end_time, idx)
#             self.subfingerprints.append(sfp)

#             print(f"Segmento {idx} ({start_time:.1f}s - {end_time:.1f}s) processado.")

#         # salva tudo em pkl
#         # self._save_to_pickle(pickle_dir)


#     # ---------------------------
#     # Método para recarregar do pickle
#     # ---------------------------

#     @classmethod
#     def load_from_pickle(cls, pickle_path):
#         """
#         Recarrega QueryFingerprint a partir de um arquivo pickle.
#         """
#         with open(pickle_path, "rb") as f:
#             fingerprint_data = pickle.load(f)

#         # Criar a instância principal
#         qf = cls(fingerprint_data["path"])
#         qf.fp_type = fingerprint_data["fp_type"]
#         qf.params = fingerprint_data["params"]

#         # Reconstruir subfingerprints
#         qf.subfingerprints = []
#         for seg in fingerprint_data["subfingerprints"]:
#             sfp = SubFingerprint(seg["segment_index"], seg["start_time"], seg["end_time"])
#             # peaks
#             sfp.peaks = [Peak(x, y) for x, y in seg["peaks"]]
#             # strongest quads
#             sfp.strongest = []
#             for qd in seg["strongest"]:
#                 quad = Quad(
#                     A=Peak(qd["A"][0], qd["A"][1]),
#                     B=Peak(qd["B"][0], qd["B"][1]),
#                     C=Peak(qd["C"][0], qd["C"][1]),
#                     D=Peak(qd["D"][0], qd["D"][1])
#                 )
#                 sfp.strongest.append(quad)
#             # hashes
#             sfp.hashes = seg["hashes"]

#             qf.subfingerprints.append(sfp)
#         return qf





# class fpType:
#     """
#     Parameters for reference/query fingerprint types
#     Presented in order (q, r, c, w, h)

#     Tuple is used to ensure immutability

#     q = quads to create per root point (A)
#     r = width of search window
#     c = distance from root point to position window
#     w = width of max filter
#     h = height of max filter

#     based on stft hop-size of 32 samples (4ms):
#     ref.r =    800ms / 4ms =  200
#     ref.c =   1375ms / 4ms = ~345
#     que.r =   1300ms / 4ms =  325
#     que.c = 1437.5ms / 4ms = ~360

#     query filter height/width are calculated as:
#     query.w = ref.w / (1 + .2) = 125
#     query.h = ref.h * (1 - .2) = 60

#     reference width changed from 151 to 150 so that
#     result is an int for epsilon of .2 (20% change in speed/tempo)
#     """
#     #             Q    R    C    W    H
#     Reference = (9, 200, 325, 150,  75)
#     Query = (500, 345, 360, 125,  60)


# class Fingerprint:

#     def __init__(self, path, fp_type):
#         self.path = path
#         if fp_type is not fpType.Reference and fp_type is not fpType.Query:
#             raise TypeError(
#                 "Fingerprint must be of type 'Reference' or 'Query'")
#         else:
#             self.params = fp_type

#     def create(self, snip=None):
#         """
#         Creates quad hashes for a given audio file
#         """
#         q, r, c, w, h = self.params
#         # print("pegou params")
#         samples = load_audio(self.path, snip=snip)
#         # print("carregou audio")
#         spectrogram = stft(samples)
#         # print("fez o spectograma")
#         self.peaks = list(find_peaks(spectrogram, w, h))
#         # self.save_spectrogram_with_peaks(spectrogram, self.peaks)
#         # print("encontrou picos")
#         # quads = find_quads(self.peaks, r, c)
#         # # print("encontrou quads")
#         # self.strongest = n_strongest(spectrogram, quads, q)
#         self.strongest = find_quads_stream_v2(self.peaks, r, c, spectrogram, q)
#         # print("selecionou os mais fortes")
#         self.hashes = [generate_hash(q) for q in self.strongest]
#         # print("gerou hashes")
    
#     def save_spectrogram_with_peaks(self, spectrogram, peaks, out_dir="plots"):
#         """
#         Salva o espectrograma em escala logarítmica com os picos encontrados.
        
#         Args:
#             spectrogram (ndarray): matriz do espectrograma
#             peaks (list): lista de picos detectados
#             out_dir (str): diretório para salvar os plots
#         """
#         # cria diretório se não existir
#         if not os.path.exists(out_dir):
#             os.makedirs(out_dir)

#         audio_filename = os.path.splitext(os.path.basename(self.path))[0]

#         # --- Versão limpa ---
#         plt.figure(figsize=(12, 6))
#         plt.imshow(
#             np.transpose(spectrogram),
#             origin="lower",
#             aspect="auto",
#             cmap="magma"
#         )
#         plt.colorbar(label="Amplitude (dB)")
#         plt.xlabel("Tempo (frames STFT)")
#         plt.ylabel("Frequência (bins)")
#         plt.title("Spectrograma")
#         plt.tight_layout()

#         out_path_clean = os.path.join(out_dir, f"{audio_filename}_spectrogram.png")
#         plt.savefig(out_path_clean, dpi=150)
#         plt.close()
#         print(f"Spectrograma (sem picos) salvo em: {out_path_clean}")

#         # --- Versão com picos ---
#         plt.figure(figsize=(12, 6))
#         plt.imshow(
#             np.transpose(spectrogram),
#             origin="lower",
#             aspect="auto",
#             cmap="magma"
#         )
#         x_vals = [p.x for p in peaks]
#         y_vals = [p.y for p in peaks]
#         plt.scatter(x_vals, y_vals, c="cyan", s=10, marker="x", label="Peaks")
#         plt.colorbar(label="Amplitude (dB)")
#         plt.xlabel("Tempo (frames STFT)")
#         plt.ylabel("Frequência (bins)")
#         plt.title("Spectrograma com picos detectados")
#         plt.legend()
#         plt.tight_layout()

#         out_path_peaks = os.path.join(out_dir, f"{audio_filename}_spectrogram_peaks.png")
#         plt.savefig(out_path_peaks, dpi=150)
#         plt.close()
#         print(f"Spectrograma (com picos) salvo em: {out_path_peaks}")

#     def save_to_pickle(self, pickle_dir="fingerprints"):
#         """
#         Saves the fingerprint attributes to a pickle file
        
#         Args:
#             pickle_dir (str): Directory to save pickle files
#         """
#         # Create directory if it doesn't exist
#         if not os.path.exists(pickle_dir):
#             os.makedirs(pickle_dir)
        
#         # Generate filename from audio file path
#         audio_filename = os.path.splitext(os.path.basename(self.path))[0]
#         pickle_filename = f"{audio_filename}_fingerprint.pkl"
#         pickle_path = os.path.join(pickle_dir, pickle_filename)
        
#         # Convert namedtuples to simple tuples to avoid pickle issues
#         peaks_data = [(peak.x, peak.y) for peak in self.peaks]
        
#         # Convert strongest quads to serializable format
#         strongest_data = []
#         for quad in self.strongest:
#             quad_data = {
#                 'A': (quad.A.x, quad.A.y),
#                 'C': (quad.C.x, quad.C.y),
#                 'D': (quad.D.x, quad.D.y),
#                 'B': (quad.B.x, quad.B.y)
#             }
#             strongest_data.append(quad_data)
        
#         # Prepare data to pickle (all important attributes)
#         fingerprint_data = {
#             'path': self.path,
#             'fp_type': self.fp_type,
#             'params': self.params,
#             'peaks': peaks_data,
#             'strongest': strongest_data,
#             'hashes': self.hashes
#         }
        
#         # Save to pickle file
#         with open(pickle_path, 'wb') as f:
#             pickle.dump(fingerprint_data, f)
        
#         print(f"Fingerprint saved to: {pickle_path}")




# class ReferenceFingerprint(Fingerprint):

#     def __init__(self, path):
#         self.fp_type = fpType.Reference
#         Fingerprint.__init__(self, path, fp_type=self.fp_type)

#     def create(self, pickle_dir="fingerprints"):
#         """
#         Creates fingerprints and saves object attributes to pickle file
        
#         Args:
#             pickle_dir (str): Directory to save pickle files (default: "fingerprints")
#         """
#         # Create the fingerprints using parent method
#         Fingerprint.create(self)
        
#         # Save to pickle file
#         self.save_to_pickle(pickle_dir)
    
    
#     @classmethod
#     def load_from_pickle(cls, pickle_path):
#         """
#         Recreates a ReferenceFingerprint object from a pickle file
        
#         Args:
#             pickle_path (str): Path to the pickle file
            
#         Returns:
#             ReferenceFingerprint: Recreated fingerprint object
#         """
#         from collections import namedtuple
        
#         with open(pickle_path, 'rb') as f:
#             fingerprint_data = pickle.load(f)
        
#         # Create new instance
#         fingerprint = cls(fingerprint_data['path'])
        
#         # Restore basic attributes
#         fingerprint.fp_type = fingerprint_data['fp_type']
#         fingerprint.params = fingerprint_data['params']
#         fingerprint.hashes = fingerprint_data['hashes']
        
#         # Recreate namedtuples
#         Peak = namedtuple('Peak', ['x', 'y'])
#         Quad = namedtuple('Quad', ['A', 'C', 'D', 'B'])
        
#         # Restore peaks from tuple data
#         fingerprint.peaks = [Peak(x, y) for x, y in fingerprint_data['peaks']]
        
#         # Restore strongest quads from dict data
#         fingerprint.strongest = []
#         for quad_data in fingerprint_data['strongest']:
#             quad = Quad(
#                 A=Peak(quad_data['A'][0], quad_data['A'][1]),
#                 C=Peak(quad_data['C'][0], quad_data['C'][1]),
#                 D=Peak(quad_data['D'][0], quad_data['D'][1]),
#                 B=Peak(quad_data['B'][0], quad_data['B'][1])
#             )
#             fingerprint.strongest.append(quad)
        
#         return fingerprint


# class QueryFingerprint(Fingerprint):

#     def __init__(self, path):
#         self.fp_type = fpType.Query
#         Fingerprint.__init__(self, path, fp_type=self.fp_type)

#     def create(self, pickle_dir="fingerprints"):
#         Fingerprint.create(self, snip=15)
#         self.save_to_pickle(pickle_dir)