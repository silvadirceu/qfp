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
        
        if os.path.exists(pickle_path):
            print(f"Fingerprint já extraída, não será sobrescrita: {pickle_path}")
            return
        
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
        self.windows = []  # Lista para armazenar dados de cada janela

    def create(self, window_size=15, step_size=5, pickle_dir="query_fingerprints"):
        """
        Cria fingerprints por janelas deslizantes e salva em pickle
        
        Args:
            window_size: Tamanho da janela em segundos (padrão: 15s)
            step_size: Passo entre janelas em segundos (padrão: 5s)
            pickle_dir: Diretório para salvar arquivos pickle
        """
        # Carrega o áudio completo
        samples = load_audio(self.path, snip=None)
        sample_rate = 8000  # Taxa de amostragem fixa do seu código
        total_duration = len(samples) / sample_rate
        
        # Calcula os intervalos das janelas
        windows = self._calculate_windows(total_duration, window_size, step_size)
        
        # Processa cada janela
        for i, (start, end) in enumerate(windows):
            # Extrai o trecho de áudio
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            window_samples = samples[start_sample:end_sample]
            
            # Processa a fingerprint para esta janela
            q, r, c, w, h = self.params
            spectrogram = stft(window_samples)
            peaks = find_peaks(spectrogram, w, h)
            strongest = find_quads_stream_v2(peaks, r, c, spectrogram, q)
            hashes = np.vstack([generate_hash(q) for q in strongest])
            
            # Armazena os dados da janela
            self.windows.append({
                'window_id': i,
                'start_time': start,
                'end_time': end,
                'peaks': peaks,
                'strongest': strongest,
                'hashes': hashes
            })
        
        # Salva em pickle após processar todas as janelas
        self.save_to_pickle(pickle_dir)

    def _calculate_windows(self, total_duration, window_size, step_size):
        """
        Calcula os intervalos de tempo para as janelas deslizantes
        """
        windows = []
        start = 0
        
        while start + window_size <= total_duration:
            end = start + window_size
            windows.append((start, end))
            start += step_size
        
        # Última janela: garante window_size recuando do final se necessário
        if total_duration > window_size:
            last_start = max(0, total_duration - window_size)
            if last_start not in [w[0] for w in windows]:  # Evita duplicata
                windows.append((last_start, total_duration))
        else:
            # Áudio mais curto que window_size: usa áudio completo
            windows.append((0, total_duration))
            
        return windows

    def save_to_pickle(self, pickle_dir="query_fingerprints"):
        """
        Salva o objeto QueryFingerprint completo em arquivo pickle
        """
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)
        
        audio_filename = os.path.splitext(os.path.basename(self.path))[0]
        pickle_filename = f"{audio_filename}_query_fingerprint.pkl"
        pickle_path = os.path.join(pickle_dir, pickle_filename)

        if os.path.exists(pickle_path):
            print(f"Fingerprint já extraída, não será sobrescrita: {pickle_path}")
            return
        
        fingerprint_data = {
            'path': self.path,
            'fp_type': self.fp_type,
            'params': self.params,
            'windows': self.windows
        }
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(fingerprint_data, f)
        
        print(f"Query fingerprint salvo em: {pickle_path}")

    @classmethod
    def load_from_pickle(cls, pickle_path):
        """
        Carrega um QueryFingerprint a partir de arquivo pickle
        """
        with open(pickle_path, 'rb') as f:
            fingerprint_data = pickle.load(f)
        
        fingerprint = cls(fingerprint_data['path'])
        fingerprint.fp_type = fingerprint_data['fp_type']
        fingerprint.params = fingerprint_data['params']
        fingerprint.windows = fingerprint_data['windows']
        
        return fingerprint

    def get_window_hashes(self, window_id):
        """
        Retorna os hashes de uma janela específica
        """
        if window_id < len(self.windows):
            return self.windows[window_id]['hashes']
        return None

    def get_window_count(self):
        """
        Retorna o número total de janelas processadas
        """
        return len(self.windows)
