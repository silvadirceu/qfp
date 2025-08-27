import pickle
import os
from .audio import load_audio
from .utils import stft, find_peaks, generate_hash, n_strongest
from .quads import find_quads


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

    def create(self, snip=None):
        """
        Creates quad hashes for a given audio file
        """
        q, r, c, w, h = self.params
        # print("pegou params")
        samples = load_audio(self.path, snip=snip)
        # print("carregou audio")
        spectrogram = stft(samples)
        # print("fez o spectograma")
        self.peaks = list(find_peaks(spectrogram, w, h))
        # print("encontrou picos")
        quads = find_quads(self.peaks, r, c)
        # print("encontrou quads")
        self.strongest = n_strongest(spectrogram, quads, q)
        # print("selecionou os mais fortes")
        self.hashes = [generate_hash(q) for q in self.strongest]
        # print("gerou hashes")


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
        
        # Convert namedtuples to simple tuples to avoid pickle issues
        peaks_data = [(peak.x, peak.y) for peak in self.peaks]
        
        # Convert strongest quads to serializable format
        strongest_data = []
        for quad in self.strongest:
            quad_data = {
                'A': (quad.A.x, quad.A.y),
                'C': (quad.C.x, quad.C.y),
                'D': (quad.D.x, quad.D.y),
                'B': (quad.B.x, quad.B.y)
            }
            strongest_data.append(quad_data)
        
        # Prepare data to pickle (all important attributes)
        fingerprint_data = {
            'path': self.path,
            'fp_type': self.fp_type,
            'params': self.params,
            'peaks': peaks_data,
            'strongest': strongest_data,
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
        from collections import namedtuple
        
        with open(pickle_path, 'rb') as f:
            fingerprint_data = pickle.load(f)
        
        # Create new instance
        fingerprint = cls(fingerprint_data['path'])
        
        # Restore basic attributes
        fingerprint.fp_type = fingerprint_data['fp_type']
        fingerprint.params = fingerprint_data['params']
        fingerprint.hashes = fingerprint_data['hashes']
        
        # Recreate namedtuples
        Peak = namedtuple('Peak', ['x', 'y'])
        Quad = namedtuple('Quad', ['A', 'C', 'D', 'B'])
        
        # Restore peaks from tuple data
        fingerprint.peaks = [Peak(x, y) for x, y in fingerprint_data['peaks']]
        
        # Restore strongest quads from dict data
        fingerprint.strongest = []
        for quad_data in fingerprint_data['strongest']:
            quad = Quad(
                A=Peak(quad_data['A'][0], quad_data['A'][1]),
                C=Peak(quad_data['C'][0], quad_data['C'][1]),
                D=Peak(quad_data['D'][0], quad_data['D'][1]),
                B=Peak(quad_data['B'][0], quad_data['B'][1])
            )
            fingerprint.strongest.append(quad)
        
        return fingerprint


class QueryFingerprint(Fingerprint):

    def __init__(self, path):
        self.fp_type = fpType.Query
        Fingerprint.__init__(self, path, fp_type=self.fp_type)

    def create(self):
        Fingerprint.create(self, snip=15)
