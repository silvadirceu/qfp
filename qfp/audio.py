from pydub import AudioSegment


def load_audio(path, downsample=True, normalize=False, target_dBFS=-20.0, snip=None):
    """
    Creates array of samples from input audio file
    snip = None -> carrega áudio completo
    snip = 15 -> carrega primeiros 15 segundos  
    snip = (start, end) -> carrega segmento específico em segundos
    """
    audio = AudioSegment.from_file(path)
    
    if downsample:
        if (audio.channels > 1) or (audio.frame_rate != 8000) or (audio.sample_width != 2):
            audio = _downsample(audio)
    
    if normalize and audio.dBFS is not target_dBFS:
        audio = _normalize(audio, target_dBFS)
    
    # Modificação para suportar segmentos específicos
    if snip is not None:
        if isinstance(snip, (int, float)):
            # Comportamento original: primeiros N segundos
            milliseconds = snip * 1000
            audio = audio[:milliseconds]
        elif isinstance(snip, tuple) and len(snip) == 2:
            # Novo comportamento: segmento start-end
            start_ms, end_ms = snip[0] * 1000, snip[1] * 1000
            audio = audio[start_ms:end_ms]
    
    return audio.get_array_of_samples()



def _downsample(audio, numChannels=1, sampleRate=8000, bitDepth=2):
    """
    Returns downsampled AudioSegment
    """
    audio = audio.set_channels(numChannels)
    audio = audio.set_frame_rate(sampleRate)
    audio = audio.set_sample_width(bitDepth)
    return audio


def _normalize(audio, target_dBFS):
    """
    Normalizes loudness of AudioSegment
    """
    change_in_dBFS = target_dBFS - audio.dBFS
    return audio.apply_gain(change_in_dBFS)
