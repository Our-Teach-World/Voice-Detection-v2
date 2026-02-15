import base64
import io
from pydub import AudioSegment
import tempfile
import os

def decode_base64_audio(base64_string: str) -> io.BytesIO:
    """Decodes a base64 string into a bytes buffer."""
    audio_data = base64.b64decode(base64_string)
    return io.BytesIO(audio_data)

def convert_mp3_to_wav(mp3_buffer: io.BytesIO) -> io.BytesIO:
    """Converts MP3 audio buffer to WAV format for processing."""
    try:
        audio = AudioSegment.from_mp3(mp3_buffer)
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        return wav_buffer
    except Exception as e:
        # Fallback: return the buffer as-is, librosa can handle MP3
        mp3_buffer.seek(0)
        return mp3_buffer
