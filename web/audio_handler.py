from pydub import AudioSegment
import io

def empathWav(ogg):
    return convertOggToWav(ogg, '1', '11025')

def vokaturiWav(ogg):
    return convertOggToWav(ogg, '2', '44100')

def convertOggToWav(ogg, ac, ar):
    audio = AudioSegment.from_file(ogg)
    return audio_segments_to_bytes(audio, ac, ar)

def readWavFromPath(wav_file_path):
    as_audio = AudioSegment.from_file(wav_file_path)
    return audio_segments_to_bytes(as_audio, '2', '44100')


def audio_segments_to_bytes(as_audio, ac, ar):
    buf = io.BytesIO()
    as_audio.export(buf, format='wav', parameters=['-ac', ac, '-ar', ar])
    return buf

def downSampleWav(wav_buffer):
    as_audio= AudioSegment(wav_buffer.read())
    return audio_segments_to_bytes(as_audio, '1', '11025')

def down_sample_wav_16000(wav_buffer):
    as_audio= AudioSegment(wav_buffer.read())
    return audio_segments_to_bytes(as_audio, '1', '16000') 

