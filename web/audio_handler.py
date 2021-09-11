from pydub import AudioSegment
import io

def empathWav(ogg):
    return convertOggToWav(ogg, '1', '11025')

def vokaturiWav(ogg):
    return convertOggToWav(ogg, '2', '44100')

def convertOggToWav(ogg,ac,ar):
    audio = AudioSegment.from_file(ogg)
    buf = io.BytesIO()
    audio.export(buf, format='wav', parameters=["-ac", ac, "-ar", ar])
    return buf

