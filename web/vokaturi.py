# measure_wav.py
# Paul Boersma 2019-06-02
#
# A sample script that uses the OpenVokaturi library to extract the emotions from
# a wav file on disk. The file can contain a mono or stereo recording.
#
# Call syntax:
#   python3 measure_wav.py path_to_sound_file.wav

import sys
import scipy.io.wavfile
import json

sys.path.append("./lib")
import Vokaturi

def init(path):
    print("Loading library...")
    Vokaturi.load(path)
    print("Analyzed by: %s" % Vokaturi.versionAndLicense())

print("Reading sound file...")

def detect(audio, ar=11025):
    print("===== sample rate %d" % ar)
    samples = audio
    buffer_length = len(samples)
    print("   %d samples, %d channels" % (buffer_length, samples.ndim))
    c_buffer = Vokaturi.SampleArrayC(buffer_length)
    if samples.ndim == 1:
        c_buffer[:] = samples[:] / 32768.0  # mono
    else:
        c_buffer[:] = 0.5*(samples[:,0]+0.0+samples[:,1]) / 32768.0  # stereo

    print("Creating VokaturiVoice...")
    voice = Vokaturi.Voice(ar, buffer_length)

    print("Filling VokaturiVoice with samples...")
    voice.fill(buffer_length, c_buffer)

    print("Extracting emotions from VokaturiVoice...")
    quality = Vokaturi.Quality()
    emProb = Vokaturi.EmotionProbabilities()
    voice.extract(quality, emProb)

    if quality.valid:
        result = {}
        result['Neutral'] = emProb.neutrality
        result['Happy'] = emProb.happiness
        result['Sad'] = emProb.sadness
        result['Angry'] = emProb.anger
        result['Fear'] = emProb.fear
        return result
    else:
        return {'msg': 'quality is invalid'}
    voice.destroy()

