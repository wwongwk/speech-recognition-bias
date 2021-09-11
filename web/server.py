from flask import Flask,send_from_directory, render_template,request
import audio_handler as ah
import vokaturi as vk
import scipy.io.wavfile as wavfile
import wave
import io
import requests
import os
import base64
import json
import copy

app = Flask(__name__, template_folder='static')

API = {'empath':'https://api.webempath.net/v2/analyzeWav',
        'deepaffect':'https://proxy.api.deepaffects.com/audio/generic/api/v2/sync/recognise_emotion'}

key = {'empath':'-DfigHIMK3a9HBwozQONFbp3DLdLCNtBlrSdFJ5UbhY',
        'deepaffect':'QUV3GNwPB7EPk5TWJQUjospamvusFGK3'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<path:path>')
def videos(path):
    return app.send_static_file(path)

@app.route('/ad', methods=['POST'])
def ad():
    ogg = request.files['audio']
    buffs = ogg.read()
    result = {}
    print('================'+ str(type(ogg))) 
    try:
        # Vokaturi
        sample_ar44100_ac2 = ah.vokaturiWav(io.BytesIO(buffs))
        (ar, vk_sample) = wavfile.read(sample_ar44100_ac2)
        result['vokaturi'] = vk.detect(vk_sample, ar)
    except BaseException as error:
        result['vokaturi'] = 'No Result'
        print('Vokaturi error with message:{}'.format(error))

    try:
        # Empath
        em_sample = ah.empathWav(io.BytesIO(buffs))
        response = requests.post(API['empath'], files={"wav":em_sample}, params={'apikey':key['empath']})
        result['empath'] = json.loads(response.text)
    except BaseException as error:
        result['empath'] = 'No Result'
        print('Empath error with message:{}'.format(error))
   
    try:
        # Deep Affect
        sample_ar44100_ac2 = ah.vokaturiWav(io.BytesIO(buffs))
        req = {'encoding':'WAV', 'sample_rate':'44100','language':'en-US', 'content':str(sample_ar44100_ac2.read())}
        response = requests.post(API['deepaffect'],
        headers={'Content-Type':'application/json'},
        data=json.dumps(req),
        params={'apikey':key['deepaffect']})
        result['deepaffect'] = json.loads(response.text)
    except BaseException as error:
        result['deepaffect'] = 'No Result'
        print('An exception occurred: {}'.format(error))
    return json.dumps(result)

if __name__ == '__main__':
    #app.run(debug=False, host='0.0.0.0', threaded=True)
    app.run(ssl_context=('cert.pem', 'key.pem'), debug=True, host='0.0.0.0', threaded=True)
