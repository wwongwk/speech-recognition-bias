from flask import Flask,send_from_directory, render_template,request
import audio_handler as ah
import commercial_api as ca
import uuid

app = Flask(__name__, template_folder='static')

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
    return ca.emotion_detect(buffs, str(uuid.uuid4()), 'ogg')
    

if __name__ == '__main__':
    #app.run(debug=False, host='0.0.0.0', threaded=True)
    app.run(ssl_context=('cert.pem', 'key.pem'), debug=True, host='0.0.0.0', threaded=True)
