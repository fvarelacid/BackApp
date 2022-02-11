from flask import Flask, render_template, jsonify
# from flask_simple_geoip import SimpleGeoIP
import torch
import numpy as np
from live_asr import LiveWav2Vec2


app = Flask(__name__)

# app.config.update(GEOIPIFY_API_KEY='at_9DFar32mZqkwP09fSKe8119INfEUM')
# simple_geoip = SimpleGeoIP(app)

@app.route('/')
@app.route("/home")

def home():
    return render_template('index.html')

# def test():
#     print("GeoIP")
#     geoip_data = simple_geoip.get_geoip_data()
#     return jsonify(data=geoip_data)


@app.route('/', methods=['POST', 'GET'])

def predict():

    print("Starting ASR")

    asr = LiveWav2Vec2("facebook/wav2vec2-large-960h-lv60-self")

    asr.start()

    help_count = 0

    while True:
        result = asr.get_last_text()[0]
        if result == 1:
            print("Result 1")
            return render_template('index.html', prediction="Voice distress detected!")
        elif result == "help":
            print("Result help")
            help_count += 1
            if help_count == 3:
                return render_template('index.html', prediction="Calling for help detected!")
        elif result != "help":
            print("Result not help")
            help_count = 0

if __name__ == '__main__':
    app.run(port=3000, debug=True)

