# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:30:35 2021

@author: rasul

"""
from flask import Flask, render_template, request, send_file, send_from_directory
import os
from os.path import exists
from gtts import gTTS

app = Flask(__name__)
#avoid using cached audio
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route("/", methods = ['POST', 'GET'])
def home():
    if request.method == "POST":
        language = request.form["text_lang"].lower()
        voice_speaker = request.form["speaker"].lower()
        string = request.form["text"]
        #accent 
        if language == voice_speaker:        
            tts =  gTTS(string, lang=language, tld = "ca")
        else:
            tts = gTTS(string, lang=voice_speaker)
        tts.save('data/test.mp3')
        return render_template("home.html")
    else:
        if exists("test.mp3"):
            os.remove("test.mp3")
        return render_template("home.html")
    
@app.route("/data/test.mp3", methods = ['GET'])
def read():
    return send_from_directory(directory="data", filename="test.mp3", cache_timeout=0)


if __name__ == "__main__":
    app.run()