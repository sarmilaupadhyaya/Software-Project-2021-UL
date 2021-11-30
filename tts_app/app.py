# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:30:35 2021

@author: rasul

"""
from flask import Flask, render_template, request, send_file, send_from_directory
import os
from os.path import exists
from web_cpu_inf import main as inf

app = Flask(__name__)
#avoid using cached audio
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route("/", methods = ['POST', 'GET'])
def home():
    if request.method == "POST":
        language = request.form["text_lang"].lower()
        voice_speaker = request.form["speaker"].lower()
        string = request.form["text"]
        diffusion = int(request.form["diffusion"])
        #accent
        inf(string, "checkpts/grad-tts.pt", timesteps=diffusion)
        return render_template("home.html")
    else:
        return render_template("home.html")
    
@app.route("/out/sample_0.wav", methods = ['GET'])
def read():
    return send_from_directory(directory="out", filename="sample_0.wav", cache_timeout=0)


if __name__ == "__main__":
    app.run()