# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:30:35 2021

@author: rasul

"""
from flask import Flask, render_template, request, redirect, send_file
from os.path import exists
from gtts import gTTS

app = Flask(__name__)
@app.route("/", methods = ['POST', 'GET'])
def home():
    if request.method == "POST":
        language = request.form["text_lang"].lower()
        voice_speaker = request.form["speaker"]
        string = request.form["text"]        
        tts =  gTTS(string, lang=language)
        tts.save('test.mp3')
        return send_file("test.mp3", as_attachment=True) 
    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run()