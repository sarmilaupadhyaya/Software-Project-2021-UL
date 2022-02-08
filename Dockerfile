FROM ubuntu
MAINTAINER Rasul Dent
ENV PYTHONPATH "${PYTHONPATH}:/MultiSpeaker-TTS-V1"
RUN apt-get update && apt install tzdata -y
ENV TZ=Europe/Rome
RUN apt-get install -y git
RUN apt-get install -y python 3.8; apt-get install -y python3-pip
RUN apt-get install libwww-perl -y
RUN git clone https://github.com/sarmilaupadhyaya/MultiSpeaker-TTS-V1.git
WORKDIR MultiSpeaker-TTS-V1
RUN pip3 install -r new_requirements.txt ;cd model/monotonic_align ;python3 setup.py build_ext --inplace; cd  ../..
RUN git submodule init && git submodule update
RUN cd kv_tts/epitran;python3 setup.py install ; cd ../..
RUN pip3 install gdown; pip3 install flask; pip3 install -U flask_cors 
RUN gdown --folder https://drive.google.com/drive/folders/1-GmZUBTSFaK0iOVXqZdbLSfTG7XHg8AK -O models
EXPOSE 80
CMD [ "python3","-u", "app.py" ]
