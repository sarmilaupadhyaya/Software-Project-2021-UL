FROM ubuntu
MAINTAINER raz
RUN apt-get update && apt install tzdata -y
ENV TZ=Europe/Rome
RUN apt-get install -y git
RUN apt-get install -y python 3.7 && apt install python3-pip
RUN git clone https://github.com/huawei-noah/CV-Backbones.git
RUN cd  Speech-Backbones/Grad-TTS
RUN pip install -r requirements.txt
RUN cd model/monotonic_align
RUN python setup.py build_ext --inplace
RUN cd../..


