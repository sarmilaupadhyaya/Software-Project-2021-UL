FROM ubuntu
MAINTAINER raz
RUN apt-get update && apt install tzdata -y
ENV TZ=Europe/Rome
RUN apt-get install -y git
RUN apt-get install -y python 3.7 && apt-get install -y python3-pip
RUN git clone https://github.com/huawei-noah/Speech-Backbones.git
RUN cd  Speech-Backbones/Grad-TTS; pip3 install -r requirements.txt ;cd model/monotonic_align ;python3 setup.py build_ext --inplace


