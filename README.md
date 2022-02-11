
# Cross Lingual Speaker Adaptation for TTS Applications


## Software Project by team for the Software Project Course at University of Lorraine.
Members: Sharmila Upadhyaya, Anna Kriukova, Rasul Dent, Claésia Costa

# Abstract

This project is a multilingual TTS system for transferring of voice characteristics between speakers in French and English. We modify grad-TTS and experiment with four different model architectures. The results are evaluated both objectively with four metrics and subjectively using MOS for assessing speaker transformation and speech quality. We provide analysis of the results and discuss possible further directions of research. The system demo is available online on (link).

# Outline

1. [Directory Structure](#directory-structure)
2. [Introduction](#introduction)
3. [Installation](#installation)
4. [Execution](#execution)
5. [Dataset](#dataset)
6. [Contributing](#contributing)
7. [Licence](#licence)


## Directory structure

```

├── articles  # articles that we refered to
│   ├── a.txt
│   ├── Glow-TTS_A Generative Flow for Text-to-Speech via.pdf
│   ├── Grad-TTS_A Diffusion Probabilistic Model for Text-to-Speech.pdf
│   ├── Natural TTS Synthesis by Conditioning Wavenet on MEL Spectrogram Predictions.pdf
│   ├── SynPaFlex-Corpus An Expressive French Audiobooks Corpus Dedicated to Expressive Speech Synthesis.pdf
│   ├── The SIWIS French Speech Synthesis Database – Design and recording of a high quality French database for speech synthesis.pdf
│   ├── The Structure of Louisiana Creoel.pdf
│   ├── TUNDRA A multilingual corpus of found data for TTS research created with light supervision.pdf
│   └── WAVENET_A GENERATIVE MODEL FOR RAW AUDIO.pdf
├── Dockerfile  # docker file
├── documents # all presentations
│   ├── presentation.pdf
│   ├── TTS - Fifth Presentation.pdf
│   ├── TTS - Fourth Presentation.pdf
│   ├── TTS-Second Presentation(1).pdf
│   ├── TTS-Seventh Presentation.pdf
│   ├── TTS - Sixths Presentation.pdf
│   └── TTS - Third Presentation.pdf
├── images #images of result
│   └── data.png
├── modules # some modules for preprocessing
│   ├── phoeneme_dictionary_extraction.py # extraction of french phonemes
│   └── preprocessing.py #preprocessing
├── preprocessing # utils file for preprocessing, only used before training
│   ├── extract_phonemes.py
│   ├── get_phonemes.pl
│   ├── --output
│   ├── phonemes.csv
│   ├── phonemes_integer.ipynb
│   ├── pipeline.py
│   ├── result.txt
│   ├── run_all.sh
│   └── test.txt
├── README.md 
├── report #reports
│   ├── Crosslingual_TTS_Draft.pdf
│   └── overleaf_link.txt
├── resources #example of data used
│   └── filelists
│       ├── final_siwis_test.txt
│       ├── final_siwis_train.txt
│       └── final_siwis_val.txt


```
    
Note: There is a submodule named: MultiSpeaker-TTS-V1 which we do not need to add explicitly as the docker will clone it and run the web app. For further detail you can visit the repo: ![Link](https://github.com/sarmilaupadhyaya/MultiSpeaker-TTS-V1)

## Introduction

In this project, we have introduced the implementation of grad-TTS: Diffusion Probabilistic Model for Text-to-Speech [reference] to generate the audio in particular language, for particular speaker. During inference, the model takes the language, text and speaker as inputs and produces the speech in the specified language, in the voice of the specified speaker. During the experiments, representations for language and speaker were varied and the performance of models are evaluated using both objective and subjective evaluation. Similarly, the text representation is done as phoneme units where we combined the phonemes for French and English language without overlapping. The representation of each phoneme is controlled by the language due to the highly similar alphabets with varying sounds for same alphabets. The main contribution of our work is as follow:


    - Multilingual Text-to-Speech representation for French and English language.
    - Speaker transformation across same or different languages.
    - Evaluating the effects of varying representations of language and speaker.
    - Evaluating the effects of adding speakers on the performance.



## Installation

Note: Since, the project it requires some memory space and RAM, Make sure you have around 3 GB physical disk space, 4 GB RAM and enough space to install the requirements. 

---

The Dockerfile **tts_app/Grad-docker** creates an image of the Grad-TTS system that is compatible that can be run on a GPU node.

- For Grid-5000 users, first install Docker on the node 

```
g5k-setup-nvidia-docker -t".
```

- Then build the image with:

```
docker build .
```


- set up the docker repo name and tag. Image id is created in the previous step. You can see the latest docker with: "docker images"

```
docker tag <image-id> web:tts
```

---


## Execution

To run the webapp, you can run the above docker.

```
docker run web:tts

```

---


## Dataset
![Dataset Distribution](images/data.png)
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Licence
[MIT](https://choosealicense.com/licenses/mit/)
