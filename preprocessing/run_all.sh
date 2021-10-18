#!/bin/bash

PATH=./siwis/SiwisFrenchSpeechSynthesisDatabase/text

for PART in part1 part2 part3 part4 part5
do

  for FILE in $PATH/$PART/*.txt
  do

    FILENAME="${FILE##*/}"
    echo $FILENAME

    ./get_phonemes.pl $PATH/$PART/$FILENAME texts hts run > ./perl_outputs/$PART/$FILENAME
    ./extract_phonemes.py --input ./perl_outputs/$PART/$FILENAME --output ./phonemes/$PART/$FILENAME

  done

done
