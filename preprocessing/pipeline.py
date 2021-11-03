import subprocess
import os




class Preprocessing:

    """
    wrapper to convert french text to phonemes

    """

    def __init__(self,text, phoneme_dictionary):
        """

        """
        print("Input:", text)
        self.input_var = "tmp/input.txt"
        self.output_var = "tmp/output.txt"
        self.phoneme_out = "tmp/phoneme_out.txt"
        self.id_to_phoneme = {v:k for k,v in phoneme_dictionary.items()}
        self.text = text
        with open(self.input_var, "w") as f:
            f.write(text)
        self.phoneme_to_id = phoneme_dictionary

    def sequence_to_integer(self):
        """

        """

        with open(self.phoneme_out, 'r') as f:
            list_int = []
            phoneme = f.read().split(' ')
            for phone in phoneme:
                list_int.append(self.phoneme_to_id.get(phone))
            #new_path = file_path.split(".txt")[0]+"_int.txt"
            #new = open(new_path,"w+")
            #new.write(" ".join(list_int))
            #new.close()
            return list_int

    def integer_to_sequence(self, list_integer):
        pass

    def get_sequence(self):
        """

        """

        ## clean the french text if needed be
        ## put text in a temp file and get the output and selete the temp file
        ## run the pearl script
        print("./get_phonemes.pl "+ self.input_var +" texts hts run > "+ self.output_var)
        pipe = subprocess.Popen(["./get_phonemes.pl "+ self.input_var +" texts hts run > "+ self.output_var],stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True )
        output, errors = pipe.communicate()
        pipe = subprocess.Popen(["python3 extract_phonemes.py  --input "+ self.output_var +" --output "+ self.phoneme_out],stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True )
        output, errors = pipe.communicate()

        result = self.sequence_to_integer()

        ## delete input and output file
        os.remove(self.input_var)
        os.remove(self.output_var)
        os.remove(self.phoneme_out)

        return result



        

        ## clean the output and get the sequence of phonemes
        ## convert phoneme sequence to id and return list of ids






import pandas as pd

df = pd.read_csv("phonemes.csv", header=None)
df.columns=["phoneme", "id"]
dictionary = {row["phoneme"]:row["id"] for index, row in df.iterrows()}
p = Preprocessing("salut je vais bien", dictionary)
print("output", p.get_sequence())
