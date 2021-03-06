# %%md
# This file handles data operations for gold standard files.
# It is a different file from data.py since I don't want to load gold related
# functions in production
# (I.e.: when segmenting raw text I can't lookup manual annotation)
#
# Gold data is used to label the parsed data structure from data.py
# A custom iu_index field will be added to each token from data.py.
# This is because I want dependency trees to be evaluated on full sentences and
# not single Idea Units.

# %%codecell
#import nltk
from src.iuextract.data import import_file, clean_str
import spacy
from spacy.tokens import Token
nlp = spacy.load("en_core_web_lg")
import re
from src.iuextract.iu_utils import iu_pprint
# %%codecell
# Spacy Token Extension
Token.set_extension("gold_iu_index", default=-1, force=True)

# %% codecell
# functions
###this function splits the initial manual index from the discontinuous IUs ###
def split_index_iu(sent):
    match = re.match("(\d+)\|(.*)",sent)
    if match is not None:
        #print(match[1],match[2])
        return (match[1],match[2])
    return (None, sent)

### this function imports the gold file and returns a matrix
### the first column will have the discontinuous IU index
### the second column will have the spacy formatted idea unit
def import_gold(filename):
    res = []
    with open(filename) as goldFile:
        reader = goldFile.readlines()
        for row in reader:
            cleaned_row = clean_str(row)
            ## skipping empty lines
            if cleaned_row != "" and cleaned_row != "start":
                disc_index, raw_iu = split_index_iu(cleaned_row)
                res.append([disc_index, nlp(raw_iu.strip())])
    return res

### Wrapper function to import all goldfiles at once
def import_all_gold_files(filenames):
    files = []
    for filename in filenames:
        try:
            files.append(import_gold(filename))
        except:
            print("{} not found. Skipping...".format(filename))
    return files

### This function assignes gold IU labels to the sents read by data.py
def assign_gold_labels(sents, ius):
    gold_iu_index = 0 #gold iu index
    disc_dict = {} #dictionary for disc ius indexes
    sent_idx = 0 #raw sents index
    token_idx = 0 #raw token index

    ## DEBUG:
    #set with the sent indexes of Disc Ius
    disc_ius_set = set()

    for disc_index, iu in ius:
        iu_index = None
        if disc_index is not None:
            #we are examining a discontinuous IU
            if disc_index not in disc_dict.keys():
                #if we don't have our index in the dictionary
                disc_dict[disc_index] = gold_iu_index
                #this increases the IU counter only the first time we find
                #a discontinuous IU
                gold_iu_index +=1
            iu_index = disc_dict[disc_index]
        else:
            #If this IU is not a discontinuous IU then we can avoid the
            #dictionary lookup.
            iu_index = gold_iu_index
            gold_iu_index +=1
        # iterate for each word in the gold idea unit
        for word in iu:
            # get the token row from the raw sents data
            token = sents[sent_idx][token_idx]
            if disc_index is not None:
                disc_ius_set.add(sent_idx)
            #check if the two texts are the same
            if word.text == token.text:
                #assign the IU index
                token._.gold_iu_index = iu_index
                #increase indexes for the exploration of the raw data matrix
                token_idx += 1
                if token_idx == len(sents[sent_idx]):
                    sent_idx += 1
                    token_idx = 0
            else:
                print("***SOMETHING REAL BAD HAPPENED***")
                print("Could not match a word from the raw sentences with the respective Idea Unit.")
                print("word: {} iu: {}".format(word.text,token.text))
                print("token.i: {} token.doc: {}".format(word.i, word.doc))
                print("sent_idx: {} token_idx: {}".format(sent_idx,token_idx))
                print("Please remain calm and call an adult")
                print("-----------")
                raise(Exception("gold"))
    return sents, disc_ius_set
