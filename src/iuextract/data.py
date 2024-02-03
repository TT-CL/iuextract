'''
This modle handles data I/O operations:
* reading from disk,
* string cleanup,
* tokenization,
* 3rd party parsing (spacy)
The data is only read from disk and cleaned up.
IU segmentation is handled by the extract module.
'''

import csv
import json
from spacy.tokens import Doc, Token
from .utils import iu2str, clean_str
# Spacy Token Extension
Token.set_extension("iu_index", default=-1, force=True)

#from pathlib import Path
# dep_parser = CoreNLPDependencyParser(url="http://localhost:9000")
# gen_parser = CoreNLPParser(url="http://localhost:9000")
ACCEPTABLE_MODELS = ["spacy", "corenlp_dep", "corenlp_ps"]


def __read_file(filename, nlp):
    ''' Simple file reader. Output is list of sentences '''
    # read file from disk
    lines = []
    with open(filename) as file:
        # Tokenize sentences
        for row in file.readlines():
            # Skip empty lines and start lines
            cleaned_row = clean_str(row)
            if cleaned_row != "" and cleaned_row != "start":
                lines.append(cleaned_row)
    joined_lines = " ".join(lines)
    sents = [sent.text for sent in nlp(joined_lines).sents]
    return sents


def __read_buffer(fileBuffer, nlp):
    ''' Simple buffer reader. Output is a list of sentences '''
    # read file from disk
    lines = []
    byteString = fileBuffer.read()
    decodedString = byteString.decode('utf-8')
    # Tokenize sentences
    for row in decodedString.splitlines():
        # Skip empty lines and start lines
        cleaned_row = clean_str(row)
        if cleaned_row != "" and cleaned_row != "start":
            lines.append(cleaned_row)
    fileBuffer.close()
    joined_lines = " ".join(lines)
    sents = [sent.text for sent in nlp(joined_lines).sents]
    return sents


def __read_filter(file):
    ''' Simple reader for files containing filters '''
    # read file from disk
    lines = []
    if isinstance(file, str):
        # file is a path
        with open(file) as f:
            # Tokenize sentences
            for row in f.readlines():
                # Skip empty lines and start lines
                cleaned_row = clean_str(row)
                if cleaned_row != "" and cleaned_row != "start":
                    lines.append(cleaned_row.lower())
    else:
        # file is a reader
        s = file.decode("utf-8")
        for row in s.split("\n"):
            # Skip empty lines and start lines
            cleaned_row = clean_str(row)
            if cleaned_row != "" and cleaned_row != "start":
                lines.append(cleaned_row.lower())
    return lines


def __parse_file(sents, nlp, input_models=["spacy"]):
    '''
    File parser
    Accepted models:
    "spacy" : spacy parsers
    "corenlp_dep": Stanford CoreNLP dependency parser
    "corenlp_ps": Stanford CoreNLP ps rule parser
    WARNING: Currently only spacy is supported
    '''
    # filter models with acceptable ones
    models = [model for model in input_models if model in ACCEPTABLE_MODELS]
    # instantiate a dict with each model as a key
    res = {}
    res["raw"] = sents  # add raw file as well
    for model in models:
        res[model] = []

    for sent in sents:
        for model in models:
            parsed_sent = None
            if model == "spacy":
                parsed_sent = nlp(sent.strip())
            '''
            # Only spacy is supported, as it is faster
            elif model == "corenlp_dep":
                parsed_sent, = dep_parser.raw_parse(sent)
            elif model == "corenlp_ps":
                parsed_sent, = gen_parser.raw_parse(sent)
            '''
            res[model].extend(parsed_sent.sents)
    return res


def import_file(f, nlp, models=["spacy"]):
    '''
    Wrapper AIO function to import a file.
    It will read from disk, perform cleanup and parse

    :param f: (str) the file URI
    :param nlp: the spacy nlp model
    :param models: a list of dependency parse models to use. Currently only spacy is supported
    :return: (List[Span]) a list of spacy sents
    '''
    raws = None
    if isinstance(f, str):
        raws = __read_file(f, nlp)
    else:
        raws = __read_buffer(f, nlp)
    file = __parse_file(raws, nlp, models)
    return file


def retrieve_filenames(namefile, folder):
    '''
    This function retrieves all the filenames listed in a support file.
    This is to allow flexibility with the amount of files and avoid uploading
    sensitive data (like filenames) on VCS

    Namefile specification:
    Each line of a namefile should contain the name of a document in a corpus.
    Source texts should be prepended by #.

    Example:
    file1.txt
    file2.txt
    #source.txt

    :param namefile: (str) the URI of the namefile
    :param folder: (str) the corpus directory. The files in the namefiles will be prepended with this directory parameter.
    :return: (List[str], List[str]) a tuple, 1. list of corpus filenames, 2. list of filenames for source texts
    '''
    names = []
    sources = []
    sourceSection = False  # bool flag to check if I am in the source section
    with open(namefile) as file:
        rows = file.readlines()
        for row in rows:
            if row[0] == "*":
                sourceSection = True
            elif row[0] != "#":
                if sourceSection:
                    sources.append(folder + row.strip())
                else:
                    names.append(folder + row.strip())
    # print("filenames")
    # print(names)
    # print(sources)
    return names, sources


def import_all_files(filenames, nlp, models=None):
    '''
    Wrapper function that imports, cleans and parses all the files at once

    :param filenames: (List[str]) a list containing the files URIs
    :param nlp: the spacy nlp model
    :param models: a list of dependency parse models to use. Currently only spacy is supported
    :return: (List[List[Span]]) a list of spacy sents for each document
    '''
    files = []
    for filename in filenames:
        try:
            files.append(import_file(filename, nlp, models))
        except Exception:
            print("{} not found. Skipping...".format(filename))
    return files

def export_csv(table, filename):
    with open(filename, 'w') as writerFile:
        writer = csv.writer(writerFile, dialect="unix")
        for row in table:
            writer.writerow(row)

def export_labeled_ius(text, filename):
    '''
    Function to export the string representation of the IUs in a text

    :param text: (Span) the text to export (must be in spacy format and parsed)
    :param filename: the output file
    '''
    raw_sents = [iu2str(sent, opener="", closer="\n") for sent in text]
    raw = "".join(raw_sents)
    with open(filename, 'w') as file:
        file.write(raw)
    return None

def __prepare_json(text, doc_name, doc_type):
    data = {}
    data['doc_name'] = doc_name
    data['doc_type'] = doc_type
    data['sents'] = []
    disc_labels = []  # list of discontinuous labels
    idx_list = {}
    last_idx = None

    word_index = 0
    max_iu_index = 0
    cur_iu_index = 0

    for sent in text:
        sent_data = {}
        sent_data['words'] = []
        for token in sent:
            if token._.iu_index != last_idx:
                if token._.iu_index in idx_list.keys():
                    disc_labels.append(token._.iu_index)
                    cur_iu_index = idx_list[token._.iu_index]
                else:
                    max_iu_index = max_iu_index + 1
                    cur_iu_index = max_iu_index
                    idx_list[token._.iu_index] = cur_iu_index
                last_idx = token._.iu_index
            word = {
                'text': token.text,
                'word_index': word_index,
                'iu_index': cur_iu_index,
                'iu_label': token._.iu_index,
                'disc': False
            }
            word_index = word_index + 1
            sent_data['words'].append(word)

        data['sents'].append(sent_data)
    for sent in data['sents']:
        for word in sent["words"]:
            if word['iu_label'] in disc_labels:
                word['disc'] = True
    return data


def __prepare_man_seg_json(text, nlp):
    seg = text
    if not isinstance(text, Doc):
        seg = nlp(clean_str(seg))

    word_index = 0
    sent_data = {}
    sent_data['words'] = []
    for token in seg:
        word = {
            'text': token.text,
            'word_index': word_index,
            'iu_index': None,
            'iu_label': 'MAN',
            'disc': False
        }
        word_index = word_index + 1
        sent_data['words'].append(word)
    return sent_data


def __prepare_man_segs_json(segs, doc_name, doc_type, nlp):
    data = {}
    data['doc_name'] = doc_name
    data['doc_type'] = doc_type
    data['sents'] = []

    cur_iu_index = 0
    word_index = 0

    for seg in segs:
        if not isinstance(seg, Doc):
            seg = nlp(clean_str(seg))
        sent_data = {}
        sent_data['words'] = []
        for token in seg:
            word = {
                'text': token.text,
                'word_index': word_index,
                'iu_index': cur_iu_index,
                'iu_label': 'MAN',
                'disc': False
            }
            word_index = word_index + 1
            sent_data['words'].append(word)
        cur_iu_index += 1

        data['sents'].append(sent_data)
    for sent in data['sents']:
        for word in sent["words"]:
            word['disc'] = False
    return data

def export_labeled_json(text, filename, doc_name):
    '''
    Export a parsed document into json format

    :param text: (Span) the document to export (must be in spacy format and parsed)
    :param filename: the output file
    :doc_name: the name of the original document. Pass "source" for source texts
    '''

    doc_type = "Source text"
    if doc_name != "source":
        doc_type = "Summary text"
    data = __prepare_json(text, doc_name, doc_type)
    with open(filename, 'w') as outputfile:
        json.dump(data, outputfile)


### This function assignes IU labels to the sents read by data.py
#UNTESTED
def __read_manual_annotation(filename, nlp):
    raws = None
    if isinstance(filename, str):
        raws = __read_file(filename, nlp)
    else:
        raws = __read_buffer(filename, nlp)
    file = __parse_file(raws, ['spacy'])

    
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