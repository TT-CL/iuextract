import sys
sys.path.append("..")

from src.iuextract.data import *
from src.iuextract.extract import label_ius
from pprint import pprint
from spacy import displacy
from src.iuextract.iu_utils import iu2str
from src.iuextract.gold import *
from utils.statistics import *
import spacy

nlp = spacy.load("en_core_web_lg")

# Import all unsegmented files
work_dir = "../data/sample/"
filenames, sourcenames = retrieve_filenames(
    namefile=work_dir + "names.txt",folder=work_dir + "base/")
filenames.extend(sourcenames)
models = ["spacy"]
files = import_all_files(filenames, nlp, models)

# import import goldfiles
goldnames, goldsourcenames = retrieve_filenames(
    namefile=work_dir+"names.txt",folder=work_dir+"gold/")
goldnames.extend(goldsourcenames)
gold_files = import_all_gold_files(goldnames, nlp)

# use list comprehension to generate an iterator
source_files = [file['spacy'] for file in files]
f_idx = 0
collections = []
for file in source_files:
    label_ius(file)
    assign_gold_labels(file,gold_files[f_idx])
    collections.append(gold_agreement(file))
    out_name = work_dir+"auto/"+filenames[f_idx].split("/")[-1]
    export_labeled_ius(file, out_name)
    doc_name = filenames[f_idx].split("/")[-1].split(".")[0]
    json_name = work_dir+"auto/"+doc_name+".json"
    export_labeled_json(file, json_name, doc_name)
    print("json_name",json_name)
    f_idx += 1

out_name = work_dir+"stats.csv"
export_csv(prepare_stats_csv(collections,filenames), out_name)

displacy_opts = {
    "collapse_punct": False,
    "collapse_phrases": False,
    "compact": False
    }
#displacy.serve(parsed_files[5], style="dep", options=displacy_opts)
