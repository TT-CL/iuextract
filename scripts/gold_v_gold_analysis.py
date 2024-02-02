import sys
sys.path.append("..")

from src.iuextract.data import *
from src.iuextract.extract import label_ius
from pprint import pprint
from spacy import displacy
from src.iuextract.iu_utils import iu_pprint
from utils.gold import *
from utils.statistics import *
from copy import deepcopy

#Import all unsegmented files
corpus_name = "L2WS2022/Napping and Learning"
work_dir = "./data/{}/".format(corpus_name)

base_dir = work_dir + "base/"
filenames, sourcenames = retrieve_filenames(
    namefile=work_dir + "names.txt", folder=base_dir)
filenames.extend(sourcenames)
models = ["spacy"]
files = import_all_files(filenames, models)

filesA = deepcopy(files)
filesB = deepcopy(files)

# import import gold labels
filesA_dir = work_dir+"rater1/"
filesB_dir = work_dir+"rater2/"
goldnamesA, goldsourcenames = retrieve_filenames(
    namefile=work_dir + "names.txt", folder=filesA_dir)
goldnamesA.extend(goldsourcenames)
goldnamesB, goldsourcenames = retrieve_filenames(
    namefile=work_dir + "names.txt", folder=filesB_dir)
goldnamesB.extend(goldsourcenames)
gold_filesA = import_all_gold_files(goldnamesA)
gold_filesB = import_all_gold_files(goldnamesB)
# use list comprehension to generate an iterator
source_filesA = [file['spacy'] for file in filesA]
source_filesB = [file['spacy'] for file in filesB]
files_pairs = zip(source_filesA, source_filesB)


f_idx = 0
collections = []

for fA, fB in files_pairs:
    assign_gold_labels(fA, gold_filesA[f_idx])
    assign_gold_labels(fB, gold_filesB[f_idx])
    collections.append(inter_annotator_agreement(fA, fB))
    f_idx += 1



stats_csv = prepare_stats_csv(collections, filenames, skip_auto=True)

out_name = work_dir+"gold_vs_gold.csv"
export_csv(stats_csv, out_name)

displacy_opts = {
    "collapse_punct": False,
    "collapse_phrases": False,
    "compact": False
}
#displacy.serve(parsed_files[5], style="dep", options=displacy_opts)
