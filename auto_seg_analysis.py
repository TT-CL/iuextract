from src.iuextract.data import *
from src.iuextract.extract import label_ius
from pprint import pprint
from spacy import displacy
from src.iuextract.iu_utils import iu_pprint
from utils.gold import *
from utils.statistics import *

#Import all unsegmented files
work_dir = "./data/sample/"
filenames, sourcenames = retrieve_filenames(
    namefile=work_dir + "names.txt", folder=work_dir+"base/")
filenames.extend(sourcenames)
models = ["spacy"]
files = import_all_files(filenames, models)

# import import goldfiles
gold_dir = work_dir+"gold/"
goldnames, goldsourcenames = retrieve_filenames(
    namefile=work_dir + "names.txt", folder=gold_dir)
goldnames.extend(goldsourcenames)
gold_files = import_all_gold_files(goldnames)
# use list comprehension to generate an iterator
source_files = [file['spacy'] for file in files]
f_idx = 0
collections = []
labelled = []

for file in source_files:
    label_ius(file)
    labelled.append(file)
    assign_gold_labels(file, gold_files[f_idx])
    collections.append(gold_agreement(file))
    f_idx += 1


stats_csv = prepare_stats_csv(collections, filenames)
stats_csv.append(["Average Kappa:", combined_kappa_auto(labelled)])
stats_csv.append(
    ["Average WindowDiff 3:", combined_windowdiff_auto(labelled, 3)])
stats_csv.append(
    ["Average WindowDiff 5:", combined_windowdiff_auto(labelled, 5)])
stats_csv.append(["Average PK 3:", combined_pk_auto(labelled, 3)])
stats_csv.append(["Average PK 5:", combined_pk_auto(labelled, 5)])
stats_csv.extend(data_set_errors(source_files))

out_name = work_dir+"gold_vs_auto.csv"
export_csv(stats_csv, out_name)

displacy_opts = {
    "collapse_punct": False,
    "collapse_phrases": False,
    "compact": False
}
#displacy.serve(parsed_files[5], style="dep", options=displacy_opts)
