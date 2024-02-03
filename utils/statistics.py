from sklearn.metrics import cohen_kappa_score as kappa
from sklearn.metrics import precision_recall_fscore_support
from nltk.metrics import pk, windowdiff
from nltk.metrics.scores import accuracy
import krippendorff
from src.iuextract.utils import gen_iu_collection
from pprint import pprint
from numpy import sum as np_sum
from numpy import average as np_avg
from numpy import var as np_var
from numpy import float64 as np_float64
from numpy import full as np_full
from numpy import random as np_random
#from numpy import string as np_string

rule_labels = ["R1","R2", "R3", "R3.1", "R3.2","R5","R6.2","R8","UNL", "SEG_BOT"]

## Function that computes agreement statistics between a document
#  and its gold annotation
def gold_agreement(doc):
    ius, disc_ius = gen_iu_collection(doc)
    gold_ius, gold_disc_ius = gen_iu_collection(doc, gold=True)
    dict_1 = iu_agreement(ius, disc_ius, gold_ius, gold_disc_ius)

    gold_bin = ius_to_binary(doc, gold=True)
    extraction_bin = ius_to_binary(doc, gold=False)
    dict_2 = binary_agreement(extraction_bin, gold_bin)
    dict_3 = file_stats_w_errors(doc)
    combined_dicts = {**dict_1, **dict_2, **dict_3}
    return combined_dicts

## Function that computes agreement statistics between two gold documents
def inter_annotator_agreement(doc1,doc2):
    gold_ius1, gold_disc_ius1 = gen_iu_collection(doc1, gold=True)
    gold_ius2, gold_disc_ius2 = gen_iu_collection(doc2, gold=True)
    dict_1 = iu_agreement(gold_ius1, gold_disc_ius1, gold_ius2, gold_disc_ius2)

    gold_bin1 = ius_to_binary(doc1, gold=True)
    gold_bin2 = ius_to_binary(doc2, gold=True)
    dict_2 =binary_agreement(gold_bin1, gold_bin2)
    combined_dicts = {**dict_1, **dict_2}
    return combined_dicts


def doc_stats(doc):
    ius, disc_ius = gen_iu_collection(doc)
    bin_ius = ius_to_binary(doc, gold=False)
    iu_lengths = [len(iu) for iu in ius.values()]
    res = {}
    res["iu_spans"] = sum(bin_ius)+1
    res["num_iu"] = len(ius)
    res["num_disc_iu"] = len(disc_ius)
    res["avg_len_iu"] = np_avg(iu_lengths)
    res["var_len_iu"] = np_var(iu_lengths)
    return res

#this function concatenates binaries for different files but
#  adds 1s between them
def concat_files_bin(bin_list): 
    return sum([sum([[1],el],[]) for el in bin_list],[])[1:]

## Function that generates the combined kappa for all the documents between gold standards
def combined_kappa_gold(files1,files2):
    bins1 = [ius_to_binary(f, gold=True) for f in files1]
    bins2 = [ius_to_binary(f, gold=True) for f in files2]
    concat1 = sum(bins1,[])
    concat2 = sum(bins2,[])
    return kappa(concat1,concat2)

def combined_windowdiff_gold(files1, files2, window):
    bins1 = [ius_to_binary(f, gold=True) for f in files1]
    bins2 = [ius_to_binary(f, gold=True) for f in files2]
    concat1 = sum(bins1,[])
    concat2 = sum(bins2,[])
    str_1 = bin_to_string(concat1)
    str_2 = bin_to_string(concat2)
    return windowdiff(str_1,str_2,window)

def combined_pk_gold(files1, files2, window):
    bins1 = [ius_to_binary(f, gold=True) for f in files1]
    bins2 = [ius_to_binary(f, gold=True) for f in files2]
    concat1 = sum(bins1,[])
    concat2 = sum(bins2,[])
    str_1 = bin_to_string(concat1)
    str_2 = bin_to_string(concat2)
    return pk(ref=str_1,hyp=str_2,k=window,boundary='1')


def combined_pre_rec_f1_gold(files1, files2):
    bins1 = [ius_to_binary(f, gold=True) for f in files1]
    bins2 = [ius_to_binary(f, gold=True) for f in files2]
    concat1 = sum(bins1, [])
    concat2 = sum(bins2, [])
    return precision_recall_f1(concat1, concat2)

## Function that generates the combined kappa for all the documents between a gold standard and the automatic labelling
def combined_kappa_auto(files):
    bins1 = [ius_to_binary(f, gold=False) for f in files]
    bins2 = [ius_to_binary(f, gold=True) for f in files]
    concat1 = sum(bins1,[])
    concat2 = sum(bins2,[])
    return kappa(concat1,concat2)

def combined_windowdiff_auto(files, window):
    bins1 = [ius_to_binary(f, gold=False) for f in files]
    bins2 = [ius_to_binary(f, gold=True) for f in files]
    concat1 = sum(bins1,[])
    concat2 = sum(bins2,[])
    str_1 = bin_to_string(concat1)
    str_2 = bin_to_string(concat2)
    return windowdiff(str_1,str_2,window)

def combined_pk_auto(files, window):
    bins1 = [ius_to_binary(f, gold=False) for f in files]
    bins2 = [ius_to_binary(f, gold=True) for f in files]
    concat1 = sum(bins1,[])
    concat2 = sum(bins2,[])
    str_1 = bin_to_string(concat1)
    str_2 = bin_to_string(concat2)
    return pk(ref=str_1,hyp=str_2,k=window,boundary='1')

def combined_pre_rec_f1_auto(files):
    bins1 = [ius_to_binary(f, gold=False) for f in files]
    bins2 = [ius_to_binary(f, gold=True) for f in files]
    concat1 = sum(bins1, [])
    concat2 = sum(bins2, [])
    return precision_recall_f1(concat1, concat2)

def combined_average_iu_length(files, gold=False):
    combined_ius = []
    for doc in files:
        ius, disc_ius = gen_iu_collection(doc, gold)
        combined_ius.extend(ius.values())
    iu_lengths = [len(iu) for iu in combined_ius]
    return np_avg(iu_lengths)

def combined_variance_iu_length(files, gold=False):
    combined_ius = []
    for doc in files:
        ius, disc_ius = gen_iu_collection(doc, gold)
        combined_ius.extend(ius.values())
    iu_lengths = [len(iu) for iu in combined_ius]
    return np_var(iu_lengths)

## Function that computes agreement statistics between two documents
def doc_agreement(doc1, doc2):
    ius_1, disc_ius_1 = gen_iu_collection(doc1, gold=True)
    ius_2, disc_ius_2 = gen_iu_collection(doc2, gold=True)
    dict_1 = iu_agreement(ius_1, disc_ius_1, ius_2, disc_ius_2)

    bin_1 = ius_to_binary(doc1, gold=True)
    bin_2 = ius_to_binary(doc2, gold=True)
    dict_2 =binary_agreement(bin_1, bin_2)
    combined_dicts = {**dict_1, **dict_2}
    return combined_dicts

## Function that computes agreement statistics between two binaries
def binary_agreement(bin_1, gold_bin):
    str_1 = bin_to_string(bin_1)
    str_2 = bin_to_string(gold_bin)
    prec_rec_f1 = precision_recall_f1(auto_bin=bin_1, gold_bin=gold_bin)
    reliability_data = [bin_1, gold_bin]
    res = {}
    res["spans"] = sum(bin_1)+1
    res["gold_spans"] = sum(gold_bin)+1
    res["perfect_spans"] = bin_perfect_match(bin_1, gold_bin)
    res["accuracy"] = accuracy(bin_1,gold_bin)
    res["kappa"] = kappa(bin_1,gold_bin)
    res["kripp"] = krippendorff.alpha(reliability_data)
    res["wdiff3"] = windowdiff(str_1,str_2,3)
    res["wdiff5"] = windowdiff(str_1,str_2,5)
    res["pk3"] = pk(ref=str_1,hyp=str_2,k=3,boundary='1')
    res["pk5"] = pk(ref=str_1,hyp=str_2,k=5,boundary='1')
    res["Ipk3"] = pk(ref=str_2,hyp=str_1,k=3,boundary='1')
    res["Ipk5"] = pk(ref=str_2,hyp=str_1,k=5,boundary='1')
    res = {**res, **prec_rec_f1}
    '''
    print("Cohen's kappa: {}".format(res["kappa"]))
    print("Krippendorff alpha: {}".format(res["kripp"]))
    print("Windowdiff size 3: {}".format(res["wdiff3"]))
    print("Windowdiff size 5: {}".format(res["wdiff5"]))
    print("PK size 3: {}".format(res["pk3"]))
    print("PK size 5: {}".format(res["pk5"]))
    '''
    return res

def ius_to_binary(sentences, gold=False):
    binary = []            #result binary container
    prev_label = None    #previous label temp var
    # func to get iu_label.
    label = lambda x: x._.iu_index
    # look at a different label for gold Ius
    if gold is True:
        label = lambda x: x._.gold_iu_index
    
    for sent in sentences:
        for word in sent:
            #iterate words
            #if we are looking at the first word, set prev_label and do nothing
            if prev_label is None:
                prev_label = label(word)
            #add 1 if we change IU, 0 otherwise
            elif label(word) is prev_label:
                binary.append(0)
            else:
                prev_label = label(word)
                binary.append(1)
    return binary


def comb_ius_to_binary(sentences, comb_label):
    bin = []  # result binary container
    prev_label = None  # previous label temp var
    # func to get iu_label.
    def label(x): return x._.iu_comb[comb_label]
    for sent in sentences:
        for word in sent:
            #iterate words
            #if we are looking at the first word, set prev_label and do nothing
            if prev_label is None:
                prev_label = label(word)
            #add 1 if we change IU, 0 otherwise
            elif label(word) is prev_label:
                bin.append(0)
            else:
                prev_label = label(word)
                bin.append(1)
    return bin

def merge_bins(bin_1, bin_2):
    res = []
    for el1, el2 in zip(bin_1, bin_2):
        merged_res = 0
        if el1 == 1 or el2 == 1:
            merged_res = 1
        res.append(merged_res)
    return res

def bin_to_segs(sents, bin_array):
    words = []
    for sent in sents:
        words.extend(sent)
    if len(words) != len(bin_array)+1:
        raise(Exception("Error mapping merged bin to textfile"))
    support_bin = [0]
    support_bin.extend(bin_array)
    segs = []
    cur_seg = []
    for word, el in zip(words,support_bin):
        if el == 0:
            cur_seg.append(word)
        else:
            segs.append(cur_seg)
            cur_seg = [word]
    segs.append(cur_seg)
    return segs
    

bin_to_string = lambda bin: ''.join(str(el) for el in bin)

iu_to_string = lambda iu: [w.text for w in iu]

def precision_recall_f1(auto_bin=None, gold_bin=None):
    if len(auto_bin) != len(gold_bin):
        raise Exception("Inconsistent binaries for Precision and recall")
    sk_res = precision_recall_fscore_support(gold_bin, auto_bin, beta=1)
    res = {
        "precision": sk_res[0][1],
        "recall": sk_res[1][1],
        "f1": sk_res[2][1]
    }
    return res


def iu_agreement(ius_1, disc_ius_1, ius_2, disc_ius_2):
    perfect_matching = 0
    acceptable_ius = 0
    perfect_disc_ius = 0
    acceptable_disc_ius = 0
    #for idx, iu in ius_1.items():
    #    if iu in ius_2.values():
    #        #print("perfect match!")
    #        perfect_matching += 1
    for idx_1, iu_1 in ius_1.items():
        for idx_2, iu_2 in ius_2.items():
            if iu_match(iu_1, iu_2):
                acceptable_ius += 1
                if iu_to_string(iu_1) == iu_to_string(iu_2):
                    perfect_matching += 1
                if idx_1 in disc_ius_1 and idx_2 in disc_ius_2:
                    acceptable_disc_ius +=1
                    if iu_to_string(iu_1) == iu_to_string(iu_2):
                        perfect_disc_ius += 1
    
    iu1_lengths = [len(iu) for iu in ius_1.values()]
    iu2_lengths = [len(iu) for iu in ius_2.values()]
    res = {}
    res["num_iu1"] = len(ius_1)
    res["words_doc1"] = sum(iu1_lengths)
    res["num_iu2"] = len(ius_2)
    res["words_doc2"] = sum(iu2_lengths)
    res["num_disc_iu1"] = len(disc_ius_1)
    res["num_disc_iu2"] = len(disc_ius_2)
    res["perfect"] = perfect_matching
    res["acceptable"] = acceptable_ius
    res["perfect_disc"] = perfect_disc_ius
    res["acceptable_disc"] = acceptable_disc_ius
    res["avg_len_iu1"] = np_avg(iu1_lengths)
    res["var_len_iu1"] = np_var(iu1_lengths)
    res["avg_len_iu2"] = np_avg(iu2_lengths)
    res["var_len_iu2"] = np_var(iu2_lengths)
    '''
    print("#IU in doc 1: {}".format(res["num_iu1"]))
    print("#IU in doc 2: {}".format(res["num_iu2"]))
    print("#Discontinuous IU in doc 1: {}".format(res["num_disc_iu1"]))
    print("#Discontinuous IU in doc 2: {}".format(res["num_disc_iu2"]))
    print("Perfect IU matches: {}".format(res["perfect"]))
    print("Acceptable IUS (3Werror): {}".format(res["acceptable"]))
    print("Perfect discontinuous IUs: {}".format(res["perfect_disc"]))
    print("Acceptable discontinuous IUs: {}".format(res["acceptable_disc"]))
    '''
    return res

def count_labels_from_file(labelled_file):
    labels = set([word._.iu_index for sent in labelled_file for word in sent])
    return count_seg_multiple_labels(labels)

def count_seg_multiple_labels(labels):
    res = {}
    for label in labels:
        rules = set("UNL")
        if label != -1:
            #extract the rule labels
            rules = label.split("-")[-1]
            #separate the rules and make them unique(set())
            rules = set(label.split(","))
        num_rules = len(rules)
        if num_rules not in res.keys():
            res[num_rules] = 1
        else:
            res[num_rules] += 1
    return res

def classify_seg_errors(labelled_file):
    ref_iu_labels = {}
    error_labels = set()
    for sent in labelled_file:
        for word in sent:
            #ignore punctuation
            if word.pos_ != "PUNCT":
                gold_label = word._.gold_iu_index
                seg_label = word._.iu_index
                #add link between gold index and extraction index
                if gold_label not in ref_iu_labels.keys():
                    if seg_label in ref_iu_labels.values():
                        error_labels.add(seg_label)
                        ref_iu_labels[gold_label] = "ERR"
                    else:
                        ref_iu_labels[gold_label] = seg_label
                # if the segmentation label does not respect the gold label
                elif seg_label != ref_iu_labels[gold_label]:
                    error_labels.add(seg_label)
                    error_labels.add(ref_iu_labels[gold_label])
    error_counts = {}
    for error in error_labels:
        rules = ["UNL"]
        #if the error is -1 then it was not segmented
        if error != -1:
            #extract the rule labels
            rules= error.split("-")[-1]
            #separate the rules and make them unique(set())
            rules = set(rules.split(","))
        for rule in rules:
            if rule not in error_counts.keys():
                error_counts[rule] = 1
            else:
                error_counts[rule] += 1
    return error_counts, error_labels

def count_rule_segments(file):
    rule_count = {}
    labels = set([word._.iu_index for sent in file for word in sent])

    for label in set(labels):
        rules = ["UNL"]
        #if the error is -1 then it was not segmented
        if label != -1:
            #extract the rule labels
            rules= label.split("-")[-1]
            #separate the rules and make them unique(set())
            rules = set(rules.split(","))
        for rule in rules:
            if rule not in rule_count.keys():
                rule_count[rule] = 1
            else:
                rule_count[rule] += 1
    return rule_count

def aggregate_rule_segments(files):
    aggregate_count = {}
    for file in files:
        for rule, count in count_rule_segments(file).items():
            if rule not in aggregate_count:
                aggregate_count[rule] = count
            else:
                aggregate_count[rule] += count
    return aggregate_count

def file_stats_w_errors(file):
    res = {}
    seg_stats = count_rule_segments(file)
    seg_errors, label_errors = classify_seg_errors(file)
    seg_label_freqs = count_labels_from_file(file)
    seg_label_errors = count_seg_multiple_labels(label_errors)

    for label in rule_labels:
        count = 0
        err = 0
        if label in seg_stats.keys():
            count = seg_stats[label]
        if label in seg_errors.keys():
            err = seg_errors[label]
        res["#"+label] = count
        res["ERR_"+label] = err

    for n_segs, freq in seg_label_freqs.items():
        err = 0
        if n_segs in seg_label_errors.keys():
            err = seg_label_errors[n_segs]
        res["ACC_{}".format(n_segs)] = (freq-err)/freq
    return res

def data_set_errors(files):
    aggregate_count = aggregate_rule_segments(files)
    combined_errors = {}
    for file in files:
        errors, label_errors = classify_seg_errors(file)
        for rule, freq in errors.items():
            if rule not in combined_errors.keys():
                combined_errors[rule] = freq
            else:
                combined_errors[rule] += freq

    res = [["RULE","EXTRACTIONS","ERRORS","%"]]
    for rule, number in aggregate_count.items():
        n_error = 0
        if rule in combined_errors.keys():
            n_error = combined_errors[rule]
        res.append([rule,number,n_error,n_error/number*1.0])
    return res

# this function checks if two ius are similar
# acceptable_error refers to number of words
def iu_match(iu_1, iu_2, acceptable_error=3):
    err_count = 0
    for word in iu_1:
        if word.text not in iu_to_string(iu_2):
            err_count +=1
    for word in iu_2:
        if word.text not in iu_to_string(iu_1):
            err_count +=1
    return err_count < acceptable_error

## placeholder for when i get data from multiple annotatators
def multiple_kripp(files, gold):
    return None

def bin_perfect_match(bin1, bin2):
    perfect_matches = 0

    prev_flag = True #the starting boundary is always in agreement
    for b1,b2 in zip(bin1, bin2):
        #If both binaries are 1
        if b1 == 1 and b2 == 1:
            # a perfect match happens only when there is an agreement
            # between both boundaries of a segment
            # I need a previous boundary agreement (signaled by prev_flag)
            # and I need to only find 0s in both binaries when the next
            # agreement happens
            if prev_flag is False:
                prev_flag = True
            else:
                perfect_matches += 1
        #if either binary is one but the other is not
        elif b1 == 1 or b2 == 1:
            # I found a disagreement, turn of the prev agreement flag
            prev_flag = False
    if prev_flag is True:
        # the end of the text is always an agreement boundary
        perfect_matches +=1
    return perfect_matches


def prepare_stats_csv(collections, filenames, skip_auto=False, iu_agreement=True,seg_bot_stats=False, labelled_files=None):
    res = []
    row = []
    row.append("File")
    if iu_agreement==True:
        row.append("#IUS")
        row.append("#Gold IUS")
        row.append("#Disc IUS")
        row.append("#Disc Gold IUS")
        row.append("#Words doc 1")
        row.append("#Words doc 2")
        row.append("Perfect IU match")
        row.append("Acceptable IUs (+/- 3 words)")
        row.append("Perfect Disc IU match")
        row.append("Acceptable Disc IUs (+/- 3 words)")
        row.append("Average IU length")
        row.append("Variance IU length")
        row.append("Average Gold IU length")
        row.append("Variance Gold IU length")
    if seg_bot_stats==True:
        row.append('#EDU')
        row.append("#Gold IUS")
        row.append("#Disc Gold IUS")
        row.append("#Words doc EDU")
        row.append("#Words doc IU")
        row.append("Perfect IU/EDU matches")
        row.append("Acceptable IU/EDU matches (+/- 3 words)")
        row.append("Average Segbot EDU length")
        row.append("Variance Segbot EDU length")
        row.append("Average Gold IU length")
        row.append("Variance Gold IU length")
        
    row.append("Spans")
    row.append("Gold spans")
    row.append("Perfect spans")
    row.append("Accuracy")
    row.append("Precision")
    row.append("Recall")
    row.append("F1")
    row.append("Cohen's Kappa")
    row.append("Krippendorff alpha")
    row.append("Windowdiff size 3")
    row.append("Windowdiff size 5")
    row.append("PK size 3")
    row.append("PK size 5")
    #row.append("Inverted PK size 3")
    #row.append("Inverted PK size 5")
    if not skip_auto:
        for label in rule_labels:
            row.append("# {}".format(label))
            row.append("ERR {}".format(label))
        row.append("IU Accuracy")
        for i in range(1, len(rule_labels)+1):
            row.append("Acc seg w {} labels".format(i))
    res.append(row)
    f_idx = 0
    for dictionary in collections:
        row = []
        row.append(filenames[f_idx])
        if iu_agreement==True:
            row.append(dictionary["num_iu1"])
            row.append(dictionary["num_iu2"])
            row.append(dictionary["num_disc_iu1"])
            row.append(dictionary["num_disc_iu2"])
            row.append(dictionary["words_doc1"])
            row.append(dictionary["words_doc2"])
            row.append(dictionary["perfect"])
            row.append(dictionary["acceptable"])
            row.append(dictionary["perfect_disc"])
            row.append(dictionary["acceptable_disc"])
            row.append(dictionary["avg_len_iu1"])
            row.append(dictionary["var_len_iu1"])
            row.append(dictionary["avg_len_iu2"])
            row.append(dictionary["var_len_iu2"])
        if seg_bot_stats == True:
            row.append(dictionary["num_edus"])
            row.append(dictionary["num_iu2"])
            row.append(dictionary["num_disc_iu2"])
            row.append(dictionary["words_doc1"])
            row.append(dictionary["words_doc2"])
            row.append(dictionary["perfect"])
            row.append(dictionary["acceptable"])
            row.append(dictionary["avg_len_edu"])
            row.append(dictionary["var_len_edu"])
            row.append(dictionary["avg_len_gold"])
            row.append(dictionary["var_len_gold"])
        row.append(dictionary["spans"])
        row.append(dictionary["gold_spans"])
        row.append(dictionary["perfect_spans"])
        row.append(dictionary["accuracy"])
        row.append(dictionary["precision"])
        row.append(dictionary["recall"])
        row.append(dictionary["f1"])
        row.append(dictionary["kappa"])
        row.append(dictionary["kripp"])
        row.append(dictionary["wdiff3"])
        row.append(dictionary["wdiff5"])
        row.append(dictionary["pk3"])
        row.append(dictionary["pk5"])
        #row.append(dict["Ipk3"])
        #row.append(dict["Ipk5"])
        if not skip_auto:
            errors = 0
            segs = 0
            for label in rule_labels:
                row.append(dictionary["#"+label])
                segs += dictionary["#"+label]
                row.append(dictionary["ERR_"+label])
                errors += dictionary["ERR_"+label]
            row.append((segs-errors)/segs)
            # initialize empty array to ensure lenght consistency
            acc_row = ["" for x in rule_labels] 
            for i in range(len(rule_labels)):
                if "ACC_{}".format(i+1) in dictionary.keys():
                    acc_row[i] = dictionary["ACC_{}".format(i+1)]
            row.extend(list(acc_row))
        res.append(row)
        f_idx += 1
    # Calculating averages and sums
    sum_row = ["SUM"]
    avg_row = ["AVG"]
    for col_idx in range(1,len(res[0])):
        #start from 1, ignore filenames
        col = [0 if type(row[col_idx]) == str else row[col_idx] for row in res]
        col = col[1:]   #ignore header
        col = [np_float64(cell) for cell in col] #cast to float
        sum_row.append(np_sum(col))
        avg_row.append(np_avg(col))
    res.append(sum_row)
    res.append(avg_row)
    return res

def prepare_combined_stats_csv(labelled_files):
    res = []
    res.append(["(Micro) Average IU length:", 
                combined_average_iu_length(labelled_files)])
    res.append(["(Micro) Variance in IU length:", 
                combined_variance_iu_length(labelled_files)])
    res.append(["(Micro) Average Gold IU length:",
                combined_average_iu_length(labelled_files, gold=True)])
    res.append(["(Micro) Variance in Gold IU length:",
                combined_variance_iu_length(labelled_files, gold=True)])

    res.append(["(Micro) Average Kappa:", combined_kappa_auto(labelled_files)])
    res.append(
        ["(Micro) Average WindowDiff 3:", combined_windowdiff_auto(labelled_files, 3)])
    res.append(
        ["(Micro) Average WindowDiff 5:", combined_windowdiff_auto(labelled_files, 5)])
    res.append(["(Micro) Average PK 3:", combined_pk_auto(labelled_files, 3)])
    res.append(["(Micro) Average PK 5:", combined_pk_auto(labelled_files, 5)])
    prf_data = combined_pre_rec_f1_auto(labelled_files)
    res.append(["(Micro) Average Precision:", prf_data['precision']])
    res.append(["(Micro) Average Recall:", prf_data['recall']])
    res.append(["(Micro) Average F1:", prf_data['f1']])
    res.extend(data_set_errors(labelled_files))
    return res


#Permutation-randomization
#Repeat R times: randomly flip each m_i(A),m_i(B) between A and B with probability 0.5, calculate delta(A,B).
# let r be the number of times that delta(A,B)<orig_delta(A,B)
# significance level: (r+1)/(R+1)
# Assume that larger value (metric) is better
# n number of items len(data_A == data_B)
# R number of randomisation suggestion = 10'000
def rand_permutation(data_A, data_B, n, R):
    delta_orig = float(sum([x - y for x, y in zip(data_A, data_B)]))/n
    r = 0
    for x in range(0, R):
        temp_A = data_A
        temp_B = data_B
        # which samples to swap without repetitions
        samples = [np_random.randint(1, 3) for i in range(n)]
        swap_ind = [i for i, val in enumerate(samples) if val == 1]
        for ind in swap_ind:
            temp_B[ind], temp_A[ind] = temp_A[ind], temp_B[ind]
        delta = float(sum([x - y for x, y in zip(temp_A, temp_B)]))/n
        if(delta <= delta_orig):
            r = r+1
    pval = float(r+1.0)/(R+1.0)
    return pval
