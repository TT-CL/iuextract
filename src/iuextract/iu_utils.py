'''
Generic utils module
Some functions include IU printer and a converter to extract IU collections from a spacy object.
'''

from itertools import combinations
from spacy.tokens import Doc
import re
import unicodedata as ud

available_rules = ["R1", "R2", "R3", "R5", "R6", "R8", "R10"]

def clean_str(s):
    ''' 
    String cleanup function
    Removes double-spaces, makes apostrophe consistent and removes emojis

    :param s: (str) the string to clean up
    :return: (str) the cleaned up string
    '''
    #remove double spaces and newlines/tabs
    space_undoubler = lambda s : re.sub(r'\s\s*',' ', s)

    res = s
    res = res.replace("\s\\t\s", " ")
    res = res.replace("\s\\n\s", " ")
    res = res.replace("\s\\r\s", " ")
    res = res.replace("\s\\f\s", " ")
    res = res.replace("’", "'")
    res = res.replace("“", "\"")
    res = res.replace("”", "\"")
    res = res.replace("``", "\"")
    res = res.replace("''", "\"")
    res = space_undoubler(res)
    #res = re.sub("\s+", " ", res) # replace multiple spaces with a single one
    # ensure that each open parens has at most one whitespace before
    res = re.sub("\s+\\(", " (", res)
    # ensure that each close parens has at most one whitespace afterwards
    res = re.sub("\\)\s+", ") ", res)
    #uncomment to ensure compatibility with segbot
    '''
    res = re.sub("\s*\\.", " .", res) 
    res = re.sub("\\.\s*", ". ", res)
    '''

    #Remove weird unicode chars (emojis)
    res = ''.join([c for c in res if ud.category(c)[0] != 'C'])
    # undouble spaces again
    res = space_undoubler(res)
    #remove trailing spaces
    res = res.strip()
    return res

def iu2str(sent, gold = False, index_sep="|", opener="",closer="\n"):
    '''
    This function converts spacy sentences into a string representation of the
    IUs contained within.
    :param sent: (Span) the sentence to convert
    :param gold: (bool) whether I want to print gold labels or not
    :param index_sep: (str) the separator between the IU and the tokens (default: "|")
    :param opener: (str) the IU string opener (default: "")
    :param closer: (str) the IU string closer (default: "\n")
    :return: a string representation of the IUs in a sentence
    '''
    texts = [token.text_with_ws for token in sent]
    indexes = None
    if gold is False:
        indexes = [token._.iu_index for token in sent]
    else:
        indexes = [token._.gold_iu_index for token in sent]
    res = ""
    cur_idx = None
    for i in range(len(texts)):
        #print("i:{}, indexes:{}, cur_idx:{}".format(i,indexes[i],cur_idx))
        if indexes[i] != cur_idx:
            #print(indexes[i], cur_idx)
            cur_idx = indexes[i]
            res += closer+opener+f"{cur_idx}"+index_sep
        res += texts[i]
    res += closer     #add final closed bracket ] at the end of the string
    res = res[1:] #crop first closed bracket ] from the beginning of the string
    return res

def iuDoc2str(doc, gold = False, index_sep="|", opener="",closer="\n", verbose=False):
    converted_sents = [iu2str(sent, gold=gold, index_sep=index_sep, opener=opener, closer=closer).strip() for sent in doc.sents]
    res = '\n'.join(converted_sents)
    if not verbose:
        readable_idxs = __gen_sequential_iu_map(doc=doc, gold=gold)
        #print(readable_idxs)
        temp_res = []
        for sent in res.splitlines():
            split = sent.partition(index_sep)
            temp_res.append(f"{readable_idxs[split[0]]}{split[1]}{split[2]}")
        res = '\n'.join(temp_res)
    return res

def __init_comb(file):
    rule_combinations = __get_rule_combinations()

    for sent in file:
        for word in sent:
            for comb in rule_combinations:
                comb_label = "_".join(comb)
                word._.iu_comb[comb_label] = -1

def __get_rule_combinations():
    rule_combinations = []
    for i in range(len(available_rules)):
        rule_combinations.extend(combinations(available_rules, i+1))
    rule_combinations = [list(comb) for comb in rule_combinations]
    return rule_combinations

'''
def get_ius_text(sent):
    res_dict = {}
    for tok in sent:
        if tok._.iu_index not in res_dict.keys():
            res_dict[tok._.iu_index] = [tok]
        else:
            res_dict[tok._.iu_index].append(tok)
    return res_dict

'''

def gen_iu_collection(doc, gold=False):
    '''
    This function converts a Doc or a list of spacy Spans into a dictionary of
    labeled IUs along with a set of keys for discontinuous units.
    :param doc: (List[Span] | Doc) the list of sentences to convert
    :param gold: (bool) whether I want to convert gold units or not
    :return iu_indexes, disc_ius: a dictionary of iu_indexes and a set of keys refering to
    discontinuous units
    '''
    sentences = doc
    if isinstance(doc, Doc):
        sentences = doc.sents
    iu_indexes = {}
    disc_ius = set()
    label = lambda x: x._.iu_index
    # look at a different label for gold Ius
    if gold is True:
        label = lambda x: x._.gold_iu_index

    prev_label = None
    for sent in sentences:
        for word in sent:
            if prev_label is None:
                # for the first word initialize the dict entry and temp var
                prev_label = label(word)
                iu_indexes[label(word)] = []
            # if the label didn't change from the previous word I can assume
            # that this label already has a dict entry
            if label(word) is prev_label:
                iu_indexes[label(word)].append(word)
            # THE LABEL CHANGED!
            # if we don't have the label in the dict then add it and move on
            elif label(word) not in iu_indexes:
                iu_indexes[label(word)] = []
                iu_indexes[label(word)].append(word)
                prev_label = label(word)
            # the label is already in the dict. We have a discontinuous IU
            else:
                iu_indexes[label(word)].append(word)
                disc_ius.add(label(word))
                prev_label = label(word)
    return iu_indexes, disc_ius

def __gen_sequential_iu_map(doc, gold=False):
    '''
    This function generates a map to rename the verbose indexes to more easily readable ones.
    :param doc: (List[Span] | Doc) the list of sentences to convert
    :param gold: (bool) whether I want to map gold indexes or not
    :return iu_idx_map: a dictionary of iu_indexes with the renamed index
    '''
    sentences = doc
    if isinstance(doc, Doc):
        sentences = doc.sents
    iu_indexes = set()
    sequential_idx=1
    iu_idx_map = {}
    label = lambda x: x._.iu_index
    # look at a different label for gold Ius
    if gold is True:
        label = lambda x: x._.gold_iu_index

    prev_idx = None
    for sent in sentences:
        for word in sent:
            if prev_idx is None:
                # for the first word initialize the dict entry and temp var
                prev_idx = label(word)
                iu_idx_map[prev_idx] = sequential_idx
                sequential_idx +=1
                iu_indexes.add(prev_idx)
            # if the label didn't change from the previous word I can assume
            # that this label already has a dict entry
            if label(word) is prev_idx:
                pass
            # THE LABEL CHANGED!
            # if we don't have the label in the dict then add it and move on
            elif label(word) not in iu_indexes:
                iu_indexes.add(label(word))
                prev_idx = label(word)
                iu_idx_map[prev_idx] = sequential_idx
                sequential_idx +=1
            # the label is already in the dict. We have a discontinuous IU
            else:
                prev_idx = label(word)
                if not str(iu_idx_map[prev_idx]).startswith('D'):
                    iu_idx_map[prev_idx] = f"D{iu_idx_map[prev_idx]}"
    return iu_idx_map

def coll2strings(iu_collection):
    '''
    Function to turn a iu_collection tuple to a simple list of strings

    :param iu_collection: a tuple where 1. is a dictionary of iu_indexes and 2. is a set of keys refering to discontinuous units. The second tuple elem is optional (only the first one is required)
    :return: (str) A string representation of the segmented IUs. Each row has a single unit
    '''
    coll, disc = iu_collection
    res = []
    for key, value in coll.items():
        s = [tok.text_with_ws for tok in value]
        s = ''.join(s)
        s = clean_str(s)
        s = re.sub(r"\s+\,", ",", s)
        s = re.sub(r"\s+\:", ":", s)
        s = re.sub(r"\s+\;", ";", s)
        s = re.sub(r"\s+\.", ".", s)
        s = re.sub(r"\(\s+", "(", s)
        s = re.sub(r"\s+\)", ")", s)
        s = re.sub(r"\[\s+", "[", s)
        s = re.sub(r"\s+\]", "]", s)
        s = re.sub(r"\{\s+", "{", s)
        s = re.sub(r"\s+\}", "}", s)
        res.append(s)
    return res

def get_ius_from_doc(doc, gold = False):
    '''
    This function converts a Doc or a list of spacy Spans into a string representation of the IUs. Each row will contain an IU. Discontinuous IUs are joined together.
    :param doc: (List[Span] | Doc) the list of sentences to convert
    :param gold: (bool) whether I want to convert gold units or not
    :return: (str) a string where each row shows one IU
    '''
    return coll2strings(gen_iu_collection(doc, gold))
