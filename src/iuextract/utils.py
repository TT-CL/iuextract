from itertools import combinations

available_rules = ["R1", "R2", "R3", "R5", "R6", "R8", "R10"]
## This function prints sentences nicely with their IU numbers in brackets
#  you can customize the brackets by changing the opener and closer parameters
def iu_pprint(sent, gold = False, opener="[",closer="]"):
    texts = [token.text for token in sent]
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
            res += closer+opener+"{}|".format(cur_idx)
        res += "{} ".format(texts[i])
    res += closer     #add final closed bracket ] at the end of the string
    res = res[1:] #crop first closed bracket ] from the beginning of the string
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

def get_ius_text(sent):
    res_dict = {}
    for tok in sent:
        if tok._.iu_index not in res_dict.keys():
            res_dict[tok._.iu_index] = [tok]
        else:
            res_dict[tok._.iu_index].append(tok)
    return res_dict

def gen_iu_collection(sentences, gold=False):
    '''
    This function extracts the labeled IUS from a document and keeps track of
    discontinuous Idea Units
    '''
    ius = {}
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
                ius[label(word)] = []
            # if the label didn't change from the previous word I can assume
            # that this label already has a dict entry
            if label(word) is prev_label:
                ius[label(word)].append(word)
            # THE LABEL CHANGED!
            # if we don't have the label in the dict then add it and move on
            elif label(word) not in ius:
                ius[label(word)] = []
                ius[label(word)].append(word)
                prev_label = label(word)
            # the label is already in the dict. We have a discontinuous IU
            else:
                ius[label(word)].append(word)
                disc_ius.add(label(word))
                prev_label = label(word)
    return ius, disc_ius
