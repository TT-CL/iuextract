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
        if indexes[i] != cur_idx:
            cur_idx = indexes[i]
            res += closer+opener+"{}|".format(cur_idx)
        res += "{} ".format(texts[i])
    res += closer     #add final closed bracket ] at the end of the string
    res = res[1:] #crop first closed bracket ] from the beginning of the string
    return res

def init_comb(file):
    rule_combinations = get_rule_combinations()

    for sent in file:
        for word in sent:
            for comb in rule_combinations:
                comb_label = "_".join(comb)
                word._.iu_comb[comb_label] = -1

def get_rule_combinations():
    rule_combinations = []
    for i in range(len(available_rules)):
        rule_combinations.extend(combinations(available_rules, i+1))
    rule_combinations = [list(comb) for comb in rule_combinations]
    return rule_combinations