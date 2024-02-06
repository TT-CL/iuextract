'''
Main iuextract module
In here you will find functions to label Spacy Docs with IU information
'''

# import re
# from itertools import combinations
# from functools import cmp_to_key
from collections import deque
# from pprint import pprint
from .data import __read_filter
from .utils import gen_iu_collection, get_iu_str_list
import spacy
from spacy.tokens import Token, Doc
import pkgutil

# initialize the list of filtered IUs from an external file
filter_file = pkgutil.get_data(__name__,"transition_signals.txt")
filtered_ius = __read_filter(filter_file)

# Spacy Token Extension for IU combinations
Token.set_extension("iu_comb", default={}, force=True)

available_rules = ["R1", "R2", "R3", "R5", "R6", "R8", "R10"]

def segment_ius(text, mode = "obj", rule_list=None, spacy_model = 'en_core_web_lg'):
    '''
    Function to segment a text into a list of IUs
    
    :param file: (List(Span)) a file parsed with spacy
    :param mode: what kind of result you wish to obtain. Accepted modes: 'obj', 'str'
    :param rule_list: (List(str)) an optional parameter, you can filter the IU extraction rules by providing an array with a list of rules you wish to use. Default: None
    :param spacy_model: (str) the spacy model to load. Default: en_core_web_lg

    :return: either an IU collection or a str representation of an IU collection
    '''
    nlp = spacy.load(spacy_model)
    parsed = nlp(text)
    label_ius(parsed, rule_list)
    res = None
    if mode == "obj":
        res = gen_iu_collection(parsed)
    elif mode == "str":
        res = get_iu_str_list(parsed)
    return res

def label_ius(file, rule_list=None):
    '''
    Function to label a Spacy object with the IUs
    
    :param file: (List(Span)) a file parsed with spacy
    :param rule_list: (List(str)) an optional parameter, you can filter the IU extraction rules by providing an array with a list of rules you wish to use. Default: None

    No output is expected, as the Idea Units labels will be stored inside
    the spacy tokens
    '''
    sents = file
    if isinstance(file, Doc):
        sents = file.sents

    s_idx = 0
    for sentence in sents:
        root = sentence[0].sent.root
        # print("**Sentence:\n{}".format(sentence))
        # print("*root: {}, POS: {}".format(root.text,root.pos_))

        to_process = __tag_nodes(sentence, combination_array=rule_list)
        if len(to_process) == 0:
            # no rule is applicable, segment the full sentence.
            # print("No rule applicable to sent:\n\t{}".format(sentence))
            to_process[root] = ["UNL"]
        to_process = __order_nodes_bfs_dict(to_process)

        combination_label = None
        if rule_list is not None:
            combination_label = "_".join(rule_list)

        __color_ius(sentence, to_process, s_idx, combination_label)

        #Rule 10
        if rule_list is None or "R10" in rule_list:
            __inline_fixes(sentence)
        # print(iu2str(sentence))
        # print()
        s_idx = s_idx + 1
    return None


subj = ["nsubj", "nsubjpass", "csubj", "csubjpass"]


def __is_V_with_S(word):
    '''
    this function says if rule 1 is applyable to word and all its dependants
    '''
    res = False
    # loop only if the word is a verb
    if word.pos_ == "VERB" or word.pos_ == "AUX":
        # don't split multiple auxiliaries as of Rule 4
        if word.dep_ not in subj:
            for child in word.children:
                # if I find one subject as a dependant
                if child.dep_ in subj:
                    res = True
                    break
    return res


def __order_nodes_bfs_dict(nodelist):
    '''
    compute a bfs to find the extraction order of nodes
    '''
    # get the root
    root = list(nodelist.keys())[0].sent.root
    order = []
    q = deque([root])
    # end the loop if I found all the nodes or if I explored the whole tree
    while q and nodelist:
        # pop child and add grandchildren to the queue
        cur_node = q.popleft()
        q.extend(cur_node.children)

        # check node
        if cur_node in nodelist.keys():
            order.append([cur_node, nodelist[cur_node]])
            nodelist.pop(cur_node)
    # filter the unwanted IUs
    for node_arr in order:
        # get the head node of the IU
        node = node_arr[0]
        # generate the IU text
        iu_text = " ".join(t.text for t in node.subtree)
        # if the iu.lowercase is in the filtered list:
        if iu_text.lower().strip(",.") in filtered_ius:
            # print("Filtering IU: {}".format(iu_text))
            order.remove(node_arr)
    order.reverse()  # the extraction order needs to be reversed
    return order


'''
# Bool function for Rule 2
def is_relcl(word):
    res = False
    #enter the loop only if we have a relative clause
    if word.dep_ == "relcl":
        for child in word.children:
            # if we have a pronoun
            if child.dep_ == "mark":
                res = True
                break
    return res
'''


def __is_sconj(word):
    ''' bool function for Rule 2 '''
    res = False
    # the sub conjunction is the introducion of a prepositional phrase
    if word.pos_ == "SCONJ" and word.dep_ == "prep":
        res = True
    if word.pos_ == "AUX" or word.pos_ == "VERB":
        for child in word.children:
            # if we have a sconj
            if child.dep_ == "mark" and child.pos_ == "SCONJ":
                res = True
                break
    return res


def __is_complementizer(word):
    ''' Rule 2B '''
    res = False
    # enter the loop only if we have a clausal complement
    if word.dep_ == "ccomp":
        for child in word.children:
            # if we have a pronoun
            if child.dep_ == "mark":
                res = True
                break
    return res


def __find_PP_PC(sentence):
    '''
    This function finds all Prepositional phrases as Prepositional complements
    It will return all prepostional modifiers
    that are followed by prepostional complements
    '''
    preps = []
    for i in range(len(sentence)):
        if sentence[i].dep_ == "prep":
            # check if I still have more words before the end of the sentence
            if (i+1) < len(sentence):
                # if my preposition is followed by the
                if sentence[i+1].dep_ == "pcomp":
                    preps.append(i)
    return preps


def __rule_PP(sentence, s_idx):
    # Prepositional phrases as Prepositional complements are phrases
    # formed by a preposition directly followed by a Prepositional complement
    # I suggest using this instead of the arbitrary 5 word limit
    return None


def __is_isolated(sequence):
    '''
    This function evaluates whether a sequence of words is isolated.
    A sequence is isolated if I only have an arc between it and the rest of the
    sentence. either 1 head or 1 child.
    '''
    outside_connections = 0
    for word in sequence:
        # if word does not have the attribute head, it is the root
        if hasattr(word, "head"):
            if word.head not in sequence and word.head.pos_ != 'PUNCT':
                outside_connections += 1
        for child in word.children:
            if child not in sequence and child.pos_ != 'PUNCT':
                outside_connections += 1
    # print(outside_connections)
    return outside_connections == 1


def __find_seq_head(sequence):
    ''' given a sequence, returns the head of the subtree '''
    node = sequence[0]
    root = node.sent.root
    if node is not root:
        father = node.head
        while father in sequence:
            node = father
            father = node.head
            # a root father will not have a head
            if father == root:
                if father in sequence:
                    node = father
                break
    return node


def __citation_check(node):
    res = False
    if node.text.isdigit():
        # the word is a digit and only child of the appos
        if len([child for child in node.children]) == 0:
            # Citation!
            res = True
    return res


def __stopword_check(node):
    res = False
    # print("node: {}".format(node))
    # print("is_stop: {}".format(node.is_stop))
    # print("pos: {}".format(node.pos_))
    if node.is_stop or node.pos_ == "PUNCT":
        res = True
        for child in node.children:
            res = res and __stopword_check(child)
    return res


def __tag_parens(sentence):
    tag_list = []
    q = []
    word_idx = 0
    for word in sentence:
        if word.text == "(":
            q.append((word, word_idx))
        elif word.text == ")":
            # pop the last (
            open_idx = None
            for el, idx in reversed(q):
                if el.text == "(":
                    q.remove((el, idx))
                    open_idx = idx
                    break
            # I found an open paren
            if open_idx is not None:
                # you can slice a sentence by the 2 indexes to find the substr
                seg_slice = sentence[open_idx+1:word_idx]
                # CHECK IF THIS SEGMENT OF TEXT IS ISOLATED
                # FROM THE REST OF THE SENT
                if __is_isolated(seg_slice):
                    slice_head = __find_seq_head(seg_slice)
                    if __citation_check(slice_head):
                        # print("R3parens - filtered due to citation")
                        pass
                    else:
                        tag_list.append([slice_head, "R3.1"])
        word_idx += 1
    return tag_list


def __tag_hyphens(sentence):
    tag_list = []
    q = []
    word_idx = 0
    for word in sentence:
        if word.text == "-":
            # first hyphen
            if len(q) == 0:
                q.append(word_idx)
            else:
                prev_idx = q[len(q)-1]
                seg_slice = sentence[prev_idx+1:word_idx]
                if __is_isolated(seg_slice):
                    slice_head = __find_seq_head(seg_slice)
                    if __citation_check(slice_head):
                        # print("R3hypen - filtered due to citation")
                        pass
                    else:
                        tag_list.append([slice_head, "R3.1"])
                else:
                    # maybe the next hyphen couple will be a parenthetic clause
                    q.append(word_idx)
        word_idx += 1
    return tag_list


def __tag_commas(sentence):
    tag_list = []
    q = []
    word_idx = 0
    for word in sentence:
        # for the first comma I want to check the slice from the beginning of
        # the sentence.
        prev_idx = -1
        # analize slices between the commas AND
        # the last slice between the last slice and the end of the sentence
        is_valid_last_slice = word_idx == len(sentence)-1 and len(q) != 0
        if word.text == "," or is_valid_last_slice:
            # if we have found some commas beforehand
            if len(q) != 0:
                # start from
                prev_idx = q[len(q)-1]
            # I don't include commas in the slices. This is because they have
            # weird dependencies. I will manually fix them with inline_fixes()
            # This has the sideffect of making it easier to find the right iu
            # when the first IU in a sentence is comprised of stopwords.
            # The comma will tell me where to attach the stopword IU.
            seg_slice = sentence[prev_idx+1:word_idx]
            # print("analyzing slice \"{}\"".format(slice))
            if __is_isolated(seg_slice):
                slice_head = __find_seq_head(seg_slice)
                # RULE 3 EXCEPTIONS
                if __citation_check(slice_head):
                    # print("R3comma - filtered due to citation")
                    # print("--- {} ---".format(slice))
                    pass
                elif __stopword_check(slice_head):
                    # print("R3B - filtered due to stopword_check")
                    # print("--- {} ---".format(slice))
                    tag_list.append([slice_head, "JOIN"])
                else:
                    # print("Splitting! Rule R3Bc")
                    # print("--- {} ---".format(slice))
                    tag_list.append([slice_head, "R3"])
            # I always want to keep track of the last comma I found.
            q.append(word_idx)
        word_idx += 1
    return tag_list


def __is_infinive_clause(word):
    ''' rule 5 and 6 '''
    res = False
    if word.dep_ == "acl" or word.dep_ == "advcl":
        # infinitival
        if word.tag_ == "TO" or word.tag_ == "VB":
            res = True
        # gerund
        elif word.tag_ == "VBG" or word.tag_ == "VBN":
            res = True
        # else:
            # print(word, word.tag_, word.dep_)
    return res


def __is_appos(word):
    ''' boolean form of rule 7b '''
    res = False
    if word.dep_ == "appos":
        # check if the apposition is a citation:
        if __citation_check(word):
            # Citation!
            # print("R3.2 - filtered due to citation")
            pass
        else:
            res = True
    return res


def __is_infinitive_verbal(word):
    ''' boolean for rule 7c '''
    res = False
    # a verb is infinitive
    if word.pos_ == "VERB" and word.dep_ == "xcomp" and word.tag_ == "VB":
        # print("VERBAL!")
        # print("word: {}, head: {}, head.pos: {}".format(word, word.head, word.head.pos_))
        # Check if we have the auxiliar TO
        for child in word.children:
            if child.tag_ == "TO":
                res = True
                break
        '''
        for child in word.children:
            # the verbal is preceded by to
            bool = child.text.lower() == "to"
            # to is an infinitival to
            bool = bool and child.pos_ == "PART"
            # to is an auxiliary
            bool = bool and child.dep_ == "aux"
            if  bool is True:
                res = True
                break
        '''
    return res


def __find_long_PP(sent, tagged_nodes):
    def add_node(word, rule):
        if word not in tagged_nodes.keys():
            tagged_nodes[word] = [rule]
        else:
            tagged_nodes[word].append(rule)
    already_marked = []
    # looking from right to left so that I ensure 5 long pps
    # for word in reversed(sent):
    for word in sent:
        # if the word is a prepositional modifier:
        if word not in already_marked:
            # if the pp head is directly dependant on a verb
            if word.dep_ == "prep" and word.head.pos_ in ["AUX", "VERB"]:
                # count all the children that are not already labeled
                visited = []
                q = deque([word])
                lenght = 0
                while len(q) > 0:
                    el = q.popleft()
                    visited.append(el)
                    if el not in tagged_nodes.keys():
                        # add 1 to the word lenght
                        if word.pos_ != "PUNCT":
                            lenght += 1
                        q.extend(el.children)
                # if the pp is long enough
                if lenght >= 5:
                    already_marked.extend(visited)
                    add_node(word, "R8")
    return tagged_nodes


def __inline_fixes(sent):
    previous_label = None
    attach_prev = [",", ".", ")", "!", "?", ";"]
    for i in range(len(sent)):
        word = sent[i]
        # PUNCTUATION FIX
        # attach each comma, fullstop, ), ! and ? to the previous word
        if word.text in attach_prev:
            # OOB check
            if i > 0:
                word._.iu_index = sent[i-1]._.iu_index
        elif word.text == "(":
            if (i+1) < len(sent):  # OOB check
                word._.iu_index = sent[i+1]._.iu_index
        # conjunctions go with their follwing word
        if word.pos_ == "CCONJ":
            if (i+1) < len(sent):  # OOB check
                word._.iu_index = sent[i+1]._.iu_index
        # JOIN FIX
        # we attach meaningless ius (stopwords) to the left.
        # If they are in the initial position, then we attach them to the right
        if word._.iu_index == "JOIN":
            if previous_label is None:
                # find a new label:
                previous_label = "JOIN"
                # go forward until I find a new label, then backtrack
                j = i+1
                # print("scanning right...")
                while previous_label == "JOIN" and j < len(sent):
                    # print(sent[j], sent[j]._.iu_index)
                    previous_label = sent[j]._.iu_index
                    j += 1
                # Now previous_label is correct
            # change the JOIN label
            word._.iu_index = previous_label

        # store every word's label (look left)
        previous_label = word._.iu_index
    return None


def __tag_nodes(sentence, combination_array=None):
    '''
    This function tags all nodes that (along with their dependencies)
    need to be segmented.
    '''
    res = {}

    if combination_array is None:
        combination_array = available_rules

    def add_node(word, rule):
        if word not in res.keys():
            res[word] = [rule]
        else:
            res[word].append(rule)
    for word in sentence:
        if "R1" in combination_array:
            if __is_V_with_S(word):
                add_node(word, "R1")
                # print("V with S: {}".format(word))
        if "R2" in combination_array:
            if __is_sconj(word):
                add_node(word, "R2")
                # print("sconj: {}".format(word))
            if __is_complementizer(word):
                add_node(word, "R2")
                # print("complementizer: {}".format(word))
        if "R3" in combination_array:
            if __is_appos(word):
                add_node(word, "R3.2")
                # print("appos: {}".format(word))
        if "R5" in combination_array:
            if __is_infinive_clause(word):
                add_node(word, "R5")
                # print("acl: {}".format(word))
        if "R6" in combination_array:
            if __is_infinitive_verbal(word):
                add_node(word, "R6.2")
                # print("verbal: {}".format(word))
    # Rule 3
    if "R3" in combination_array:
        for tag in __tag_parens(sentence):
            word, rule = tag
            add_node(word, rule)
        for tag in __tag_hyphens(sentence):
            word, rule = tag
            add_node(word, rule)
        for tag in __tag_commas(sentence):
            word, rule = tag
            add_node(word, rule)
    # pprint(res)
    # print("TAGGING LONG PPS")
    # Rule 8
    if "R8" in combination_array:
        res = __find_long_PP(sentence, res)
    # pprint(res)
    return res


def __color_ius(sentence, to_process, s_idx, combination_label=None):
    '''
    This function colors each word in the sentence according to
    nodes found in the list to_process.
    '''
    iu_idx = 1
    for node, rule_labels in to_process:
        label = "JOIN"
        if "JOIN" not in rule_labels:
            label = "{}-{}-{}".format(s_idx, iu_idx, ",".join(rule_labels))

        iu_idx = iu_idx + 1
        # color words according to Idea Unit following the order
        q = deque([node])
        while q:
            cur_node = q.popleft()
            q.extend(cur_node.children)
            # print(q)

            if combination_label is None:
                #if I don't specify a comb label, default to standard iu index
                if cur_node._.iu_index == -1:
                    # only color unexplored nodes
                    cur_node._.iu_index = label
            else:
                if cur_node._.iu_comb[combination_label] == -1:
                    # only color unexplored nodes
                    cur_node._.iu_comb[combination_label] = label
    return None
