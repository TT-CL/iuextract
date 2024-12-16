import argparse
import sys
import os
import warnings
from .data import import_file
from .extract import label_ius
from .iu_utils import iuDoc2str
import spacy

def main():
    parser = argparse.ArgumentParser(prog='iuextract', description='Segment a raw text into Idea Units')
    parser.add_argument('-i', '--input', help='the {i}nput text. If it is not provided, the program will read the positional arguments input as a text.', type=argparse.FileType('r'), required=False, default=None)
    parser.add_argument('-o', '--output', help="the {o}utput file where to store the ius. By default the segmentation will be shown on the terminal and will not be stored on disk.", nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    parser.add_argument('pos_input', help="the input text to analyze if no filename is provided.", nargs='*')
    #parser.add_argument('-o', '--output', help="the {o}utput file where to store the ius. By default the segmentation will be shown on the terminal and will not be stored on disk. \nAccepted filetype: .txt", required=False, default=None)
    parser.add_argument('-b', '--before', help="a sequence of text to place {b}efore each IU. By default, no prefix is set.", required=False, default="")
    parser.add_argument('-s', '--separator', help="a sequence of text to {s}eparate the index from the IU. By default, the separator is the character | .", required=False, default="|")
    parser.add_argument('-a', '--after', help="a sequence of text to place {a}fter each IU. By default, the suffix is the newline character.", required=False, default="\n")

    args = parser.parse_args()
    input_file = args.input
    text_input = args.pos_input
    text_input = ' '.join(text_input) if len(text_input) > 0 else None
    if input_file is None and text_input is None:
        raise IOError("Please provide a valid input text. Run iuextract -h for more help.")
    if input_file is not None and text_input is not None:
        input_warning = "WARNING: the program detected both an input file passed via the -i argument and input text via positional arguments. Ignoring the positional arguments and only processing the input file."
        warnings.warn(input_warning)
    raw_input = text_input
    if input_file is not None:
        raw_input = input_file.read()
    outputFile = args.output
    prefix = args.before
    suffix = args.after
    separator = args.separator

    nlp = spacy.load("en_core_web_lg")
    parsed = import_file(raw_input, nlp=nlp)
    label_ius(parsed)
    raw_output = iuDoc2str(parsed,gold=False, index_sep=separator, opener=prefix, closer=suffix)
    if not raw_output.endswith('\n'):
        raw_output = f'{raw_output}\n'
    outputFile.write(raw_output)
    return os.EX_OK

if __name__ == "__main__":
    sys.exit(main())