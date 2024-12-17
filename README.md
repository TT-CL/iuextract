# IUExtract
Rule-based Idea Unit segmentation algorithm for the English language.

## Installation
First of all, you need to install the dependencies:
```
pip install spacy
python -m spacy download en_core_web_lg
```
To install the package with the command line tool [install pipx](https://pipx.pypa.io/latest/installation/) and run the following command:
```
pipx install iuextract
```
If the install fails you might want to try to pin a specific python version with the following command:
```
pipx install iuextract --python 3.9
```
**Note:** on first run, the program will download the Spacy model `en_core_web_lg`. This could take some time. A custom Spacy model can be selected when importing the module into your python projects.

If you only wish to use the package in your python projects you can install without executable via
```
pip install iuextract
```

## Command Line Interface (CLI) Usage
Once installed via `pipx`, you can run iuextract via
```
iuextract -i input_file.txt 
```
to segment `file.txt` into Idea Units. 
If installed iuextract via pip you can still run the program via CLI with the following command:
```
python -m iuextract -i input_file.txt 
```
### Input text as argument
If you run iuextract without the `-i` argument, the program will look for positional arguments to segment.
For example
```
iuextract My dog, Chippy, just won its first grooming competition.
```
will output
```
D1|My dog,
2|Chippy,
D1|just won its first grooming competition.
```
**Note:** all positional arguments are joined into a single string. There is no need to put the text between quotation marks.

### Output file
If you don't specify an outp file in the arguments, the segmented file will be printed on the console as standard output.
You can specify an output file with the `-o` parameter.
```
iuextract -i input_file.txt -o output_file.txt
```

### Additional arguments
For additional arguments, such as specifying the separator between the IUs and the index, you can call iuextract with the help argument and get a list of possible arguments.
```
iuextract -h
```