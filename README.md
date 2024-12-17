# IUExtract
Rule-based Idea Unit segmentation algorithm for the English language.

## Example Segmentation
```
My dog, Chippy, just won its first grooming competition.
```
Will be segmented into the following Idea Units:
```
D1|My dog,
2|Chippy,
D1|just
```
Each line denotes a segment. At the beginning of each line there is an Idea Unit index. Each Unit is assigned an index in sequential order. Discontinuous Units are prefixed by the character "D". Naturally, these indexes can be found on multiple lines and the complete Idea Unit can be obtained by joining the lines with the same index.
## Installation
### Installation as standalone executable via pipx
To install the package as a command line tool first install pipx. Specific instructions for your operative system can be found [here](https://pipx.pypa.io/latest/installation/).
If you have python installed, you can install pipx with the following commands:
```
python3 -m pip install -U pipx
python3 -m pipx ensurepath
```
After pipx is installed, you can install IUExtract with the following command:
```
pipx install iuextract
```
If the install fails you might want to try to pin a specific python version with the following command:
```
pipx install iuextract --python 3.9
```
**Note:** on first run, the program will download the Spacy model `en_core_web_lg`. This could take some time. A custom Spacy model can be selected if you install iuextract as a python module.

### Installation as a python module
If you want to use IUExtract in your python projects you will need to install it as a regular python module.
First of all, you need to install the dependencies:
```
pip install spacy
python -m spacy download en_core_web_lg
```
You can then install IUExtract.
```
pip install iuextract
```
## Command Line Interface (CLI) Usage
Once installed via `pipx`, you can run iuextract directly from the CLI.

Example:
```
iuextract My dog, Chippy, just won its first grooming competition.
```
will output
```
D1|My dog,
2|Chippy,
D1|just won its first grooming competition.
```
If you installed iuextract as a python module, you can still run the program via CLI with the following command:
```
python -m iuextract My dog, Chippy, just won its first grooming competition.
```

**Note:** When running from CLI, all positional arguments are grouped into a single string and parsed as input text. If you need to use named arguments put them before the input text or use the `-i` argument to parse a file as input.
### Input text from file
You can run iuextract with the `-i` argument to parse a file.
For example
```
iuextract -i input_file.txt
```
will read `input_file.txt` from the working directory and output the segmentation to the console.

### Output file
You can specify an output file with the `-o` parameter.
```
iuextract -i input_file.txt -o output_file.txt
```
This command will segment `input_file.txt` and put the resulting segmentation into `output_file.txt`.

### Additional arguments
For additional arguments, such as specifying the separator between the IUs and the index, you can call iuextract with the help argument and get a list of possible arguments.
```
iuextract -h
```

## Usage as module

Simple text segmentation:
```
from iuextract.extract import segment_ius

text = "My dog, Chippy, just won its first grooming competition."
print(segment_ius(text, mode='str'))
```
```
D1|My dog, 
2|Chippy, 
D1|just won its first grooming competition.
```

For more examples check `example.ipynb`