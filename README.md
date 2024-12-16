# IUExtract
Rule-based Idea Unit segmentation algorithm for the English language.

## Installation
To install the package with the command line tool [install pipx](https://pipx.pypa.io/latest/installation/) and run the following command:
```
pipx install iuextract
```

If you only wish to use the package in your python projects you can install without executable via
```
pip install iuextract
```

## Command Line Interface (CLI) Usage
Once installed via `pipx`, you can run
```
iuextract -i input_file.txt 
```
to segment `file.txt` into Idea Units. The segmented file will be printed on the console as standard output.
You can specify an output file with the `-o` parameter.
```
iuextract -i input_file.txt -o output_file.txt
```
For more options you can call the help argument.
```
iuextract -h
```