# Ads Detection Project

usage: main.py [-h] [--input_file INPUT_FILE] [--output_file OUTPUT_FILE]

Test task solution. For the program to work correctly, you need to enter a file with with one sentence per line. The output is another file with one predicted       
label per line.

optional arguments:

  -h, --help            show this help message and exit
  
  --input_file INPUT_FILE
                        path to input file in format csv
                        
  --output_file OUTPUT_FILE
                        path to output file



For the script to work correctly, you still need to add the binary to the models folder, link to the binary:
https://drive.google.com/file/d/1k19wemFn3POS1iJCjVENhISwmD3n9uo_/view?usp=sharing 
The main idea that I implemented is to divide all the sentences in the text into 2 classes-advertising and the rest. After that, I did fine-tuning of the existing Bert model "bert-base-uncased".

I got the following results on the test set:

accuracy: 0.96

precision: 0.74

recall: 0.40

MCC: 0.528

Examples of ideas that have not been implemented, but I want to be implemented in the future:
1) Try multiclass classification
2) Use heuristics (for example ".com" often a mark of ad)
3) Consider the location of sentences. If both of the closest sentences in the text have the opposite label, change the label of this sentence.
