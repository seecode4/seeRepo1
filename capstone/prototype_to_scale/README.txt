Capstone Step8 - ML Prototype for Scaling

Use llama3.1 with 8B parameters and try the questions from the triviaqa dataset and use the 'NormalizedAliases' and 'NormalizedValue' in the ‘Answer’ to check for accuracy and evaluate

prototype_triviaqa_llm.ipynb is a Jupyter notebook run in colab. 
The requirements.txt and *.py files in code/ are a step toward docker containerization.

Here packages to be installed are in requirements.txt. Ollama server will be used to pull Large Language Models to query using the triviaqa dataset. In this example we use llama3.1:8b. The dataset used here was download to a local GDrive from the github reference for the paper "TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension".

The main code is in app_ijson_ollama.py as used in the notebook. Usage is as below.
In this example we use llama3.1:8b to answer 100 questions from wikipedia-dev.json in the trivia-qa dataset. The results are captured in test* file. At the end of the notebook is the exercise of using the results to plot (or further analyze) with the answers received.

Usage: app_ijson_ollama.py [-h] [-v VERBOSE] [-d DATAPATH] [-n NUMQUES] [-m MODELNAME] [-e EVAL]

get qa from *.json given datapath

options:
  -h, --help            show this help message and exit
  -v VERBOSE, --verbose VERBOSE
                        do debug prints (default: 0)
  -d DATAPATH, --datapath DATAPATH
                        dir path to .json files (default: ./data/)
  -n NUMQUES, --numques NUMQUES
                        number of questions to try (default: 2)
  -m MODELNAME, --modelname MODELNAME
                        modelname llama3.1:8b, gemma2 or mistral (default: llama3.1:8b)
  -e EVAL, --eval EVAL  evaluate with BLEU-1gram score (default: 1)