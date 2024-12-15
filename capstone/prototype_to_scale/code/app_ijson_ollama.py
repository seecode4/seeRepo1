#!/usr/bin/python
# app_ijson_ollama.py

# Following example is based on the following link
# https://www.fahdmirza.com/2024/07/run-llama-31-with-ollama-and-google.html

# To install needed packages, run the following command in your terminal:
#   pip install --no-cache-dir -r requirements.txt
#   curl -fsSL https://ollama.com/install.sh | sh # download ollama api
from IPython.display import clear_output, Markdown, display

import os
import sys
import argparse
import json
import ijson
import tensorflow as tf

import triviaqa_ijson_utils
from triviaqa_ijson_utils import create_datsets_dict_list, print_sample_qa
from triviaqa_ijson_utils import dbg_print_qa_files_info, write_dict_list_to_json_file

import qa_metrics_utils
from qa_metrics_utils import normalize_answer, get_num_words_in_str, get_BLEUscore, display_metrics

# Following imports needed for running Ollama Client with llama model
import os
import signal
import threading
import subprocess
import requests
import nltk
import ollama
import time
import textwrap
import pprint

# pp = pprint.PrettyPrinter(indent=1)

# Install the following with a bash script or from terminal
# !pip install ijson
# !pip install ollama
# !curl -fsSL https://ollama.com/install.sh | sh # download ollama api


def ollama_process():
    """start ollama server"""
    # These environment variables need to be set in the shell environment, not just within the subprocess.
    # os.environ['OLLAMA_HOST'] = '0.0.0.0:11434'
    # os.environ['OLLAMA_ORIGINS'] = '*'
    try:
        process = subprocess.Popen(["ollama", "serve"])
        return process
    except subprocess.CalledProcessError as e:
        print(f"Error starting ollama server: {e}")
        return None


def stop_ollama_server_not_on_colab():
    """POST request to the /api/shutdown endpoint to stop ollama server"""
    # Msg: System has not been booted with systemd as init system (PID 1). Can't operate.
    try:
        requests.post("http://localhost:11434/api/shutdown")
        print("Ollama server stopped successfully.")
    except requests.exceptions.RequestException as e:
        print("Error stopping Ollama server:", e)


def stop_ollama_server():
    try:
        subprocess.run(["systemctl", "stop", "ollama.service"])
        print("Ollama server stopped successfully.")
    except subprocess.CalledProcessError as e:
        print("Error stopping Ollama server:", e)


def ollama_action_model(model_name, actionstr, test_path):
    """Pulls an Ollama model using the subprocess module."""

    try:
        ofname = test_path + "output_ollama_" + actionstr + ".txt"
        with open(ofname, 'w') as ofile:
            subprocess.run(["ollama", actionstr, model_name],
                           stdout=ofile, check=True)
        print(f"\nDone ollama {model_name} {actionstr}")
    except subprocess.CalledProcessError as e:
        print(f"\nError ollama {model_name} {actionstr}: {e}")


def ask_llm_model(ollama_client, question, model="llama3.1:8b"):
    # ollama_client = ollama.Client(base_url="http://localhost:11434")
    ans = ollama_client.generate(
        model=model,
        prompt=question
    )
    if ans:
        return ans['response']
    else:
        return 'Got No Answer'


def get_answers_from_llm(qa_dict_list, num, model="llama3.1:8b"):
    """ Query llama3.1 for num questions from qa_dict_list add columns GotAnswer and TimetoAnswer"""
    # pip install ollama
    ollama_client = ollama.Client()

    cnt = min(len(qa_dict_list), num)
    print(f'Getting answers for {cnt} questions')
    for i in range(cnt):
        testqa = qa_dict_list[i]
        # print(f'{i+1:05}', testqa['Question'], "...", testqa['Value'])
        t_start = time.time()
        # Get the answer to question from LLM
        nwords_exp = get_num_words_in_str(testqa['NormalizedValue'])
        ques_to_ask = f"In about {nwords_exp} words, {testqa['Question']}"
        output = ask_llm_model(ollama_client, ques_to_ask, model)
        t_end = time.time()
        t_ans_sec = t_end - t_start
        testqa['GotAnswer'] = output
        testqa['TimetoAnswer'] = f'{t_ans_sec:.3f}'
    return qa_dict_list


def get_qa_metrics_bleu1_score(qa_dict_list, num):
    """ Get BLUE-1 score and add GotNormalizedValue, MetricBLEUscore to qa_dict_list"""

    cnt = min(len(qa_dict_list), num)
    print(f'Getting answers for {cnt} questions')
    for i in range(cnt):
        qatest = qa_dict_list[i]
        qatest['GotNormalizedValue'] = normalize_answer(qatest['GotAnswer'])
        qatest['MetricBLEUscore'] = get_BLEUscore(qatest['NormalizedValue'],
                                                  qatest['NormalizedAliases'],
                                                  qatest['GotNormalizedValue'])
    return qa_dict_list


def main():
    global dbglevel
    data_path = './'
    numq = 2  # number of queries to try
    print("platform version", sys.platform)
    print("python version", sys.version)
    print("ijson version:", ijson.__version__)

    # print("My colab data_path = './data_local/triviaqa-rc/qa/'")
    print("Current working directory: ", os.getcwd())

    # Create the parser
    parser = argparse.ArgumentParser(
        description='get qa from *.json given datapath',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("-v", "--verbose", action='store_true', help='do debug prints')
    parser.add_argument("-v", "--verbose", type=int, default=0,
                        help='do debug prints')
    parser.add_argument("-d", "--datapath", default='./data/',
                        help='dir path to .json files')
    parser.add_argument("-n", "--numques", type=int, default=2,
                        help='number of questions to try')
    parser.add_argument("-m", "--modelname", default="llama3.1:8b",
                        help='modelname llama3.1:8b, gemma2 or mistral')
    parser.add_argument("-e", "--eval", type=int, default=1,
                        help='evaluate with BLEU-1gram score')
    # Parse the arguments
    args = parser.parse_args()
    data_path = args.datapath
    numq = args.numques
    model_name = args.modelname
    do_eval = args.eval
    dbglevel = 0
    if args.verbose:
        dbglevel = 1  # just one level for now args.verbose
    # if args.eval:
    #     do_eval = 1
    print(f"Running {sys.argv[0]} -v {dbglevel} -d {data_path} -n {numq}",
          f"-m {model_name} -e {do_eval}")

    datsets_dict_list = create_datsets_dict_list(data_path)
    if datsets_dict_list == []:
        print(f'No qa data in {data_path}')
        sys.exit(0)

    if dbglevel > 0:
        print('--' * 10)
        dbg_print_qa_files_info(datsets_dict_list)

    cwd = os.getcwd()
    test_path = os.path.join(cwd, "test/")
    if not os.path.exists(test_path):
        os.mkdir(test_path)
        print("Test path:", test_path)
    else:
        print(f"{test_path} already exists.")

    # Use GPU if present
    if tf.test.is_gpu_available():
        print("GPU is available")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        print("GPU is not available")
    # Update environment variable within the shell session
    os.environ['OLLAMA_HOST'] = '0.0.0.0:11434'
    os.environ['OLLAMA_ORIGINS'] = '*'

    # Ensure Ollama server is running and accessible before proceeding.
    # stop_event = threading.Event()
    # ollama_thread = threading.Thread(target=ollama_process, args=(stop_event,))
    # ollama_thread = threading.Thread(target=ollama_process)
    process = ollama_process()
    if process == None:
        sys.exit()
    ollama_thread = threading.Thread(target=process)
    ollama_thread.start()

    # Allow some time for the server to start before attempting to connect
    time.sleep(5)  # Wait for 5 seconds
    if dbglevel > 0:
        print("\nStarted ollama process")

    # from IPython.display import clear_output
    # !ollama pull llama3.1:8b
    # clear_output()
    # Get llm model
    # model_name = "llama3.1:8b"
    ollama_action_model(model_name, "pull", test_path)
    time.sleep(5)  # Wait for 5 seconds

    ollama_action_model("", "list", test_path)
    time.sleep(5)  # Wait for 5 seconds

    # from ollama import Ollama
    # ollama = Ollama()
    num_files = len(datsets_dict_list)
    for i in range(num_files):
        qa_filename = datsets_dict_list[i]['filename']
        curr_qa_list = datsets_dict_list[i]['qa_dict_list']
        print(f"Num qa in curr_qa_list = {len(curr_qa_list)}")
        # Get answers to numq number of questions from curr_qa_list
        qa_dict_list = get_answers_from_llm(curr_qa_list, numq, model_name)

        # print example qa with llm's answer
        pp = pprint.PrettyPrinter(width=101, compact=True)
        # pp.pprint(curr_qa_list[0])
        pp.pprint(qa_dict_list[0])

        if dbglevel > 0:
            print(f"\nDone llm Queries for {qa_filename} ....")

        if (do_eval > 0):
            get_qa_metrics_bleu1_score(qa_dict_list, numq)

        if dbglevel > 0:
            print(f"\nDone qa evaluation for {qa_filename} ....")

        # Write the test answers to a JSON file to use for evaluation metrics
        # ans_file = data_path + "/test_wikipedia_dev_qa.json"

        ans_file = test_path + "test-" + model_name + "-" + qa_filename
        maxnumq = min(numq, len(qa_dict_list))
        do_plot = 1
        bin_counts, b64_string = display_metrics(qa_dict_list[:maxnumq], 'MetricBLEUscore',
                                                 qa_filename, dbglevel, do_plot)
        # dict_list = wikipedia_dev_qa_list
        write_dict_list_to_json_file(ans_file, qa_dict_list, maxnumq)
        time.sleep(5)  # Wait for 5 seconds

    # remove llm model to return resources
    ollama_action_model(model_name, "rm", test_path)
    time.sleep(5)  # Wait for 5 seconds

    # stop ollama server
    # ollama_action_model("", "stop", test_path)
    stop_ollama_server()
    time.sleep(5)  # Wait for 5 seconds
    print("\nStopping ollama server..")

    # stop_event.set()  # Signal the thread to stop
    ollama_thread.join()  # Wait for the thread to finish
    print("\nStopping ollama thread.. ")
    print("ps and delete ollam process if running again")
    print(f'copy {ans_file} to local drive space')
    time.sleep(5)  # Wait for 5 seconds

    # Terminate the process
    # os.kill(os.getpid(), signal.SIGTERM)
    process.terminate()


if __name__ == "__main__":
    # name_of_script = sys.argv[0]
    dbglevel = 0
    data_path = './'
    numq = 2
    model_name = "llama3.1:8b"
    do_eval = 0
    main()
