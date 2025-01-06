#!/usr/bin/python
# # qa_metrics_utils.py

# utilities to compute qa metrics (BLUE),
# save to files, present on the web with flask and ngrok tunnel

# Ref: https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py

import string
import re
import nltk

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import io
import base64

# for flask and ngrok
import os
import signal
import threading
import flask
from pyngrok import ngrok
from flask import Flask, render_template, render_template_string, request
import getpass
from pyngrok import ngrok, conf


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def get_num_words_in_str(inpstr):
    """Return number of words in given string"""
    num_words = len(inpstr.split())
    return (num_words)


def get_BLEUscore(act_norm_ans, act_norm_aliases, got_norm_ans):
    """ Get BLUEscore-1gram for got_norm_ans; is a value between 0 and 1"""
    if (got_norm_ans == act_norm_ans) or (got_norm_ans in act_norm_aliases):
        # exact match
        return 1
    # Calculate BLEU for only 1-gram overlaps since answers are usually 1-2 words
    # matches 1 word at a time in each alias
    act_aliases_ref = [x.split() for x in act_norm_aliases]
    got_norm_ans_ref = got_norm_ans.split()
    print(f"get_BLEUscore got: {got_norm_ans} act: {act_norm_aliases}")
    bscore = nltk.translate.bleu_score.sentence_bleu(act_aliases_ref,
                                                     got_norm_ans_ref,
                                                     weights=(1, 0, 0, 0))
    return bscore


def display_metrics(test_qadata, metricname, file_name, dbgval, do_plot):
    """Present the percent correct and create bar chart as needed 
    return bin_counts and bar plot as b64 encoded string"""
    df = pd.DataFrame(test_qadata)
    num_rows = df.shape[0]
    num_exact_match = (df[metricname] == 1).sum()

    # percentage of correct answers
    percent_correct = num_exact_match * 100. / num_rows

    # average time to answer a question; this may vary by processor used
    avg_time = df["TimetoAnswer"].astype(float).mean()

    # separate metric values to bins for a bar chart
    nbins = np.arange(0.0, 1.01, 0.2)
    df['bins'] = pd.cut(df[metricname], nbins, include_lowest=True)
    bin_counts = df['bins'].value_counts().sort_index()

    # write metrics to a file
    mfname = file_name + '-metrics.txt'
    with open(mfname, "w") as fname:
        fname.write('----------\n')
        fname.write(f'Num qa: {num_rows}, '
                    f'Got correct ans for {num_exact_match} ({percent_correct:.2f}%) \n')
        fname.write(f'Average TimetoAnswer: {avg_time:.2f}sec. \n')
        fname.write(f'{metricname} \n')
        for index, item in bin_counts.items():
            fname.write(f"{index}: {item}\n")
        fname.write('----------\n')
    if dbgval > 0:
        print('--'*10)
        print(f'Num qa: {num_rows}, '
              f'Got correct ans for {num_exact_match} ({percent_correct:.2f}%) \n')
        print(f'Average TimetoAnswer: {avg_time:.2f}sec. \n')
        print(f'{metricname}', type(bin_counts))
        print(bin_counts)
        print('--'*10)

    if do_plot > 0:
        # save bar plaot to a file
        bin_counts.plot(kind='bar')
        title_text = f'{os.path.basename(file_name)}.json ' + \
            f'{metricname} {percent_correct:.2f}% '
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.title(title_text)
        plt.tight_layout()
        # plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
        plt.show()
        fname = file_name+'-bar_plot.png'
        plt.savefig(fname)
        with open(fname, "rb") as img_file:
            b64_bytes = base64.b64encode(img_file.read())
            b64_string = b64_bytes.decode('utf-8')
        return bin_counts, b64_string
    else:
        return bin_counts, ""


def flask_ngrok_web_url_display(bin_counts, b64_str, file_name, host_ip, port):
    """Use a falsk app and ngrok tunnel to display the ,etrics and bar plot"""

    # auth token in .env file from https://dashboard.ngrok.com/get-started/your-authtoken
    conf.get_default().auth_token = os.getenv("NGROK_AUTHTOKEN")

    # Open a ngrok tunnel to the HTTP server
    # port = 5000
    public_url = ngrok.connect(port).public_url
    url_info = f'* ngrok tunnel "{public_url}" -> "http://127.0.0.1:{port}"'
    # print(' * ngrok tunnel "{}" -> "http://127.0.0.1:{}"'.format(public_url, port))
    fname = file_name + '-url.txt'
    with open(fname, "w") as fname:
        fname.write(url_info)
    print(url_info)

    # Create Flask app specifying template_folder
    os.environ["FLASK_DEBUG"] = "development"
    cwd = os.getcwd()
    print("Flask current working dir is", cwd)
    app = Flask(__name__, template_folder='./templates')

    # Update any base URLs to use the public ngrok URL
    app.config["BASE_URL"] = public_url

    @app.route('/', methods=['GET', 'POST'])
    def index():
        # chart_url = None
        # b64_str = None
        print("request.method:", request.method)
        return render_template("index.html", chart_url=b64_str)

    @app.route("/bincounts")
    def bincounts():
        return bin_counts.to_html()

    @app.route("/hello")
    def hello():
        hstr = "Hello from Flask!"
        return hstr

    # Start the Flask server in a new thread; host_ip not needed for local runs
    if host_ip:
        threading.Thread(target=app.run, kwargs={
                         "host": host_ip, "use_reloader": False}).start()
    else:
        threading.Thread(target=app.run, kwargs={
                         "use_reloader": False}).start()

    return public_url
