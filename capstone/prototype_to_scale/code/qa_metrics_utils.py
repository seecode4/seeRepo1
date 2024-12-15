# as in https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py
import string
import re
import nltk

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import io
import base64


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
    """ Get BLUEscore-1gram for got_norm_ans"""
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
    """Presnt the percent correct and plot bar chart"""
    df = pd.DataFrame(test_qadata)
    num_rows = df.shape[0]
    num_exact_match = (df[metricname] == 1).sum()
    percent_correct = num_exact_match * 100. / num_rows
    nbins = np.arange(0.0, 1.01, 0.2)
    df['bins'] = pd.cut(df[metricname], nbins, include_lowest=True)
    bin_counts = df['bins'].value_counts().sort_index()
    if dbgval > 0:
        print('--'*10)
        print(f'Num qa: {num_rows}, '
              f'Got correct ans for {num_exact_match} ({percent_correct:.2f}%)')
        print(bin_counts)
        print('--'*10)
    if do_plot > 0:
        bin_counts.plot(kind='bar')
        title_text = f'{metricname} for {file_name}'
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.title(title_text)
        plt.show()
        fname = file_name+'-bar_plot.png'
        plt.savefig(fname)
        with open(fname, "rb") as img_file:
            b64_bytes = base64.b64encode(img_file.read())
            b64_string = b64_bytes.decode('utf-8')
        return bin_counts, b64_string
    else:
        return bin_counts, ""
        # title_str = f'BLUE-1 Scores for {file_name}'
        # plt.hist(df[metricname], bins=5, rwidth=0.5)
        # plt.xlabel('Score')
        # plt.ylabel('Frequency')
        # plt.title(title_str)
        # plt.show()
