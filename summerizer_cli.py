import sys
import argparse
import re
import pyfiglet
import warnings
import pyperclip
import json

from string import punctuation
from collections import Counter
from heapq import nlargest

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from transformers import pipeline, logging

from llm_rest_client import ContextClient, LLMClient


def summerize(stype, text: str, max_len):
    if stype == 'abstractive':
        return summerize_abstractive(text, max_len)

    elif stype == 'extractive':
        return summerize_extractive(text)

    elif stype == 'llm':
        return summarize_llm(text)

    return text


def summerize_extractive(text: str) -> str:
    global nlp
    doc = nlp(text)

    keyword = []
    stopwords = list(STOP_WORDS)
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    for token in doc:
        if token.text in stopwords or token.text in punctuation:
            continue
        if token.pos_ in pos_tag:
            keyword.append(token.text)

    freq_word = Counter(keyword)

    max_freq = Counter(keyword).most_common(1)[0][1]
    for word in freq_word.keys():
        freq_word[word] = int(freq_word[word] / max_freq)
    freq_word.most_common(5)

    sent_strength = {}
    for sent in doc.sents:
        for word in sent:
            if word.text in freq_word.keys():
                if sent in sent_strength.keys():
                    sent_strength[sent] += freq_word[word.text]
                else:
                    sent_strength[sent] = freq_word[word.text]

    summarized_sentences = nlargest(3, sent_strength, key=sent_strength.get)
    final_sentences = [w.text for w in summarized_sentences]
    summary = ' '.join(final_sentences)
    return summary


def summerize_abstractive(text: str, max_len) -> str:
    global summarizer

    text_sum = summarizer(text, min_length=10, max_length=max_len, truncation=True)
    summary = ' '.join([i['summary_text'] for i in text_sum])
    return summary


def summarize_llm(text: str):
    global llm_client

    con_client = ContextClient("summarizer", args['host'], args['port'])
    status = con_client.create_context(history=0, system_prompt="summarize the following text.")
    if status[0]:
        resp_gen, _resp_id = con_client.submit_directive(text)
        return resp_gen
    else:
        return None


def print_summarized(stype, summarized_text):
    print("*** Summarization using '{}' method ***".format(stype))
    if isinstance(summarized_text, str):
        print(summarized_text)
        print()

    elif summarized_text is not None:
        max_line_len = 120
        current_line_len = 0
        for sword in summarized_text:
            wd = sword.replace("\n", "")
            if not filter_words(wd):
                current_line_len += len(wd) + 1
                if current_line_len > max_line_len:
                    print('\n', end='', flush=True)
                    current_line_len = 0
                print(wd + " ", end="", flush=True)
    else:
        print("Not summarized")


def display_model_info():
    # Display LLM model information
    print("Current model information:")
    info = llm_client.get_model_info()[2]
    print(json.dumps(info, indent=4))
    print()


def filter_words(theword):
    filter_these = ['response:', 'ai:', 'answer:']
    return any(theword.lower().startswith(w) for w in filter_these)


def match_marker(string):
    pattern = r'^\|[^|-]+-[^|-]+\|$'
    return re.match(pattern, string) is not None


# Hack to fix warning bug in pipeline
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

if __name__ == "__main__":
    print(pyfiglet.figlet_format("Summarizer"))

    ap = argparse.ArgumentParser()
    ap.add_argument("file", default="paste", help="File to summerize")
    ap.add_argument("-s", "--type", default="all", help="Summarize type")
    ap.add_argument("-m", "--max", type=int, default=100, help="max size")
    ap.add_argument("-t", "--host", type=str, default="ai-001.local", help="server host")
    ap.add_argument("-p", "--port", type=int, default=8080, help="server port")
    args = vars(ap.parse_args())

    if len(args) == 0 or args["file"] == "paste":
        # From clipboard
        raw_text = pyperclip.paste().strip()
        print("Text from clipboard:")
    else:
        # from text file
        try:
            f = open(args['file'], 'r')
            raw_text = f.read()
            f.close()
            print("Text from file '{}'".format(args['file']))
        except FileNotFoundError:
            print("File '{}' not found.".format(args['file']))
            sys.exit(1)

    print(raw_text.strip())
    print()

    # Set up the history summerizer
    nlp = None
    summarizer = None
    summarizer_type = args["type"]

    valid_type = False
    if summarizer_type == 'abstractive' or summarizer_type == 'all':
        logging.set_verbosity_error()
        summarizer = pipeline("summarization")
        summerized = summerize('abstractive', raw_text, args['max'])
        print_summarized('abstractive', summerized)
        valid_type = True

    if summarizer_type == 'extractive' or summarizer_type == 'all':
        nlp = spacy.load('en_core_web_sm')
        summerized = summerize('extractive', raw_text, args['max'])
        print_summarized('extractive', summerized)
        valid_type = True

    if summarizer_type == 'llm' or summarizer_type == 'all':
        llm_client = LLMClient(args['host'], args['port'])
        summarized = summerize('llm', raw_text, args['max'])
        print_summarized('LLM', summarized)

        valid_type = True

    if not valid_type:
        print("Unknown summarizer type '{}'".format(args['type']))

    print()
