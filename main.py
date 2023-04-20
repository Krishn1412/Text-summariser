from flask import Flask,render_template,request
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import PorterStemmer
from nltk.corpus import stopwords 
import math
from statistics import mean
import argparse
from transformers import LEDTokenizer, LEDForConditionalGeneration, LEDConfig
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from rouge_score import rouge_scorer
from rouge import Rouge
nltk.download('punkt')
nltk.download('stopwords')

ref_text=[]
# Abstractive model building
src_text = ("While a number of definitions of artificial intelligence (AI) have surfaced over the last few decades, John McCarthy offers the following definition in this 2004 paper (PDF, 106 KB) (link resides outside IBM), It is the science and engineering of making intelligent machines, especially intelligent computer programs. It is related to the similar task of using computers to understand human intelligence, but AI does not have to confine itself to methods that are biologically observable.However, decades before this definition, the birth of the artificial intelligence conversation was denoted by Alan Turing's seminal work, Computing Machinery and Intelligence (PDF, 89.8 KB) (link resides outside of IBM), which was published in 1950. In this paper, Turing, often referred to as the father of computer science, asks the following question, Can machines think?  From there, he offers a test, now famously known as the Turing Test, where a human interrogator would try to distinguish between a computer and human text response. While this test has undergone much scrutiny since its publish, it remains an important part of the history of AI as well as an ongoing concept within philosophy as it utilizes ideas around linguistics." )
# tokenizer = AutoTokenizer.from_pretrained('t5-base')
# model = AutoModelWithLMHead.from_pretrained('t5-base', return_dict=True)
# text=[]

### T5
tokenizer1 = AutoTokenizer.from_pretrained('t5-base')
model1 = AutoModelWithLMHead.from_pretrained('t5-base', return_dict=True)
text1=[]

### LED
tokenizer2 = LEDTokenizer.from_pretrained('allenai/led-base-16384')
model2 = LEDForConditionalGeneration.from_pretrained('allenai/led-base-16384')
text2=[]

# BART
tokenizer3 = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model3 = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
text3=[]

# Extractive Functions
text4=[]
parser = argparse.ArgumentParser()
parser.add_argument('--filepath', help="File Path", default="articles/article.txt")
args = parser.parse_args()  

def frquency_matrix(sentences):
    matrix = {}
    stop_wrd = stopwords.words("english")
    stemmer = PorterStemmer()

    for sentence in sentences:
        sent_freq = {}
        words = word_tokenize(sentence)
        for word in words:
            word = word.lower()
            word = stemmer.stem(word)
            if word in stop_wrd:
                continue
            elif word in sent_freq.keys():
                sent_freq[word] += 1
            else:
                sent_freq[word] = 1

        matrix[sentence[:10]] = sent_freq
    return matrix


def term_freq_matrix(matrix_freq):
    term_freq = {}
    for s, table in matrix_freq.items():
        sent_table = {}

        word_count_in_sent = len(table)

        for word, freq in table.items():
            sent_table[word] = freq / word_count_in_sent

        term_freq[s] = sent_table

    return term_freq

def total_word_count(matrix_freq):
    total_word_freq = {}
    for sent, wtable in matrix_freq.items():
        for word, count in wtable.items():
            if word in total_word_freq.keys():
                total_word_freq[word] += 1
            else:
                total_word_freq[word] = 1

    return total_word_freq


def idf_matrix(matrix_freq, word_count, num_sent):
    idf = {}

    for sentence, freq in matrix_freq.items():
        idf_sent = {}

        for word in freq.keys():
            idf_sent[word] = math.log10(num_sent/ float(word_count[word]))

        idf[sentence] = idf_sent

    return idf

def tf_idf_matrix(term_freq_mat, matrix_idf):
    matrix_tf_idf = {}

    for (sentence1, tfreq1), (sentence2, tfreq2) in zip(term_freq_mat.items(), matrix_idf.items()):
        tf_idf_sent = {}

        for word, freq in tfreq1.items():
            freq2 = tfreq2[word]
            tf_idf_sent[word] = float(freq * freq2)

        matrix_tf_idf[sentence1] = tf_idf_sent

    return matrix_tf_idf

def listToString(s):
    str1 = ""
    for ele in s:
        str1 += ele
    return str1

def sentence_scores(matrix_tf_idf):
    sent_score = {}

    for sent, tf_idf_matrix in matrix_tf_idf.items():
        total_score = 0
        word_count = len(tf_idf_matrix)

        for word, score in tf_idf_matrix.items():
            total_score = total_score + score

        sent_score[sent] = round(total_score/word_count, 2)

    return sent_score

def find_average_score(scores):
    avg_score = mean(scores[sent] for sent in scores)

    return round(avg_score, 2)

def generate_summary(sentences, scores, threshold):
    count = 0
    summary = ''

    print(round(threshold, 2))

    for sentence in sentences:
        if sentence[:10] in scores and scores[sentence[:10]] >= round(threshold, 2):
            summary = summary + " " + sentence
            count += 1

    return summary




app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')



## T5
@app.route('/abstract2', methods=['POST'])
def abstract2():
    projectpath = request.form['text']
    # print(projectpath)
    inputs = tokenizer1.encode("summarize: " + projectpath,
                          return_tensors='pt',
                          max_length=512,
                          truncation=True)

    summary_ids = model1.generate(inputs, max_length=150, min_length=30, length_penalty=5., num_beams=2)
    data = tokenizer1.decode(summary_ids[0])
    text1=data
    f = open('data.json')
    data1 = json.load(f)
    f.close()
    data1["text1"] = data
    with open("data.json", "w") as jsonFile:
        json.dump(data1, jsonFile)
    return render_template('index1.html',data=data)

## LED
@app.route('/abstract3', methods=['POST'])
def abstract3():
    projectpath = request.form['text']
    print(projectpath)
    inputs = tokenizer2([projectpath], max_length=1024, return_tensors='pt')
    summary_ids = model2.generate(inputs['input_ids'])
    summary = [tokenizer2.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids] 
    data = summary
    text2=data
    f = open('data.json')
    data1 = json.load(f)
    f.close()
    data1["text2"] = data
    with open("data.json", "w") as jsonFile:
        json.dump(data1, jsonFile)
    return render_template('index1.html',data=data)

# BART
@app.route('/abstract4', methods=['POST'])
def abstract4():
    projectpath = request.form['text']
    print(projectpath)
    inputs = tokenizer3([projectpath], max_length=1024, return_tensors='pt')
    summary_ids = model3.generate(inputs['input_ids'])
    summary = [tokenizer3.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    data = summary
    text3=data
    f = open('data.json')
    data1 = json.load(f)
    f.close()
    data1["text3"] = data
    with open("data.json", "w") as jsonFile:
        json.dump(data1, jsonFile)
    return render_template('index1.html',data=data)


@app.route('/extract1', methods=['POST'])
def extract1():
    text = request.form['text']
    sentences = sent_tokenize(text)
    num_sent = len(sentences)
    matrix_freq = frquency_matrix(sentences)
    term_freq_mat = term_freq_matrix(matrix_freq)
    word_count = total_word_count(matrix_freq)
    matrix_idf = idf_matrix(matrix_freq, word_count, num_sent)
    matrix_tf_idf = tf_idf_matrix(term_freq_mat, matrix_idf)
    scores = sentence_scores(matrix_tf_idf)
    avg_score = find_average_score(scores)
    summary = generate_summary(sentences, scores, 0.9*avg_score)
    data=summary
    text4=summary
    # print(text4)
    f = open('data.json')
    data1 = json.load(f)
    f.close()
    data1["text4"] = data
    with open("data.json", "w") as jsonFile:
        json.dump(data1, jsonFile)
    return render_template('index1.html',data=data)

## ROUGE 
@app.route('/rouge', methods=['POST'])
def rouge():
    reference='John really loves data science very much and studies it a lot.'
    candidate='John very much loves data science and enjoys it a lot.'
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    f = open('data.json')
    data1 = json.load(f)
    f.close()
    str1=data1["text1"]
    str2=data1["text2"]
    str3=data1["text3"]
    str4=data1["text4"]
    scores = scorer.score(str1,str4)
    print(scores)
    return render_template('index1.html')

app.run()