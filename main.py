from flask import Flask,render_template,request,send_file
from transformers import AutoTokenizer, AutoModelWithLMHead
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
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
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from io import BytesIO

def preprocess_text(text):
    # Tokenize text into sentences and words
    sentences = sent_tokenize(text)
    words = [word_tokenize(sentence) for sentence in sentences]

    # Flatten the list of words and convert to lowercase
    words = [word.lower() for sentence in words for word in sentence]

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word.isalnum() and word not in stop_words]

    return words, sentences

def build_similarity_matrix(sentences, words):
    # Build a similarity matrix based on word overlap
    vectorizer = CountVectorizer(vocabulary=list(set(words)))
    word_counts = vectorizer.fit_transform(sentences)
    similarity_matrix = word_counts.dot(word_counts.T)
    similarity_matrix = normalize(similarity_matrix, axis=1)

    return similarity_matrix

def pagerank(similarity_matrix, d=0.85, max_iter=100, tol=1e-4):
    n = similarity_matrix.shape[0]
    scores = np.ones(n) / n
    for i in range(max_iter):
        prev_scores = scores.copy()
        for j in range(n):
            scores[j] = (1 - d) + d * np.sum(similarity_matrix[j, :] * prev_scores, axis=0)
        if np.linalg.norm(scores - prev_scores) < tol:
            break
    return scores


def text_summarization(text, num_sentences=3):
    words, sentences = preprocess_text(text)
    similarity_matrix = build_similarity_matrix(sentences, words)
    scores = pagerank(similarity_matrix)
    sorted_indices = np.argsort(scores)[::-1][:num_sentences]
    summary = [sentences[i] for i in sorted_indices]

    return " ".join(summary)

def calculate_rouge_scores(reference, summary):
    rouge = Rouge()
    scores = rouge.get_scores(summary, reference)
    return scores


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
    data=data[5:-4]
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
    data = summary[0]
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
    data = summary[0]
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


@app.route('/extract2', methods=['POST'])
def extract2():
    text = request.form['text']
    summary = text_summarization(text, num_sentences=2)
    data=summary
    f = open('data.json')
    data1 = json.load(f)
    f.close()
    data1["text5"] = data
    with open("data.json", "w") as jsonFile:
        json.dump(data1, jsonFile)
    return render_template('index1.html',data=data)

## ROUGE 
@app.route('/plot1', methods=['GET'])
def plot1():
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
    str5=data1["text5"]
    ref=data1["ref"]
    
    score1 = calculate_rouge_scores(ref, str1)
    score2 = calculate_rouge_scores(ref, str2)
    score3 = calculate_rouge_scores(ref, str3)
    score4 = calculate_rouge_scores(ref, str4)
    score5 = calculate_rouge_scores(ref, str5)
    
    rouge1_score1 = score1[0]['rouge-1']['f']
    rouge2_score1 = score1[0]['rouge-2']['f']
    rougeL_score1 = score1[0]['rouge-l']['f']
    
    rouge1_score2 = score2[0]['rouge-1']['f']
    rouge2_score2 = score2[0]['rouge-2']['f']
    rougeL_score2 = score2[0]['rouge-l']['f']
    
    rouge1_score3 = score3[0]['rouge-1']['f']
    rouge2_score3 = score3[0]['rouge-2']['f']
    rougeL_score3 = score3[0]['rouge-l']['f']
    
    rouge1_score4 = score4[0]['rouge-1']['f']
    rouge2_score4 = score4[0]['rouge-2']['f']
    rougeL_score4 = score4[0]['rouge-l']['f']
    
    rouge1_score5 = score5[0]['rouge-1']['f']
    rouge2_score5 = score5[0]['rouge-2']['f']
    rougeL_score5 = score5[0]['rouge-l']['f']

    summarizer_names = ['T5', 'LED', 'BART','TF-IDF','PAGE-RANK']
    summary_lengths = [len(str1), len(str2), len(str3),len(str4),len(str5)]
    rouge1_scores = [rouge1_score1, rouge1_score2, rouge1_score3,rouge1_score4,rouge1_score5]
    rouge2_scores = [rouge2_score1, rouge2_score2, rouge2_score3,rouge2_score4,rouge2_score5]
    rougeL_scores = [rougeL_score1, rougeL_score2, rougeL_score3,rougeL_score4,rougeL_score5]

    # Bar Chart for Summary Lengths
    plt.figure(figsize=(8, 6))
    sns.barplot(x=summarizer_names, y=summary_lengths)
    plt.title('Summary Lengths by Summarizer')
    plt.xlabel('Summarizer')
    plt.ylabel('Summary Length')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    # plt.show()
    plt.clf()
    return send_file(buffer,mimetype='image/png')


@app.route('/plot2', methods=['GET'])
def plot2():
    f = open('data.json')
    data1 = json.load(f)
    f.close()
    str1=data1["text1"]
    str2=data1["text2"]
    str3=data1["text3"]
    str4=data1["text4"]
    str5=data1["text5"]
    ref=data1["ref"]
    
    score1 = calculate_rouge_scores(ref, str1)
    score2 = calculate_rouge_scores(ref, str2)
    score3 = calculate_rouge_scores(ref, str3)
    score4 = calculate_rouge_scores(ref, str4)
    score5 = calculate_rouge_scores(ref, str5)
    
    rouge1_score1 = score1[0]['rouge-1']['f']
    rouge2_score1 = score1[0]['rouge-2']['f']
    rougeL_score1 = score1[0]['rouge-l']['f']
    
    rouge1_score2 = score2[0]['rouge-1']['f']
    rouge2_score2 = score2[0]['rouge-2']['f']
    rougeL_score2 = score2[0]['rouge-l']['f']
    
    rouge1_score3 = score3[0]['rouge-1']['f']
    rouge2_score3 = score3[0]['rouge-2']['f']
    rougeL_score3 = score3[0]['rouge-l']['f']
    
    rouge1_score4 = score4[0]['rouge-1']['f']
    rouge2_score4 = score4[0]['rouge-2']['f']
    rougeL_score4 = score4[0]['rouge-l']['f']
    
    rouge1_score5 = score5[0]['rouge-1']['f']
    rouge2_score5 = score5[0]['rouge-2']['f']
    rougeL_score5 = score5[0]['rouge-l']['f']
    
    
    summarizer_names = ['T5', 'LED', 'BART','TF-IDF','PAGE-RANK']
    summary_lengths = [len(str1), len(str2), len(str3),len(str4),len(str5)]
    rouge1_scores = [rouge1_score1, rouge1_score2, rouge1_score3,rouge1_score4,rouge1_score5]
    rouge2_scores = [rouge2_score1, rouge2_score2, rouge2_score3,rouge2_score4,rouge2_score5]
    rougeL_scores = [rougeL_score1, rougeL_score2, rougeL_score3,rougeL_score4,rougeL_score5]

    # Line Chart for ROUGE Scores
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=summarizer_names, y=rouge1_scores)
    plt.title('ROUGE Scores by Summarizer')
    plt.xlabel('Summarizer')
    plt.ylabel('ROUGE-1 Score')
    # plt.show()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    # Clear the plot
    plt.clf()
    # Return the plot as an image file
    return send_file(buffer, mimetype='image/png')

@app.route('/plot3', methods=['GET'])
def plot3():
    f = open('data.json')
    data1 = json.load(f)
    f.close()
    str1=data1["text1"]
    str2=data1["text2"]
    str3=data1["text3"]
    str4=data1["text4"]
    str5=data1["text5"]
    ref=data1["ref"]
    
    
    score1 = calculate_rouge_scores(ref, str1)
    score2 = calculate_rouge_scores(ref, str2)
    score3 = calculate_rouge_scores(ref, str3)
    score4 = calculate_rouge_scores(ref, str4)
    score5 = calculate_rouge_scores(ref, str5)
    
    rouge1_score1 = score1[0]['rouge-1']['f']
    rouge2_score1 = score1[0]['rouge-2']['f']
    rougeL_score1 = score1[0]['rouge-l']['f']
    
    rouge1_score2 = score2[0]['rouge-1']['f']
    rouge2_score2 = score2[0]['rouge-2']['f']
    rougeL_score2 = score2[0]['rouge-l']['f']
    
    rouge1_score3 = score3[0]['rouge-1']['f']
    rouge2_score3 = score3[0]['rouge-2']['f']
    rougeL_score3 = score3[0]['rouge-l']['f']
    
    rouge1_score4 = score4[0]['rouge-1']['f']
    rouge2_score4 = score4[0]['rouge-2']['f']
    rougeL_score4 = score4[0]['rouge-l']['f']
    
    rouge1_score5 = score5[0]['rouge-1']['f']
    rouge2_score5 = score5[0]['rouge-2']['f']
    rougeL_score5 = score5[0]['rouge-l']['f']
    
    summarizer_names = ['T5', 'LED', 'BART','TF-IDF','PAGE-RANK']
    summary_lengths = [len(str1), len(str2), len(str3),len(str4),len(str5)]
    rouge1_scores = [rouge1_score1, rouge1_score2, rouge1_score3,rouge1_score4,rouge1_score5]
    rouge2_scores = [rouge2_score1, rouge2_score2, rouge2_score3,rouge2_score4,rouge2_score5]
    rougeL_scores = [rougeL_score1, rougeL_score2, rougeL_score3,rougeL_score4,rougeL_score5]

    # Line Chart for ROUGE Scores
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=summarizer_names, y=rouge2_scores)
    plt.title('ROUGE Scores by Summarizer')
    plt.xlabel('Summarizer')
    plt.ylabel('ROUGE-2 Score')
    # plt.show()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.clf()
    return send_file(buffer,mimetype='image/png')

@app.route('/plot4', methods=['GET'])
def plot4():
    f = open('data.json')
    data1 = json.load(f)
    f.close()
    str1=data1["text1"]
    str2=data1["text2"]
    str3=data1["text3"]
    str4=data1["text4"]
    str5=data1["text5"]
    ref=data1["ref"]
    
    score1 = calculate_rouge_scores(ref, str1)
    score2 = calculate_rouge_scores(ref, str2)
    score3 = calculate_rouge_scores(ref, str3)
    score4 = calculate_rouge_scores(ref, str4)
    score5 = calculate_rouge_scores(ref, str5)
    
    rouge1_score1 = score1[0]['rouge-1']['f']
    rouge2_score1 = score1[0]['rouge-2']['f']
    rougeL_score1 = score1[0]['rouge-l']['f']
    
    rouge1_score2 = score2[0]['rouge-1']['f']
    rouge2_score2 = score2[0]['rouge-2']['f']
    rougeL_score2 = score2[0]['rouge-l']['f']
    
    rouge1_score3 = score3[0]['rouge-1']['f']
    rouge2_score3 = score3[0]['rouge-2']['f']
    rougeL_score3 = score3[0]['rouge-l']['f']
    
    rouge1_score4 = score4[0]['rouge-1']['f']
    rouge2_score4 = score4[0]['rouge-2']['f']
    rougeL_score4 = score4[0]['rouge-l']['f']
    
    rouge1_score5 = score5[0]['rouge-1']['f']
    rouge2_score5 = score5[0]['rouge-2']['f']
    rougeL_score5 = score5[0]['rouge-l']['f']
    
    
    summarizer_names = ['T5', 'LED', 'BART','TF-IDF','PAGE-RANK']
    summary_lengths = [len(str1), len(str2), len(str3),len(str4),len(str5)]
    rouge1_scores = [rouge1_score1, rouge1_score2, rouge1_score3,rouge1_score4,rouge1_score5]
    rouge2_scores = [rouge2_score1, rouge2_score2, rouge2_score3,rouge2_score4,rouge2_score5]
    rougeL_scores = [rougeL_score1, rougeL_score2, rougeL_score3,rougeL_score4,rougeL_score5]

    # Box Plot for Summary Lengths
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=summarizer_names, y=summary_lengths)
    plt.title('Summary Lengths Distribution by Summarizer')
    plt.xlabel('Summarizer')
    plt.ylabel('Summary Length')
    # plt.show()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    # Clear the plot
    plt.clf()
    # Return the plot as an image file
    return send_file(buffer, mimetype='image/png')

@app.route('/plot5', methods=['GET'])
def plot5():
    f = open('data.json')
    data1 = json.load(f)
    f.close()
    str1=data1["text1"]
    str2=data1["text2"]
    str3=data1["text3"]
    str4=data1["text4"]
    str5=data1["text5"]
    ref=data1["ref"]
    
    score1 = calculate_rouge_scores(ref, str1)
    score2 = calculate_rouge_scores(ref, str2)
    score3 = calculate_rouge_scores(ref, str3)
    score4 = calculate_rouge_scores(ref, str4)
    score5 = calculate_rouge_scores(ref, str5)
    
    rouge1_score1 = score1[0]['rouge-1']['f']
    rouge2_score1 = score1[0]['rouge-2']['f']
    rougeL_score1 = score1[0]['rouge-l']['f']
    
    rouge1_score2 = score2[0]['rouge-1']['f']
    rouge2_score2 = score2[0]['rouge-2']['f']
    rougeL_score2 = score2[0]['rouge-l']['f']
    
    rouge1_score3 = score3[0]['rouge-1']['f']
    rouge2_score3 = score3[0]['rouge-2']['f']
    rougeL_score3 = score3[0]['rouge-l']['f']
    
    rouge1_score4 = score4[0]['rouge-1']['f']
    rouge2_score4 = score4[0]['rouge-2']['f']
    rougeL_score4 = score4[0]['rouge-l']['f']
    
    rouge1_score5 = score5[0]['rouge-1']['f']
    rouge2_score5 = score5[0]['rouge-2']['f']
    rougeL_score5 = score5[0]['rouge-l']['f']
    
    summarizer_names = ['T5', 'LED', 'BART','TF-IDF','PAGE-RANK']
    summary_lengths = [len(str1), len(str2), len(str3),len(str4),len(str5)]
    rougeL_scores = [rougeL_score1, rougeL_score2, rougeL_score3,rougeL_score4,rougeL_score5]

    # Line Chart for ROUGE Scores
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=summarizer_names, y=rougeL_scores)
    plt.title('ROUGE Scores by Summarizer')
    plt.xlabel('Summarizer')
    plt.ylabel('ROUGE-L Score')
    # plt.show()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    # Clear the plot
    plt.clf()
    # Return the plot as an image file
    return send_file(buffer, mimetype='image/png')

@app.route('/plot6', methods=['GET'])
def plot6():
    f = open('data.json')
    data1 = json.load(f)
    f.close()
    str1=data1["text1"]
    str2=data1["text2"]
    str3=data1["text3"]
    str4=data1["text4"]
    str5=data1["text5"]
    ref=data1["ref"]
    
    score1 = calculate_rouge_scores(ref, str1)
    score2 = calculate_rouge_scores(ref, str2)
    score3 = calculate_rouge_scores(ref, str3)
    score4 = calculate_rouge_scores(ref, str4)
    score5 = calculate_rouge_scores(ref, str5)
    
    rouge1_score1 = score1[0]['rouge-1']['f']
    rouge2_score1 = score1[0]['rouge-2']['f']
    rougeL_score1 = score1[0]['rouge-l']['f']
    
    rouge1_score2 = score2[0]['rouge-1']['f']
    rouge2_score2 = score2[0]['rouge-2']['f']
    rougeL_score2 = score2[0]['rouge-l']['f']
    
    rouge1_score3 = score3[0]['rouge-1']['f']
    rouge2_score3 = score3[0]['rouge-2']['f']
    rougeL_score3 = score3[0]['rouge-l']['f']
    
    rouge1_score4 = score4[0]['rouge-1']['f']
    rouge2_score4 = score4[0]['rouge-2']['f']
    rougeL_score4 = score4[0]['rouge-l']['f']
    
    rouge1_score5 = score5[0]['rouge-1']['f']
    rouge2_score5 = score5[0]['rouge-2']['f']
    rougeL_score5 = score5[0]['rouge-l']['f']
    
    summarizer_names = ['T5', 'LED', 'BART','TF-IDF','PAGE-RANK']

    # Radar Chart for Performance Metrics
    plt.figure(figsize=(8, 6))
    metrics = ['Summary Length', 'ROUGE-L Score']  # Add more metrics as needed
    data = [[len(str1), rougeL_score1], [len(str2), rougeL_score2], [len(str3), rougeL_score3], [len(str4), rougeL_score4], [len(str5), rougeL_score5]]  # Example data for each summarizer
    sns.lineplot(x=metrics, y=data[0], label=summarizer_names[0])
    sns.lineplot(x=metrics, y=data[1], label=summarizer_names[1])
    sns.lineplot(x=metrics, y=data[2], label=summarizer_names[2])
    sns.lineplot(x=metrics, y=data[3], label=summarizer_names[3])
    sns.lineplot(x=metrics, y=data[4], label=summarizer_names[4])
    plt.title('Performance Metrics by Summarizer')
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.legend(title='Summarizer')
    # plt.show()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.clf()
    return send_file(buffer,mimetype='image/png')


@app.route('/rouge1',methods=['GET', 'POST'])
def rouge1():
    return render_template('plot1.html')

@app.route('/rouge2',methods=['GET', 'POST'])
def rouge2():
    return render_template('plot2.html')

@app.route('/rouge3',methods=['GET', 'POST'])
def rouge3():
    return render_template('plot3.html')

@app.route('/rouge4',methods=['GET', 'POST'])
def rouge4():
    return render_template('plot4.html')

@app.route('/rouge5',methods=['GET', 'POST'])
def rouge5():
    return render_template('plot5.html')

@app.route('/rouge6',methods=['GET', 'POST'])
def rouge6():
    return render_template('plot6.html')

app.run()