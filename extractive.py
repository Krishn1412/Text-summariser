import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import PorterStemmer
from nltk.corpus import stopwords 
import math
from statistics import mean
import argparse
nltk.download('punkt')
nltk.download('stopwords')

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


# path = args.filepath
# with open(path, encoding="utf8") as f:
#     text = f.read()
text=("While a number of definitions of artificial intelligence (AI) have surfaced over the last few decades, John McCarthy offers the following definition in this 2004 paper (PDF, 106 KB) (link resides outside IBM), It is the science and engineering of making intelligent machines, especially intelligent computer programs. It is related to the similar task of using computers to understand human intelligence, but AI does not have to confine itself to methods that are biologically observable.However, decades before this definition, the birth of the artificial intelligence conversation was denoted by Alan Turing's seminal work, Computing Machinery and Intelligence (PDF, 89.8 KB) (link resides outside of IBM), which was published in 1950. In this paper, Turing, often referred to as the father of computer science, asks the following question, Can machines think?  From there, he offers a test, now famously known as the Turing Test, where a human interrogator would try to distinguish between a computer and human text response. While this test has undergone much scrutiny since its publish, it remains an important part of the history of AI as well as an ongoing concept within philosophy as it utilizes ideas around linguistics.")
sentences = sent_tokenize(text)
num_sent = len(sentences)
# print(sentences)

matrix_freq = frquency_matrix(sentences)
# for k,v in matrix_freq.items():
#     print(k)
#     print(v)
term_freq_mat = term_freq_matrix(matrix_freq)
# for k,v in term_freq_mat.items():
#     print(k)
#     print(v)
word_count = total_word_count(matrix_freq)
# for k,v in word_doc_count.items():
#     print(k)
#     print(v)
matrix_idf = idf_matrix(matrix_freq, word_count, num_sent)
# for k,v in matrix_idf.items():
#     print(k)
#     print(v)
matrix_tf_idf = tf_idf_matrix(term_freq_mat, matrix_idf)
# for k, v in matrix_tf_idf.items():
#     print(k)
#     print(v)
scores = sentence_scores(matrix_tf_idf)
# for k, v in scores.items():
#     print(k)
#     print(v)
avg_score = find_average_score(scores)

summary = generate_summary(sentences, scores, 0.9*avg_score)

print(summary)