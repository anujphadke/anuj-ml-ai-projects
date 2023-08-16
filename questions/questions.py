import math
import os
import string

import nltk
import sys

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    corpus = {}
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and filename.endswith(".txt"):
            with open(file_path, "r", encoding='utf8') as file:
                corpus[filename] = file.read()

    return corpus


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    punctuation = string.punctuation
    stop_words = nltk.corpus.stopwords.words("english")

    words = nltk.word_tokenize(document.lower())
    ret = [word for word in words if word not in punctuation and word not in stop_words]

    return ret


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    counts = dict()

    for filename in documents:
        seen_words = set()

        for word in documents[filename]:
            if word not in seen_words:
                seen_words.add(word)
                try:
                    counts[word] += 1
                except KeyError:
                    counts[word] = 1

    ret = {}
    for word in counts:
        ret[word] = math.log(len(documents) / counts[word])
    return ret


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    ret = dict()
    for file, words in files.items():
        total_tf_idf = 0
        for word in query:
            total_tf_idf += words.count(word) * idfs[word]
        ret[file] = total_tf_idf

    sorted_files = sorted(ret.items(), key=lambda x: x[1], reverse=True)
    sorted_files = [x[0] for x in sorted_files]

    return sorted_files


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentences_score = dict()
    for sentence in sentences:
        score = {'matching word measure': 0, 'query term density': 0}

        matched_words = 0
        for word in query:
            if word in sentences[sentence]:
                score['matching word measure'] += idfs[word]
                matched_words += 1

        score['query term density'] = matched_words / len(sentences[sentence])
        sentences_score[sentence] = score

    ret = sorted(sentences_score, key=lambda k: (
        sentences_score[k]['matching word measure'], sentences_score[k]['query term density']), reverse=True)

    top_n = ret[:n]
    return top_n


if __name__ == "__main__":
    main()
