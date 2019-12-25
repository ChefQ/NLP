import numpy as np
import sys
import argparse
import os
import json
import re
import csv
import statistics as st

indir = '/u/cs401/A1/Wordlists/'
idir = '/Users/oluwaseuncardoso/CSC401/A1/Wordlists/'  # NOTE change all this to their respective paths in the lab machines
csv_path1 = '/Users/oluwaseuncardoso/CSC401/A1/Wordlists/BristolNorms+GilhoolyLogie.csv'
csv_path2 = '/Users/oluwaseuncardoso/CSC401/A1/Wordlists/Ratings_Warriner_et_al.csv'
index_path = '/Users/oluwaseuncardoso/CSC401/A1/feats/'
npy_path = '/Users/oluwaseuncardoso/CSC401/A1/feats/'


def Words():  # check if you imported functions still allow you to use dependent functions, like this.
    files = {"First-person": [], "Second-person": [], "Third-person": [], "Slang": []}
    for category in files:
        path = idir + "/" + category
        file = open(path)
        for i in file:
            word = i.lower().strip("\n")
            if word != " " and word != "":
                files[category].append(word)

    return files


files = Words()


def normsDic(csv_path, col):
    words = dict()
    with open(csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            word = row[col[0]]
            words[word] = [row[col[1]], row[col[2]], row[col[3]]]
    return words


col1 = ["WORD", "AoA (100-700)", "IMG", "FAM"]  # columns we need in csv_path1

col2 = ["Word", "V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"]
wordScores = normsDic(csv_path1, col1)

warringeScores = normsDic(csv_path2, col2)


# counts the occurance of tags or individual words in a
def CountOf(comment, inputs, option):
    count = 0
    word = 0  # searching using words
    tag = 1  # searching using tags
    regex = 2  # searching using regex
    body = comment['body']
    # i did this because some sentences start with a pronoun, and my logic below can't count them with out it
    body = " " + body
    if option == word:
        for i in inputs:
            token = " " + i + "/"  # because every token has a '/' write next to it. so i don't count strings within strings
            count += body.count(token)
    elif option == tag:
        token = "/" + inputs  # because every token has a '/' write next to it. so i don't count strings within strings
        count += body.count(token)
    elif option == regex:
        count = len(re.findall(inputs, body))
    return count


def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    # TODO: your code here
    row = np.zeros(174)
    word = 0  # searching using words
    tag = 1  # searching using tags
    regex = 2  # searching using regex
    # Number of first - person pronouns
    row[0] = CountOf(comment, files["First-person"], word)
    # Number of second - person pronouns
    row[1] = CountOf(comment, files["Second-person"], word)
    # Number of third - person pronouns
    row[2] = CountOf(comment, files["Third-person"], word)
    # Number of coordinating conjunctions
    row[3] = CountOf(comment, "cc", tag)
    # Number of past-tense verbs
    row[4] = CountOf(comment, "vbd", tag)

    # Number of future-tense verbs
    count = CountOf(comment, "'ll", word)
    count += CountOf(comment, "will", word)
    count += CountOf(comment, "gonna", word)
    count += CountOf(comment, "ongoing", word)
    # TODO "going + to + VB"
    row[5] = count

    # Number of commas
    row[6] = CountOf(comment, ",", word)

    # Number of multi-character punctuation tokens
    exp = '[!\"#$%&()*+,\-:;<=>?@\[\]^_`{|}~]{2,}/'  # regex expression
    row[7] = CountOf(comment, exp, regex)

    # Number of common nouns
    exp = '/nn\s'
    count = 0
    count += CountOf(comment, exp, regex)
    exp = '/nns\s'
    count += CountOf(comment, exp, regex)
    row[8] = count

    # Number of Proper nouns
    exp = '/nnp\s'
    count = 0
    count += CountOf(comment, exp, regex)
    exp = '/nnps\s'
    count += CountOf(comment, exp, regex)
    row[9] = count

    # Number of adverbs
    adverbs = ['rb', 'rbr', 'rbs']
    count = 0
    for i in adverbs:
        exp = "/" + i + "\s"
        count += CountOf(comment, exp, regex)
    row[10] = count

    # Number of wh-words:
    wh_words = ['wdt', 'wp', 'wrb', 'wp\$']
    count = 0
    for i in wh_words:
        exp = "/" + i + "\s"
        count += CountOf(comment, exp, regex)
    row[11] = count

    # Number of slangs acronyms:
    row[12] = CountOf(comment, files["Slang"], word)

    # Number of sentences in uppercase
    exp = "\s[A-Z][a-zA-Z]{3,}/"
    row[13] = CountOf(comment, exp, regex)

    # Average length of sentences, in tokens

    # expression for number of tokens
    exp = "\S\w+/[\w]+\S"
    num_of_tokens = CountOf(comment, exp, regex)
    exp = "\n"
    num_of_sentences = CountOf(comment, exp, regex)
    try:
        # ZeroDivisionError: float division by zero
        row[14] = float(num_of_tokens) / num_of_sentences
    except ZeroDivisionError:
        row[14] = float(num_of_tokens)
    # Average length of tokens, excluding punctuatuion-only token, in characters
    tokens = re.findall(r"\S\w+/", comment["body"])  # NOTE
    length_of_tokens = 0
    for token in tokens:
        length_of_tokens += len(token)
        length_of_tokens -= 1  # remove "/" from count
    try:

        row[15] = float(length_of_tokens) / num_of_tokens
    except ZeroDivisionError:
        row[15] = float(length_of_tokens)

    # Number of sentences.
    row[16] = num_of_sentences

    # Average of AoA (100-700) from Bristol, Gilhooly, and Logie norms
    tokens = re.findall(r"\S\w+(?=/)", comment["body"])
    AoAScores = []
    IMGScores = []
    FAMScores = []
    VMSums = []
    AMSums = []
    DMSums = []
    for token in tokens:

        if token in wordScores:
            AoA = float(wordScores[token][0])
            IMG = float(wordScores[token][1])
            FAM = float(wordScores[token][2])
            AoAScores.append(AoA)
            IMGScores.append(IMG)
            FAMScores.append(FAM)

        if token in warringeScores:
            VMS = float(warringeScores[token][0])
            AMS = float(warringeScores[token][1])
            DMS = float(warringeScores[token][2])
            VMSums.append(VMS)
            AMSums.append(AMS)
            DMSums.append(DMS)

    AoAMean, IMGMean, FAMMean, AoAStd, IMGStd, FAMStd = 0, 0, 0, 0, 0, 0
    if len(AoAScores) > 0:  # if any word has any score then they should have the rest too
        AoAMean, IMGMean, FAMMean = st.mean(AoAScores), st.mean(IMGScores), st.mean(FAMScores)
    if len(AoAScores) > 1:
        AoAStd, IMGStd, FAMStd = st.stdev(AoAScores), st.stdev(IMGScores), st.stdev(FAMScores)

    VMSMean, AMSMean, DMSMean, VMSStd, AMSStd, DMSStd = 0, 0, 0, 0, 0, 0
    if len(VMSums) > 0:  # if any word has any score then they should have the rest too
        VMSMean, AMSMean, DMSMean = st.mean(VMSums), st.mean(AMSums), st.mean(DMSums)
    if len(VMSums) > 1:
        VMSStd, AMSStd, DMSStd = st.stdev(VMSums), st.stdev(AMSums), st.stdev(DMSums)

    # Averages and standard deviations of scores from Bristol, Gilhooly, and Logie norms
    row[17] = AoAMean
    row[18] = IMGMean
    row[19] = FAMMean
    row[20] = AoAStd
    row[21] = IMGStd
    row[22] = FAMStd

    # Averages and standard deviations of scores from Warringer norms
    row[23] = VMSMean
    row[24] = AMSMean
    row[25] = DMSMean
    row[26] = VMSStd
    row[27] = AMSStd
    row[28] = DMSStd

    # LIWC/Receptiviti
    cat = comment['cat']
    path = index_path + cat + "_IDs.txt"
    path2 = npy_path + cat + "_feats.dat.npy"
    data = np.zeros(144, )
    with open(path) as id:
        index = 0
        for i in id:
            index += 1
            if (i.strip("\n") == comment['id']):
                data = np.load(path2)[index]
                break
    row[29:173] = data

    category = {
        "Left": 0,
        "Center": 1,
        "Right": 2,
        "Alt": 3
    }
    row[173] = category[cat]

    return row


def main(args):
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173 + 1))

    # TODO: your code here
    index = 0
    for comment in data:
        feats[index] = extract1(comment)
        index += 1

    np.savez_compressed(args.output, feats)  # look at this chang output to npz


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()

    main(args)

