import sys
import argparse
import os
import json
import re
import html
import spacy
import string

# indir = '/u/cs401/A1/data/';
indir = '/Users/oluwaseuncardoso/CSC401/A1/data/'
cardosoo_nlp = spacy.load('en', disable=['parser', 'ner']) # put the model on top
# TODO: read abbrev from here

cardossoo_idir = '/Users/oluwaseuncardoso/CSC401/A1/Wordlists/'
def cardosoo_words():
    files = {"abbrev.english": []}
    for category in files:
        path = cardossoo_idir + "/" + category
        file = open(path)
        for i in file:
            word = i.lower().strip("\n")
            if word != " " and word != "":
                files[category].append(word)

    return files

cardosoo_files = cardosoo_words()

def preproc1( comment , steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step

    Returns:
        modComm : string, the modified comment
    '''

    if 1 in steps:
        comment = re.sub("\n",' ',comment)
        print("1", comment)
    if 2 in steps:
        htmlCodes =  re.findall(r"&#*\w*;\s*",comment)
        for code in htmlCodes:
           asci = html.unescape(code)
           comment = re.sub(code,asci, comment)
        print("2", comment)

    if 3 in steps:
        comment = re.sub(r"www.[^\s]+\b|https?:[^\s]+\b", " ", comment)
        print("3", comment)
    if 4 in steps:
        # add spaces to the left side of punctuations while ignoring ' and .
        step1 = re.sub(r'(?<=[!"#$%&()*+,\-/:;<=>?@[\]^_`{|}~])(?=[^\s!"#$%&.()*+,\-/:;<=>?@[\]^_`{|}~])', r' ', comment)
        # add spaces to the right side of punctuations while ignoring ' and .
        step2 = re.sub(r'(?<=[^\s!"#$%&.()*:+,\-/;<=>?@[\]^_`{|}~])(?=[!"#$%&()*+,\-/:;<=>?@[\]^_`{|}~])', r' ', step1)
        # add spaces to the left of periods
        comment = re.sub(r'(?<=[.])(?=[\s])', r' ', step2)
        # add spaces to the right of periods
        #Heauristic: if there is a 3 letter word or more after ""
        re.findall(r'\S\.*\w+\.*\w*\S', step2)
        comment = re.sub(r'(\w{3,})(?=[.])', r'\1 ', comment) #add wordlist here
        print("4", comment)
    if 5 in steps:
        comment = re.sub(r'(?<=[\w][^n])(?=[\'])', r' ', comment)
        comment = re.sub(r'(?<=[\w]d)(?=n\'t)', r' ', comment)
        comment = re.sub(r'(?<=[\w][\w])(?=[\'])', r' ', comment)
        print("5", comment)
    if 10 in steps:
        # https://stackoverflow.com/questions/4145451/using-a-regular-expression-to-replace-upper-case-repeated-letters-in-python-with
        comment = re.sub(r'(.*)', lambda pat: pat.group(1).lower(), comment) #should we make the tags lower: she said yes
        print("10", comment)

    cardoso_utt = cardosoo_nlp(u"" + comment + "")

    if 8 in steps:
        step = ""
        for token in cardoso_utt:
            if token.text == " ":  # skip spaces
                continue
            if token.lemma_[0] != "-":  # if lemma doesn't start with -
                step = step + token.lemma_ + " "
            else:
                step = step + token.text + " "
        comment = step
        print("8", comment)

    if 6 in steps: #Spacy
        step = ""
        for token in cardoso_utt:
            if token.text == " ": # skip spaces
                continue
            else:
                step = step + token.text + "/" + token.tag_ + " "
        comment = step
        print("6", comment)


    if 9 in steps: #6,8,9 will be together
        comment = re.sub(r'(?<=\W/\.)(?=\s)', r' \n', comment)
        print("9", comment)
    if 7 in steps:
        file = open('/Users/oluwaseuncardoso/CSC401/A1/Wordlists/StopWords','r') #change this
        for line in file:
            regex = "\\b" + line.strip("\n") + "/[a-zA-Z]+\\b"
            comment = re.sub(r""+regex, "", comment)
        print("7", comment)




    return comment

def main( args ):

    allOutput = []
    i = 0
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            #print( "Processing " + fullFile)
            cat = re.search(r"\w+$",fullFile).group()
            data = json.load(open(fullFile))

            # DONE: select appropriate args.max lines
            start = args.ID[0] % len(data)
            data = data[ start : start + args.max]
            comment ={}
            for line in data:
                j = json.loads(line)
                comment['id'] = j['id']
                comment['cat'] = cat
                comment['body'] = preproc1(j['body'])
                allOutput.append(dict(comment))



            # TODO: choose to retain fields from those lines that are relevant to you
            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...)
            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            # TODO: replace the 'body' field with the processed text
            # TODO: append the result to 'allOutput'

            i += 1

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description ='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()
    args.max = int(args.max)
    if (args.max > 200272):
        #print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)

    main(args)
