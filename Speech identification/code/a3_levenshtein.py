import os, fnmatch
import numpy as np
import re
#dataDir = '/u/cs401/A3/data/'
dataDir = "/Users/oluwaseuncardoso/CSC401/A3/data" #NOTE remove this
def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())      #ask in Pia
    Inf 0 3 0                                                                           
    """

    n = len(r) # The number of words in REF
    m = len(h) # The number of words in HYP
    R = np.zeros((n+1,m+1))
    B = np.zeros((n+1,m+1))

    #for all i,j s.t.  i = 0 or  j = 0,	set	R[i,j] ← max (i,j) end
    R[0,:] = np.arange(m+1)
    R[:,0] = np.arange(n+1)
    # i think we should do this aswell
    up = 0
    left = 1
    up_left = 2
    up_left2 = 3
    B[0,:] = left
    B[:,0] = up
    B[0,0] = up
    for i in range(1,n+1):
        for j in range(1,m+1):
            dele = R[i - 1, j] + 1 # delete
            sub = R[i - 1, j - 1] + (1,0)[r[i-1] == h[j-1]] #substitute #NOTE look at this
            ins = R[i, j-1] + 1 #insert

            R[i,j] = min(dele,sub,ins)
            if R[i,j] == dele:
                B[i , j] = up
            elif R[i , j] == ins:
                B[i,j] = left
            else:
                B[i,j] = (up_left, up_left2 )[r[i-1] == h[j-1]]

    i,j = n,m
    nSub,nDel,nIns = 0, 0, 0
    transversal = True
    while transversal == True:
        path = B[i,j]
        if i <=  0 and j <=0:
            transversal = False
            break
        if path == up_left:
            i -= 1
            j -= 1
            nSub += 1
        elif path == left:
            j -= 1
            nIns +=1
        elif path == up:
            i -= 1
            nDel +=1
        else: # correct
            i -= 1
            j -= 1

    return R[n,m]/n, nSub, nIns, nDel

def preprocess(lines:list) -> list:
    """

    :param lines: list of unprocessed lines of text
    :return: list of listof strings
    """

    new_lines = []


    for line in lines:
        colon = line.index(":") # looks for first occurance of : in line e.g 2 T/HE:SURVIVAL
        comment = re.compile(r"[!\"#$%&\'()*+,\-./:;<=>?@\\^_`{|}~]").sub(" ",line[colon:]) # don't remove [ and ]
        comment = re.sub(r'(.*)', lambda pat: pat.group(1).lower(), comment)
        comment = re.findall(r"\S+",comment) # remove blank spaces
        new_lines.append(comment[1:])

    return new_lines


if __name__ == "__main__":
    googleWERs = []
    KaldiWERs = []
    for subdir,dirs, file  in  os.walk(dataDir):
        for speaker in dirs:

            gText = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*.Google.txt')[0]
            kText = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*.Kaldi.txt')[0]
            refText = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), 'transcripts.txt')[0]
            files = [gText,kText, refText]

            transcripts = {gText:[],kText:[],refText:[]}
            for file in files:
                # read lines in the file
                file_path = dataDir+"/"+speaker+"/"+file
                with open(file_path) as text:
                    transcripts[file]= text.readlines()
                # preproccess list of lines
                transcripts[file] = preprocess(transcripts[file])
            # put input in lavenshtein
            # NOTE: assume that all transcript files have the same number of lines for now
            i = 0
            for refText,HypText, HypText2 in zip(transcripts[refText], transcripts[kText], transcripts[gText]):

                WER, S, I, D = Levenshtein(refText,HypText)
                output = " ".join([speaker,"Kaldi",str(i) ,str(WER), "S:"+str(S), "I:"+str(I), "D:"+str(D)])
                googleWERs.append(WER)
                print(output)
                print()
                WER, S, I, D = Levenshtein(refText, HypText2)
                output = " ".join([speaker,"Google",str(i) ,str(WER), "S:"+str(S), "I:"+str(I), "D:"+str(D)])
                KaldiWERs.append(WER)
                print(output)

                i+=1

    googleWERs = np.asarray(googleWERs)
    KaldiWERs = np.asarray(KaldiWERs)

    print("Google")
    print("Mean:    " + str(np.mean(googleWERs)) )
    print("Standard Deviation of     " + str(np.std(googleWERs)))

    print("Kaldi")
    print("Mean      " + str(np.mean(KaldiWERs) ))
    print("Standard Deviation     " + str( np.std(KaldiWERs) ))


