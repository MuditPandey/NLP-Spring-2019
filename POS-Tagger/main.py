import numpy as np 
import argparse
from multiprocessing import Pool
from tqdm import tqdm
import time

from viterbi import Viterbi
from data_loader import DataLoader


def load_file_with_keys(files):
    data = []
    labels = []
    sentences = 0
    for file in files:
        print('Load test file: {}'.format(file))
        with open(file,'r') as fil:
            i = 1
            X = []
            Y = []
            for line in fil:
                try:
                    word = line[:len(line)-1].split('\t')[0]
                    tag = line[:len(line)-1].split('\t')[1]
                    X.append(word)
                    Y.append(tag)
                except:
                    data.append(X)
                    labels.append(Y)
                    sentences+=1
                    X = [] 
                    Y = []                       
                i+=1
    return data,labels

def load_file_without_keys(files):
    data = []
    labels = []
    sentences = 0
    for file in files:
        print('Load test file: {}'.format(file))
        with open(file,'r') as fil:
            i = 1
            X = []
            for line in fil:
                if line =='\n':
                    data.append(X)
                    sentences+=1
                    X = []  
                else:
                    word = line[:len(line)-1].split('\n')[0]
                    X.append(word)
                                        
                i+=1
    return data



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='POS tagging')
    parser.add_argument('--train',nargs = '+', default =['./WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_02-21.pos'], help = 'Training files for viterbi')
    parser.add_argument('--num_threads',type = int, default = 4, help = 'Set number of worker threads')
    parser.add_argument('--test', default = './WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_24.words',help = 'Test files')
    parser.add_argument('--output', default = 'output.txt', help = 'Name of the output file')
    args = parser.parse_args()
    
    training =  DataLoader(files = args.train, unknown_thresh = 1)
    #validation = load_file_without_keys(files =[args.test])
 
    if args.test.endswith(".pos"):
        validation, _ = load_file_with_keys(files =[args.test])
    else:
        validation= load_file_without_keys(files =[args.test])
    
    print('Sentences in test file : {}'.format(len(validation)))
    
    s = time.monotonic()
    v = Viterbi(training)

    result = []
    if args.num_threads:
        with Pool(6) as p:
            result = p.map(v.predict,validation)

        print('Writing to output file: {}'.format(args.output))
        with open(args.output,'w') as fp:
            for i,sentence in enumerate(validation):
                for word,tag in zip(sentence,result[i][1]):
                    fp.write(word+'\t'+tag+'\n')
                fp.write('\n')
    else:
        print('Writing to output file: {}'.format(args.output))
        with open(args.output,'w') as fp:
            i = 0
            for sentence in tqdm(validation):
                #tags = labels[i]
                i+=1 
                probs, tags = v.predict(sentence)
                if len(tags) != len(sentence):
                    print('Sentence No: {} length: {} Tag length:{}'.format(i,len(sentence),len(tags)) )
                #print('Sentence length: {} Tag Length: {}'.format(len(sentence),len(tags))) 
                for word,tag in zip(sentence,tags):
                    fp.write(word+'\t'+tag+'\n')
                fp.write('\n')  
                
    
    print('Total time taken: {} mins'.format((time.monotonic()-s)/60))

