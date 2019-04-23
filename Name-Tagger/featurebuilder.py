import argparse
import numpy as np
import time
import sys
import pickle 


verbose = False
embedding_key = {}
embeddings = np.zeros((4000000,50))

mean_embeddings = {}

count_label = dict()
count_token = dict()
count_label_token= dict()
npmi = dict()
prototypes = {}

def cosine_distance(v1,v2):
    if np.dot(v1,v2) == 0:
        return 0
    return np.dot(v1,v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))

def print_summary(pro):
    
    with open('summary.log', 'w') as f:
        f.write('---------- Labels ------------\n')
        for k,v in count_label.items():
            f.write('(l)Key: {}\t\t\t\tCount:{}\n'.format(k,v))
        
        f.write('---------- Words ------------\n')
        for k,v in count_token.items():
            f.write('(w)Key: {}\t\t\t\tCount:{}\n'.format(k,v))
        
        f.write('---------- Words and label ------------\n')
        for k,v in count_label_token.items():
            f.write('Key: {}\n'.format(k))
            for w,c in v.items():
                f.write('      (wl-{})Word: {}\t\t\t\tCount:{}\n'.format(k,w,c))
        
        f.write('---------- NPMI ------------\n')
        for k,v in npmi.items():
            f.write('Key: {}\n'.format(k))
            for w,val in v.items():
                f.write('      (npmi-{})Word: {}\t\t\tValue:{}\n'.format(k,w,val))
        f.write('---------- Prototypes ------------\n')
        for k,v in pro.items():
            f.write(k + '\n')
            f.write('-----------\n')
            for words in v:
                f.write(words + '\n')
            f.write('----------------------\n')

def initialize_probabilities(data, num_proto = 5):

    for sentence in data:
        for word_line in sentence:
            count_label[word_line[3]] = count_label.setdefault(word_line[3],0) + 1
            count_token[word_line[0]] = count_token.setdefault(word_line[0],0) + 1
            count_label_token[word_line[3]][word_line[0]] = count_label_token.setdefault(word_line[3],dict()).setdefault(word_line[0],0) + 1

    #print(len(count_label.values()))
    #print(len(count_token.values()))
    total_labels = sum(val for val in count_label.values())
    total_words = sum(val for val in count_token.values())
    #print('Total labels: {} Total words: {}'.format(total_labels,total_words))
    assert(total_labels == total_words)
    for k in count_label.keys():
        npmi[k] = dict()
        for w in count_token.keys():
            npmi[k][w] = -1 + np.log((count_label[k]/total_labels) * (count_token[w]/total_words))/np.log(count_label_token[k].get(w,0)/total_labels)

    num_proto = min(num_proto, len(count_token))
    for k in count_label.keys():
        word_index = []
        for words in npmi[k].keys():
            word_index.append(words)
        
        proto_idx = np.argpartition(list(npmi[k].values()), -num_proto)[-num_proto:]
        proto_list = [word_index[idx] for idx in proto_idx]
        prototypes[k] = proto_list

    if verbose:
        print('Printing summary')
        print_summary(prototypes)
    pickle_out = open("prototypes.pickle","wb")
    pickle.dump(prototypes, pickle_out,protocol = 0)
    pickle_out.close()



def load_mean_embeddings():
    for i,row in enumerate(embeddings.T):
        print('Loading mean embeddings [%d/50]\r'%i,end="")
        pos =[]
        neg =[]
        for val in row:
            if val >= 0:
                pos.append(val)
            else:
                neg.append(val)
        pos = np.mean(pos)
        neg = np.mean(neg)
        mean_embeddings[i] = (pos,neg)

def load_embeddings(filename):
    global embeddings
    with open(filename,'r') as f:
        i = 0
        for lines in f:
            print('Loading embeddings [%d/4000000]\r'%i,end="")
            lines = lines.split(' ')
            embeddings[i] = np.array([float(i) for i in lines[1:]])
            embedding_key[lines[0]] = i
            # if embeddings is None:
            #     embeddings = emb_vector
            # else:
            #     embeddings = np.concatenate((embeddings,emb_vector),axis = 1)
            i+=1
    print('Loaded words: {} embeddings: {} embedding length: {}'.format(len(embedding_key),len(embeddings),len(embeddings[0])))

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


def get_ngrams(string, n):
    n_grams = []
    for i in range(0, len(string) - n + 1):
        n_grams.append(string[i:i+n])

    return n_grams

def get_data(filename):
    
    with open(filename,'r') as f:

        numSentences = 0
        data = []
        sentence = []
        for line in f:
            
            if line == '\n':
                numSentences +=1
                data.append(sentence)
                sentence = []
                continue
            
            line = line.rstrip('\n').split('\t')
            if filename.endswith('.pos-chunk-name'):
                assert(len(line) == 4)
            else:
                assert(len(line) == 3)
            
            sentence.append(line)           
    
        print('Number of sentences: {}'.format(numSentences))           
        print('Data -> Number of sentences: {}'.format(len(data)))

        #print(data[1])   

    return data

def _gen_features_naive(token, POS, chunk, tag):
    result = {}
    result['token'] = token
    if POS != '':
        result['POS'] = POS
        #print('POS: {}'.format(POS))
    if chunk != '':
        result['chunk'] = chunk
        #print('Chunk: {}'.format(chunk))
    
    result['tag'] = tag
    return result

def _gen_features_better(index,
                      token, 
                      POS, 
                      chunk, 
                      tag, 
                      prev_tag,
                      prev_POS_tag,
                      forward_POS_tag,
                      prev_token,
                      forward_token,
                      prefix_length = 4,
                      suffix_length = 4,
                      embed_bin_thresh = 0.5,
                      prototype_thresh = 0.5):
    '''
    List of features:
    token
    POS
    chunk
    --------
    prev tag
    pre-prev tag
    prev tag, pre-prev tag

    prev_pos tag
    pre-prev_pos tag
    curr_pos, pre-prev_pos tag, prev_pos tag
    fw_pos tag
    fw_fw_pos tag
    curr_pos, fw_pos tag, fw_fw_pos tag
    --------
    Start_caps_init
    Start_caps_not_init
    --------
    tag


    '''
    
    result = {}
    result['token'] = token
    if POS != '':
        result['POS'] = POS
        #print('POS: {}'.format(POS))
    if chunk != '':
        result['chunk'] = chunk
        #print('Chunk: {}'.format(chunk))
    result['tag'] = tag

    # Local features
    #Name tags
    if len(prev_tag) == 2:
        result['prev_tag_1'] = prev_tag[0]
        result['prev_tag_2'] = prev_tag[1]
        result['prev_seq_1'] = prev_tag[1] + ',' + prev_tag[0]
        result['prev_seq_2'] = prev_tag[0] + ',' + tag
    elif len(prev_tag) == 1:
        result['prev_tag_1'] = prev_tag[0]
        #result['prev_seq_1'] = prev_tag[0] + ',' + tag
    # POS tags
    # Prev
    if len(prev_POS_tag) == 2:
        result['prev_pos_tag_1'] = prev_POS_tag[0]
        result['prev_pos_tag_2'] = prev_POS_tag[1]
        result['prev_pos_seq_1'] = prev_POS_tag[1] + ',' + prev_POS_tag[0]
        result['prev_pos_seq_2'] = prev_POS_tag[0] + ',' + POS
    elif len(prev_POS_tag) == 1:
        result['prev_pos_tag_1'] = prev_POS_tag[0]
        result['prev_pos_seq_all'] = prev_POS_tag[0] + ',' + POS
    
    # Forward
    if len(forward_POS_tag) == 2:
        result['fw_pos_tag_1'] = forward_POS_tag[0]
        result['fw_pos_tag_2'] = forward_POS_tag[1]
        result['fw_pos_seq_1'] = POS + ',' + forward_POS_tag[0]
        result['fw_pos_seq_2'] = forward_POS_tag[0] + ',' + forward_POS_tag[1]
    elif len(forward_POS_tag) == 1:
        result['fw_pos_tag_1'] = forward_POS_tag[0]
        result['fw_pos_seq_all'] = POS + ',' + forward_POS_tag[0]

    if len(forward_POS_tag) >= 1 and len(prev_POS_tag) >=1:
        #result['pos_seq_half'] = prev_POS_tag[0] + ',' + POS + ',' + forward_POS_tag[0]
        pass
    
    # Words
    if len(prev_token) == 2:
        result['prev_token_1'] = prev_token[0]
        result['prev_token_2'] = prev_token[1]
        result['prev_token_seq_1'] = prev_token[1] + ',' + prev_token[0]
        result['prev_token_seq_2'] = prev_token[0] + ',' + token
        #result['prev_token_all'] = prev_token[1] + ',' + prev_token[0] + ',' + token
    elif len(prev_token) == 1:
        result['prev_token_1'] = prev_token[0]
        result['prev_token_all'] = prev_token[0] + ',' + token
    
    if len(forward_token) == 2:
        result['fw_token_1'] = forward_token[0]
        result['fw_token_2'] = forward_token[1]
        result['fw_token_seq_1'] = token + ',' + forward_token[0]
        result['fw_token_seq_2'] = forward_token[0] + ',' + forward_token[1]
    elif len(forward_token) == 1:
        result['fw_token_1'] = forward_token[0]
        result['fw_token_all'] = token + ',' + forward_token[0]
    
    # Lexical features
    # Starting letter capital
    if token[0].isalpha() and token[0].isupper():
        result['initcaps'] = 'True'
        if index == 0:
            result['firstword_initcaps'] = 'True'
        else:
            result['firstword_initcaps'] = 'False'
    else:
        result['initcaps'] = 'False'
        result['firstword_initcaps'] = 'False'

    # All capital
    if token.isalpha() and token.isupper():
        result['allcaps'] = 'True'
    else:
        result['allcaps'] = 'False'

    if hasNumbers(token):
        result['hasnum'] = 'True'
    else:
        result['hasnum'] = 'False'

    # Prefixes

    for i in range(1, min(prefix_length,len(token)) + 1):
        result['prefix_' + str(i)] = token[:i]

    # Suffixes
    
    for i in range(1, min(suffix_length,len(token)) + 1):
        result['suffix_' + str(i)] = token[len(token)-i:]
    
    # n-grams
    # No gain observed
    # unigrams = get_ngrams(token, 1)
    # for gram in unigrams:
    #     result[gram] = gram
    
    # bigrams = get_ngrams(token, 2)
    # for gram in bigrams:
    #     result[gram] = gram
    
    # trigrams = get_ngrams(token, 3)
    # for gram in trigrams:
    #     result[gram] = gram
    
    # four_grams = get_ngrams(token, 4)
    # for gram in four_grams:
    #     result[gram] = gram
    
    # five_grams = get_ngrams(token, 5)
    # for gram in five_grams:
    #     result[gram] = gram
    
    # Word embeddings
    idx = embedding_key.get(token.lower(),-1)
    # Plain binarization
    if idx != -1:
        word_embed = embeddings[idx]
        for i,val in enumerate(word_embed):
            # if val > mean_embeddings[i][0]:
            #     result['embed_'+str(i)] = 'P'
            # if val < mean_embeddings[i][1]:
            #     result['embed_'+str(i)] = 'N'
            if val > embed_bin_thresh:
                result['embed_'+str(i)] = 'P'
            if val < -embed_bin_thresh:
                result['embed_'+str(i)] = 'N'

    #Previous and forward embeddings
    # if len(prev_token) == 2:
    #     #previous word
    #     idx = embedding_key.get(prev_token[0].lower(),-1)
    #     if idx != -1:
    #         word_embed = embeddings[idx]
    #         for i,val in enumerate(word_embed):
    #             # if val > mean_embeddings[i][0]:
    #             #     result['embed_'+str(i)] = 'P'
    #             # if val < mean_embeddings[i][1]:
    #             #     result['embed_'+str(i)] = 'N'
    #             if val > embed_bin_thresh:
    #                 result['prev_embed_1_'+str(i)] = 'P'
    #             if val < -embed_bin_thresh:
    #                 result['prev_embed_1_'+str(i)] = 'N'
    #     #pre-previous word
    #     idx = embedding_key.get(prev_token[1].lower(),-1)
    #     if idx != -1:
    #         word_embed = embeddings[idx]
    #         for i,val in enumerate(word_embed):
    #             # if val > mean_embeddings[i][0]:
    #             #     result['embed_'+str(i)] = 'P'
    #             # if val < mean_embeddings[i][1]:
    #             #     result['embed_'+str(i)] = 'N'
    #             if val > embed_bin_thresh:
    #                 result['prev_embed_2_'+str(i)] = 'P'
    #             if val < -embed_bin_thresh:
    #                 result['prev_embed_2_'+str(i)] = 'N'
    # elif len(prev_token) == 1:
    #     idx = embedding_key.get(prev_token[0].lower(),-1)
    #     if idx != -1:
    #         word_embed = embeddings[idx]
    #         for i,val in enumerate(word_embed):
    #             # if val > mean_embeddings[i][0]:
    #             #     result['embed_'+str(i)] = 'P'
    #             # if val < mean_embeddings[i][1]:
    #             #     result['embed_'+str(i)] = 'N'
    #             if val > embed_bin_thresh:
    #                 result['prev_embed_1_'+str(i)] = 'P'
    #             if val < -embed_bin_thresh:
    #                 result['prev_embed_1_'+str(i)] = 'N'
        
    
    # if len(forward_token) == 2:
    #     #previous word
    #     idx = embedding_key.get(forward_token[0].lower(),-1)
    #     if idx != -1:
    #         word_embed = embeddings[idx]
    #         for i,val in enumerate(word_embed):
    #             # if val > mean_embeddings[i][0]:
    #             #     result['embed_'+str(i)] = 'P'
    #             # if val < mean_embeddings[i][1]:
    #             #     result['embed_'+str(i)] = 'N'
    #             if val > embed_bin_thresh:
    #                 result['fw_embed_1_'+str(i)] = 'P'
    #             if val < -embed_bin_thresh:
    #                 result['fw_embed_1_'+str(i)] = 'N'
    #     #pre-previous word
    #     idx = embedding_key.get(forward_token[1].lower(),-1)
    #     if idx != -1:
    #         word_embed = embeddings[idx]
    #         for i,val in enumerate(word_embed):
    #             # if val > mean_embeddings[i][0]:
    #             #     result['embed_'+str(i)] = 'P'
    #             # if val < mean_embeddings[i][1]:
    #             #     result['embed_'+str(i)] = 'N'
    #             if val > embed_bin_thresh:
    #                 result['fw_embed_2_'+str(i)] = 'P'
    #             if val < -embed_bin_thresh:
    #                 result['fw_embed_2_'+str(i)] = 'N'
        
    # elif len(forward_token) == 1:
    #     idx = embedding_key.get(forward_token[0].lower(),-1)
    #     if idx != -1:
    #         word_embed = embeddings[idx]
    #         for i,val in enumerate(word_embed):
    #             # if val > mean_embeddings[i][0]:
    #             #     result['embed_'+str(i)] = 'P'
    #             # if val < mean_embeddings[i][1]:
    #             #     result['embed_'+str(i)] = 'N'
    #             if val > embed_thresh:
    #                 result['fw_embed_1_'+str(i)] = 'P'
    #             if val < -embed_thresh:
    #                 result['fw_embed_1_'+str(i)] = 'N'
    
    # Prototype features:
    idx = embedding_key.get(token.lower(),-1)
    # Plain binarization
    word_embed = [0] * 50
    if idx != -1:
        word_embed = embeddings[idx]
   
    #print('Word: {} \t Word embedding: {}'.format(token,word_embed))
    for _,v in prototypes.items():
        for proto in v:
            id_proto = embedding_key.get(proto.lower(),-1)
            proto_embed = [0] * 50
            if id_proto != -1:
                proto_embed = embeddings[id_proto]
            
            #print('Prototype: {} \t Prototype embedding: {}'.format(proto,proto_embed))
            if cosine_distance(proto_embed,word_embed) >= prototype_thresh:
                #print('Cosine similiarity: {}'.format(cosine_distance(proto_embed,word_embed)))
                result['proto_' + proto] = 'proto_' + proto
    
        
    
    return result
    


def generate_features(data,
                      train = True, use_POS = True,
                      use_chunk = True, 
                      method = 'naive', 
                      prev_tag_window = 0,
                      prev_token_window = 0,
                      prev_POS_window = 0,
                      forward_POS_window = 0, 
                      forward_token_window = 0):
    
    assert(prev_tag_window <=2)
    assert(prev_POS_window <=2)
    assert(prev_token_window <=2)
    assert(forward_token_window <=2)
    assert(forward_POS_window <=2)

    features = []

    for sentence in data:

        sentence_features = []
        
        prev_token = []
        prev_tag = []
        forward_token = []
        forward_POS = []

        # Previous
        if prev_token_window > 0:
            prev_token = ['-!START!-'] * prev_token_window
        
        if prev_tag_window > 0:
            if train:
                prev_tag = ['#'] * prev_tag_window
            else:
                prev_tag.append('@@')
                if prev_tag_window == 2:
                    prev_tag.append('$$')

        if prev_POS_window > 0:
            prev_POS_tag = ['START'] * prev_POS_window
        
        # Forward

        if forward_token_window > 0:
            forward_token = ['-!END!-'] * forward_token_window
        
        if forward_POS_window > 0:
            forward_POS = ['END'] * forward_POS_window

        for i,word_line in enumerate(sentence):
            POS = ''
            chunk = ''
            token = word_line[0]
            if use_POS:
                POS = word_line[1]
            if use_chunk:
                chunk = word_line[2]
            #print('POS: {} Chunk: {}'.format(POS,chunk))
            tag = '?'
            if train:
                tag = word_line[3]
            
            # FORWARD
            # POS
            if forward_POS_window > 0:
                try: 
                    forward_POS[0] = sentence[i + 1][1]
                except:
                    forward_POS[0] = 'END'
                
                if forward_POS_window == 2:
                    try:
                        forward_POS[1] = sentence[i + 2][1]
                    except:
                        forward_POS[1] = 'END'
            # Token
            if forward_token_window > 0:
                try: 
                    forward_token[0] = sentence[i + 1][0]
                except:
                    forward_token[0] = '-!END!-'
                
                if forward_token_window == 2:
                    try:
                        forward_token[1] = sentence[i + 2][0]
                    except:
                        forward_token[1] = '-!END!-'
            
            if method == 'naive':
                result = _gen_features_naive(token, POS, chunk, tag)
            elif method == 'better':
                result = _gen_features_better(i, 
                                              token, 
                                              POS, 
                                              chunk, 
                                              tag, 
                                              prev_tag, 
                                              prev_POS_tag, 
                                              forward_POS, 
                                              prev_token, 
                                              forward_token, 
                                              prefix_length = 6,
                                              suffix_length = 6,
                                              embed_bin_thresh = 0.6)

            sentence_features.append(result)
            
            # Tags transition data 
            # PREVIOUS
            if prev_tag_window > 0:
                # Name tags
                if train:
                    if prev_tag_window == 2:
                        prev_tag[1] = prev_tag[0]

                    prev_tag[0] = tag
                else:
                    if prev_tag_window == 2:
                        prev_tag[1] = '$$'

                    prev_tag[0] = '@@'
                
            # POS tags
            if use_POS:
                if prev_POS_window > 0:
                    if prev_POS_window == 2:
                        prev_POS_tag[1] = prev_POS_tag[0]
                    prev_POS_tag[0] = POS
                
            # Words prev
            if prev_token_window > 0:
                if prev_token_window == 2:
                    prev_token[1] = prev_token[0]
                prev_token[0] = token


                

        
        features.append(sentence_features)

    #print(features[1])
    return features
            


def generate_enhanced_feature_file(features, filename = 'features.train'):

    # Features is list of sentences
    # sentences is a list of dictionaries -[token, features, tag ]

    with open(filename, 'w') as f:

        for sentence in features:

            for word_dict in sentence:  
                print_string = ''
                print_string += word_dict['token']
                
                for key, value in word_dict.items():
                    if key in ['token','tag']:
                        continue
                    if key == value:
                        feature = key
                    else:
                        feature = key + '=' + value
                    print_string = print_string + '\t' + feature 
                
                print_string = print_string + '\t' + word_dict['tag'] + '\n'
                f.write(print_string)
            f.write('\n')

                         
parser = argparse.ArgumentParser()

parser.add_argument('file', help = 'Generate features for this file')
parser.add_argument('--train', default = False, action = 'store_true', help = 'Toggle if training file')
parser.add_argument('--out', default = 'features.train', help = 'Name of the output file')
parser.add_argument('--pos', default = False, action = 'store_true')
parser.add_argument('--chunk', default = False, action = 'store_true')
parser.add_argument('--num_proto', default = 2, action = 'store_true')

args = parser.parse_args()

#train_file = './CONLL_NAME_CORPUS_FOR_STUDENTS/CONLL_train.pos-chunk-name'
print('Loading word embeddings ..')
start = time.monotonic()
load_embeddings('glove.6B.50d.txt')
print('Embeddings loaded in {}s'.format(time.monotonic()-start))
# start = time.monotonic()
# load_mean_embeddings()
# print('Embeddings loaded in {}s'.format(time.monotonic()-start))

data = get_data(args.file)
if args.train:
    print('Calculating.. prototype features')
    initialize_probabilities(data, num_proto = args.num_proto)

pickle_in = open("prototypes.pickle","rb")
prototypes = pickle.load(pickle_in)
# for k,v in prototypes.items():
#     print(k)
#     print('-----------')
#     for words in v:
#         print(words)
#     print('----------------------')

features = generate_features(data, 
                            method = 'better', 
                            prev_tag_window = 1, 
                            prev_POS_window = 2, 
                            forward_POS_window = 2, 
                            prev_token_window = 2,
                            forward_token_window = 2,
                            train = args.train, 
                            use_POS = args.pos, 
                            use_chunk = args.chunk)
generate_enhanced_feature_file(features, args.out)