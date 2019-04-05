import argparse

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
                      suffix_length = 4):
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
        result['prev_seq_all'] = prev_tag[1] + ',' + prev_tag[0]
    elif len(prev_tag) == 1:
        result['prev_tag_1'] = prev_tag[0]
    # POS tags
    # Prev
    if len(prev_POS_tag) == 2:
        result['prev_pos_tag_1'] = prev_POS_tag[0]
        result['prev_pos_tag_2'] = prev_POS_tag[1]
        result['prev_pos_seq_half'] = POS + ',' + prev_POS_tag[0]
        result['prev_pos_seq_all'] = POS + ',' + prev_POS_tag[1] + ',' + prev_POS_tag[0]
    elif len(prev_POS_tag) == 1:
        result['prev_pos_tag_1'] = prev_POS_tag[0]
        result['prev_pos_seq_all'] = POS + ',' + prev_POS_tag[0]
    
    # Forward
    if len(forward_POS_tag) == 2:
        result['fw_pos_tag_1'] = forward_POS_tag[0]
        result['fw_pos_tag_2'] = forward_POS_tag[1]
        result['fw_pos_seq_half'] = POS + ',' + forward_POS_tag[0]
        result['fw_pos_seq_all'] = POS + ',' + forward_POS_tag[0] + ',' + forward_POS_tag[1]
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
        result['prev_token_half'] = prev_token[0] + ',' + token
        result['prev_token_all'] = prev_token[1] + ',' + prev_token[0] + ',' + token
    elif len(prev_token) == 1:
        result['prev_token_1'] = prev_token[0]
        result['prev_token_all'] = prev_token[0] + ',' + token
    
    if len(forward_token) == 2:
        result['fw_token_1'] = forward_token[0]
        result['fw_token_2'] = forward_token[1]
        result['fw_token_half'] = token + ',' + forward_token[0]
        result['fw_token_all'] = token + ',' + forward_token[0] + ',' + forward_token[1]
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
                if prev_POS_window == 2:
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
                                              suffix_length = 6)

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
args = parser.parse_args()

#train_file = './CONLL_NAME_CORPUS_FOR_STUDENTS/CONLL_train.pos-chunk-name'

data = get_data(args.file)
features = generate_features(data, 
                            method = 'better', 
                            prev_tag_window = 1, 
                            prev_POS_window = 1, 
                            forward_POS_window = 1, 
                            prev_token_window = 2,
                            forward_token_window = 2,
                            train = args.train, 
                            use_POS = args.pos, 
                            use_chunk = args.chunk)
generate_enhanced_feature_file(features, args.out)