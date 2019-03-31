import numpy as np

class DataLoader:
    '''
    Functionality right now : either array or file passed at the time of creation of object
    '''
    def __init__(self,input_data = None,output_data = None,files = None, unknown_thresh = 1):
        self.files = files
        self.unknown_thresh = unknown_thresh
        self.sentences = 0
        self.tags = dict()
        self.words = dict()
        self.start = dict()
        self.transition = dict()
        self.emission = dict()

        if self.files is not None:
            data,labels = self._load_file()
            print('Loaded '+str(len(data))+' sentences and '+str(len(labels))+' output tag sequences from file!')
            self._process(data,labels)
        else:
            self._process(input_data,output_data)

    def _load_file(self):
        data = []
        labels = []
        sentences = 0
        
        for file in self.files:
            print('Load training file: {}'.format(file))
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
                        #print(str(i)+" "+word+" "+tag)    
                        prev_tag = tag
                    except:
                        data.append(X)
                        labels.append(Y)
                        sentences+=1
                        X = []
                        Y = []
                        
                    i+=1
        self.sentences = sentences
        #print('Sentences:'+str(self.sentences))
        return data,labels

    def _process_unknown(self):
        candidates = [word for word,count in self.words.items() if count <= self.unknown_thresh]
        #self.words['_UNKNOWN_'] = 0
        for unk in candidates:
            for tag in self.tags:
                if self.emission[tag].get(unk,0) !=0:
                    #self.words['_UNKNOWN_'] +=1
                    self.emission[tag]['_UNKNOWN_'] = self.emission[tag].setdefault('_UNKNOWN_',0) + 1              
                    #self.tags[tag] +=1
                    break

    def _process(self,data,labels):
        
        #Fetch each sentence and corresponding tag sequence
        b = 0
        for seq, tag_seq in zip(data,labels):
            #Set previous tag for transitions
            
            prev_tag = None
            b +=1
            #Fetch each word and tag in the sentence
            
            i = 0
            for word,tag in zip(seq,tag_seq): 
                #Add tag to tag list if not added already else increase its count
                
                self.tags[tag] = self.tags.setdefault(tag,0) + 1
                _ = self.emission.setdefault(tag,dict())
   
                #Add word to word list if not added already else increase its count
                
                self.words[word] = self.words.setdefault(word,0) + 1

                #Increment start count if previous tag is None, else increment transition count
                if prev_tag is None:
                    self.start[tag] = self.start.setdefault(tag,0) + 1
                else:
                    #Increment transition count
                    self.transition[prev_tag][tag] = self.transition.setdefault(prev_tag,dict()).setdefault(tag,0) + 1
                
                #Increment emission count of given word for the tag
                self.emission[tag][word] = self.emission[tag].setdefault(word,0) + 1
                
                prev_tag = tag
            
        self._process_unknown()
        


if __name__ == "__main__":
    d =  DataLoader(files =['./WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_02-21.pos'])
    print('Sentences: {} Unique Tags: {} Unique Words: {}'.format(d.sentences,len(d.tags),len(d.words)))
    print('Total words: {} Total tags: {}'.format(np.sum(list(d.words.values())),np.sum(list(d.tags.values()))))
    print('=======Sanity Checks==========')
    print('---- No. of tags t == No. of emissions from tag t----')
    for k in d.tags.keys():
        assert(d.tags[k] == np.sum(list(d.emission[k].values())))
    print('Passed!')
    print('----Tags:----')
    for k in d.tags.keys():
        print('Key:{}  \t\t Value:{}'.format(k,d.tags[k]))

    print('----Start Tags:----')
    print('Total Start tags: {}'.format(np.sum(list(d.start.values()))))
    for k in d.tags.keys():
        print('Key:{} \t\t Value:{} \t\t Start Prob: {}'.format(k,d.start.get(k,0),np.log(d.start.get(k,0)/d.sentences)))
    
    # d._process_unknown()
    # print('-----After processing unknowns----')  
    # print('No of unknown words: {}'.format(d.words['_UNKNOWN_']))        
    # print('Total words: {} Total tags: {}'.format(np.sum(list(d.words.values())),np.sum(list(d.tags.values()))))
    # print('=======Sanity Checks==========')
    # print('---- No. of tags t == No. of emissions from tag t----')
    # for k in d.tags.keys():
    #     assert(d.tags[k] == np.sum(list(d.emission[k].values())))
    # print('Passed!')
    # print('-----Unknown word counts-----')
    # for k in d.tags.keys():
    #     print('Tag: {} Unknown word count: {} Emission prob: {}'.format(k,d.emission[k].get('_UNKNOWN_',0),np.log(d.emission[k].get('_UNKNOWN_',0)/d.tags[k])))
    
