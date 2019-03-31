
import numpy as np
from data_loader import DataLoader

class Viterbi:
    def __init__(self,data: DataLoader):
        self.data = data
        self.states = list(self.data.tags.keys())
        self.vb = dict()
        self.backptr = dict()
        self.lambda1 = 0
        self.lambda2 = 0
        self._deleted_interpolation()
        self.N = np.sum(list(self.data.words.values()))
    def _initialize_viterbi(self,t):
        for state in self.states:
            self.vb[state] = np.full(t,-np.inf,dtype=np.double)
            self.backptr[state] = [None] * t 

    def _deleted_interpolation(self):
        lambda1 = lambda2 = 0
        total_tags = np.sum(list(self.data.tags.values()))
        for prev_state in self.states:
            
            for state in self.data.transition[prev_state].keys():
                count_bigram = self.data.transition[prev_state].get(state,0)
                count_unigram = self.data.tags[state]
                if count_bigram > 0:
                    try:
                        a = (count_bigram - 1)/(self.data.tags[prev_state] - 1)
                    except ZeroDivisionError:
                        a = 0
                    try:
                        b = (count_unigram - 1)/(total_tags - 1)
                    except ZeroDivisionError:
                        b = 0
                    
                    if a > b:
                        lambda2 += count_bigram
                    else:
                        lambda1 += count_bigram

        norm = lambda1 + lambda2
        self.lambda2 = lambda2/norm
        self.lambda1 = lambda1/norm
        print('Lambda2: {} Lambda1:{} Sum:{}'.format(self.lambda2,self.lambda1,self.lambda1 + self.lambda2))

    def _generate_sequence(self,end_state,t):

        seq = []
        start = end_state
        for i in range(t,-1,-1):
            seq.append(start)
            if self.backptr[start][i] == None:
                break
            else:
                start = self.backptr[start][i]

        _ = seq.reverse()
        return seq
    
    def predict(self,input_seq): 
        if len(input_seq) < 1:
            return []
        
        fp = open("log_viterbi.txt","w+")
        #Initialize the matrices
        self._initialize_viterbi(len(input_seq))

        
        # Set first observation
        first = input_seq[0]
        for state in self.states:
           
            numerator = self.data.start.get(state,0)
            start_prob = -np.inf
            if self.data.start.get(state,0) == 0:
                start_prob = -np.inf
            else:
                start_prob = np.log(numerator/self.data.sentences)
            fp.write('---------------------------------------\n')
            fp.write('State: {} Start prob: {}\n'.format(state,start_prob))
            # First observation emission. Handle unknown words.
            numerator_first = self.data.emission[state].get(first, 0)
            lmbda = 1 + self.data.emission[state].get('_UNKNOWN_', 0)
            p_backoff = (1 + self.data.words.get(first,0))/(len(self.data.words) + self.N)
            first_ob_emission = np.log(numerator_first + lmbda * p_backoff) - np.log(lmbda + self.data.tags[state])

            self.vb[state][0] = start_prob + first_ob_emission 
            fp.write('State: {} Start Word:{} Count: {} EmProb:{} StartProb:{}\n'.format(state,first,numerator_first,first_ob_emission,self.vb[state][0]))
            fp.write('---------------------------------------\n')
        # Starting Viterbi
        for  t in range(1,len(input_seq)):
            observation  = input_seq[t]
            fp.write('--TIME STEP: {} OBSERVATION: {} -------------------------------\n'.format(t,observation))
            #print('Current Observation : {}'.format(observation))
            for cur_state in self.states:

                # Calculate emission probability. Handle unknown words accordingly
                fp.write('-----------CUR STATE: {}-------------------------------\n'.format(cur_state))
                
                numerator_emission = self.data.emission[cur_state].get(observation, 0)
                lmbda = 1 + self.data.emission[cur_state].get('_UNKNOWN_', 0)
                p_backoff = (1 + self.data.words.get(observation,0))/(len(self.data.words) + self.N)
                emission = np.log(numerator_emission + lmbda * p_backoff) - np.log(lmbda + self.data.tags[cur_state])
                
                for prev_state in self.states:

                    # Calculate transition probability (with backoff)
                    numerator_transition_1 = self.data.transition.get(prev_state,{}).get(cur_state,0)
                    numerator_transition_2 = self.data.tags[cur_state]
                    transition = -np.inf
                    if numerator_transition_1 == 0:
                        #transition = np.log(10 ** -15)
                        transition = np.log(self.lambda1) + np.log(numerator_transition_2/self.N)
                    else:
                        transition =  np.log(self.lambda2 * numerator_transition_1/self.data.tags[prev_state] + self.lambda1 * numerator_transition_2/self.N)
                        #transition =  np.log(numerator_transition_1/self.data.tags[prev_state])
                    # Update values for current state with most likely value
                    update_val = self.vb[prev_state][t-1] + transition + emission
                    if update_val != -np.inf:
                        fp.write('-----------------------PREV STATE: {} \tTransition: {} \t\tEmission: {} \t\tTotal: {}-------------------------------\n'.format(prev_state,transition,emission,update_val))
                    #print('-----------------------PREV STATE: {} \tTransition: {} \t\tEmission: {} \t\tTotal: {}-------------------------------'.format(prev_state,transition,emission,update_val))
                    if update_val > self.vb[cur_state][t]:
                        temp = self.vb[cur_state][t]
                        self.vb[cur_state][t] = update_val
                        self.backptr[cur_state][t] = prev_state
                        fp.write('------------------------------Updating observation ({}) : {} Current State: {} Prev state: {} from {} -> {}\n'.format(t,observation,cur_state, self.backptr[cur_state][t],temp,self.vb[cur_state][t]))

        #Set termination probabilities to find sequence
        
        t = len(input_seq)-1
        
        max_prob = self.vb[self.states[0]][t]
        end_state = self.states[0]
        for state in self.states:
            if self.vb[state][t] > max_prob:
                max_prob = self.vb[state][t]
                end_state = state
        # max_prob = -np.inf
        # end_state = None
        # for state in self.states:
        #     numerator_termination = self.data.tags[state] - np.sum(list(self.data.transition[state].values()))
        #     termination_prob = -np.inf
        #     if numerator_termination == 0:
        #         termination_prob = -np.inf
        #     else:
        #         termination_prob  = np.log(numerator_termination/self.data.tags[state])
            
        #     if self.vb[state][t] + termination_prob > max_prob:
        #         max_prob = self.vb[state][t] + termination_prob
        #         end_state = state

        fp.write('End state: {} Prob: {} Backptr: {} \n'.format(end_state,max_prob,self.backptr[end_state][t]))
        # Completed Viterbi. Generate sequence for states
        fp.close()
        predict_tag_seq = self._generate_sequence(end_state,t)

        return max_prob, predict_tag_seq
