## Part of Speech (POS) Tagger

This HMM based POS tagger uses Viterbi decoder to generate POS tags for the data. 





The `DataLoader` assumes training data in the format of the Wall Street Journal (WSJ) Corpus. However it can be easily used with other formats. 
The `DataLoader` accepts the training file in the WSJ format OR two lists :
<br>
-A list of sentences. Each sentence is a list of words.
<br>
-A list of correct POS tags. Each element in this list corresponds to the correct tags list for a sentence.
<br>
You can then essentially, write a function that reads your file (in whatever format) and return the two lists as described above.
These can then be passed to the `DataLoader`.
