# Generate features for training and test files

#Training
echo 'Generating enhanced feature file for training data'
python3 featurebuilder.py --train --pos --out  features.train $1
#Test
echo 'Generating enhanced feature file for dev/test data'
python3 featurebuilder.py --pos --out  features.test $2
#Compiling java code
javac -classpath .:maxent-3.0.0.jar:trove.jar *.java
echo 'Starting training ...'
java -cp .:maxent-3.0.0.jar:trove.jar MEtrain features.train model.train
echo 'Generating output file for greedy: result_out.greedy'
java -cp .:maxent-3.0.0.jar:trove.jar MEtag features.test model.train result_out.greedy greedy
echo 'Generating output file for viterbi: result_out.viterbi'
java -cp .:maxent-3.0.0.jar:trove.jar MEtag features.test model.train result_out.viterbi viterbi
# Generate result
echo '----------------------------------'
echo 'Scoring greedy ..'
echo '----------------------------------'
python3 score_name.py --key $3 result_out.greedy

echo '----------------------------------'
echo 'Scoring viterbi ..'
echo '----------------------------------'
python3 score_name.py --key $3 result_out.viterbi

