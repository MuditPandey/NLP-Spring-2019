// Wrapper for maximum-entropy tagging

// NYU - Natural Language Processing - Prof. Grishman

// invoke by:  java  MEtag dataFile  model  responseFile

import java.io.*;
import java.lang.Math.*;
import java.util.regex.Matcher;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import opennlp.maxent.*;
import opennlp.maxent.io.*;

// reads line with tab separated features
//  writes feature[0] (token) and predicted tag

public class MEtag {

    static List<String []> Viterbi(GISModel m, List<String> sentence){
		List<String[]> answers = new ArrayList<>();	
		int N = m.getNumOutcomes();
		int T = sentence.size();

		String[] tags = new String[N];
		//System.out.println("-------- OUTCOME MAP -----------");
		for(int i = 0;i<N;i++)
		{
			tags[i] = m.getOutcome(i);
			//System.out.println("Index: " + i + " Value: " + tags[i]);
		}
		double[][] vb = new double[N][T];
		for (double[] row: vb)
    		Arrays.fill(row, Math.log(0));
		
		int[][] backptr = new int[N][T];
		for (int[] row: backptr)
    		Arrays.fill(row, -1);

		String first = sentence.get(0).replaceAll("@@", Matcher.quoteReplacement("#"));
		//System.out.println(first);
		String[] first_features = first.split("\t");
		double[] first_likelihoods = m.eval(first_features);
		for(int i = 0 ;i < N; i++)
		{
			vb[i][0] = Math.log(first_likelihoods[i]);
		}
		
		// System.out.println("----------------- Starting Probabilities --------------");
		// for(int i = 0; i< N;i++)
		// {
		// 	System.out.println("Index: "+i+" Tag: "+tags[i]+ " Prob: "+vb[i][0]);
		// }




		for(int t = 1; t< T; t++)
		{
			String obs = sentence.get(t);
			//System.out.println("----- TIMESTEP: "+ t);
			//System.out.println(obs);
			for(int s = 0; s < N; s++)
			{
				//System.out.println("---------------- CURRENT STATE: "+ tags[s]+" ----------");
				for(int j = 0; j < N ;j++)
				{
					String line = obs.replaceAll("@@", Matcher.quoteReplacement(tags[j]));
					//System.out.println(line);
					String[] features = line.split("\t");
					double score = vb[j][t-1] + Math.log(m.eval(features)[s]);
					//System.out.println("----------------------- PREV STATE: "+tags[j]+" Old Value: "+ vb[j][t-1]+ " New value: "+score);
					if(vb[s][t] < score)
					{
						double temp = vb[s][t];
						vb[s][t] = score;
						backptr[s][t] = j;
						//System.out.println("------------------------------------ Updating Observation("+t+") Current State: "+tags[s]
						//+" Previous State: "+backptr[s][t] + " {"+temp+"} -> {"+vb[s][t]+"}");
					}
					
				}
			}
		}
		
		double max_value = vb[0][T-1];
		int max_index = 0;
		for(int i = 1 ; i < N; i++)
		{
			if(max_value < vb[i][T-1])
			{
				max_value = vb[i][T-1];
				max_index = i;
			}
		}

		int state = max_index;
		int t = T-1;
		String[] predicted_tags = new String[T];
		while(state!= -1)
		{
			predicted_tags[t] = tags[state];
			state = backptr[state][t];
			t--;
		}

		for(int i = 0; i < T;i++)
		{
			String[] add_out = new String[2];
			add_out[0] = sentence.get(i).split("\t")[0];
			add_out[1] = predicted_tags[i];
			answers.add(add_out);
		}
		return answers;
	}
	
	public static void main (String[] args) {
	if (args.length != 4) {
		System.err.println ("MEtag requires 4 arguments:  dataFile model responseFile tagger");
		System.err.println ("Tagger has two options: 'greedy' or 'viterbi'");
	    System.exit(1);
	}
	String dataFileName = args[0];
	String modelFileName = args[1];
	String responseFileName = args[2];
	String option = args[3];
	try {
	    GISModel m = (GISModel) new SuffixSensitiveGISModelReader(new File(modelFileName)).getModel();
	    BufferedReader dataReader = new BufferedReader (new FileReader (dataFileName));
	    PrintWriter responseWriter = new PrintWriter (new FileWriter (responseFileName));
	    String priorTag = "#";
		String prepriorTag = "#";
		String line;
		List<String> sentence = new ArrayList<>();	
	    while ((line = dataReader.readLine()) != null) {

		if (line.equals("")) {
			if(option.equals("viterbi"))
			{
				List<String []> val_pred = Viterbi(m,sentence);
				for(String [] str: val_pred)
				{
					responseWriter.println(str[0] + "\t" + str[1]);
				}
				sentence.clear();
			}
			responseWriter.println();
		    priorTag = "#";
			prepriorTag = "#";
			
		} else {
			if(option.equals("viterbi"))
			{
				sentence.add(line);
			}
			else
			{
				line = line.replaceAll("@@", Matcher.quoteReplacement(priorTag));
				line = line.replaceAll("$$", Matcher.quoteReplacement(prepriorTag));
				String[] features = line.split("\t");
				String tag = m.getBestOutcome(m.eval(features));
				responseWriter.println(features[0] + "\t" + tag);
				prepriorTag = priorTag;
				priorTag = tag;
			}
		}
	    }
	    responseWriter.close();
	} catch (Exception e) {
	    System.out.print("Error in data tagging: ");
	    e.printStackTrace();
	}
    }

}
