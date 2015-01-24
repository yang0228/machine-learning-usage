
import java.io.*;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class DoRandomForestCrossValidation
{

	/**
	 * @param args
	 *
	 * Random forest with weka API
	 * Compile: javac -cp weka.jar: DoRandomForestCrossValidation.java
	 * Usage: java -cp weka.jar: DoRandomForestCrossValidation 0 500
	 */

	
	
	public static void main(String[] args) throws Exception
	{
		// TODO Auto-generated method stub
		System.out.println("Doing 5-fold cross-validation with Random Forest...");
		
		File file = new File("pt_cdht_N25_100slt_infog.arff");
		ArffLoader loader = new ArffLoader();
		loader.setFile(file);
		Instances samples = loader.getDataSet();
		//samples.setClassIndex(samples.numAttributes()-1);
		samples.setClassIndex(0);
		samples.randomize(new Random());
		RandomForest rf = new RandomForest();
		rf.setNumFeatures(Integer.parseInt(args[0]));
		rf.setNumTrees(Integer.parseInt(args[1]));
		rf.setSeed(new Random().nextInt());
		System.out.println("atr: " + rf.getNumFeatures() + "   trees: " + rf.getNumTrees());
		
		double AUC = 0.0;
		double accuracy = 0.0;
		double TP = 0.0;
		double TN = 0.0;
		double FP = 0.0;
		double FN = 0.0;
		double FMeasure = 0.0;
		double mcc = 0.0;
		double SP = 0.0;
		double SN = 0.0;
		
		
		
		for(int i = 0 ; i <5 ;i ++)
		{
			Instances train = samples.trainCV(5, i);
			Instances test = samples.testCV(5, i);

			rf.buildClassifier(train);
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(rf, test);
			
			AUC = AUC + eval.areaUnderROC(0);
			accuracy =accuracy + (1-eval.errorRate());
			TP = TP + eval.numTruePositives(0);
			TN = TN + eval.numTrueNegatives(0);
			FP = FP + eval.numFalsePositives(0);
			FN =  FN + eval.numFalseNegatives(0);
			FMeasure = FMeasure + eval.fMeasure(0);
			mcc = mcc + (eval.numTruePositives(0)*eval.numTrueNegatives(0)-eval.numFalsePositives(0)*eval.numFalseNegatives(0))/Math.sqrt((eval.numTruePositives(0)+eval.numFalsePositives(0))*(eval.numTruePositives(0)+eval.numFalseNegatives(0))*(eval.numTrueNegatives(0)+eval.numFalsePositives(0))*(eval.numTrueNegatives(0)+eval.numFalseNegatives(0)));
			SP = SP + eval.numTrueNegatives(0)/(eval.numFalsePositives(0)+eval.numTrueNegatives(0));
			SN = SN + eval.numTruePositives(0)/(eval.numTruePositives(0)+eval.numFalseNegatives(0));
		}
		
		File out = new File("test" + Integer.parseInt(args[0]) + "_" +  Integer.parseInt(args[1]) +  ".csv");
		FileWriter fw = new FileWriter(out, true);
		BufferedWriter bw = new BufferedWriter(fw);
		
		System.out.println("AUC:" + AUC/5);
		System.out.println("Accuracy: " + accuracy/5);
		System.out.println("TP: " + TP/5);
		System.out.println("TN: " + TN/5);
		System.out.println("FP: " + FP/5);
		System.out.println("FN: " + FN/5);
		System.out.println("FMeasure: " + FMeasure/5);
		System.out.println("MCC: " + mcc/5);
		System.out.println("SP: " + SP/5);
		System.out.println("SN: " + SN/5);
		
		bw.write(AUC/5+","+accuracy/5 + "," + TP/5 + "," + TN/5 + "," + FP/5 + "," + FN/5 + "," + FMeasure/5 + "," + mcc/5 +"," + SP/5 + "," + SN/5 + "\n");
		bw.flush();
		
		bw.close();
		fw.close();
	}

}
