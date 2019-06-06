package org.apache.main;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.eval.KFoldRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.model.DataModel;

public class MainBruteForce {

	public static void main(String[] args) {
		try {
			System.setOut(new PrintStream(new FileOutputStream("C:\\Users\\Moham\\git\\mahout\\mr\\logs\\resultLogBrute.log")));
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		}
		float threshold=3 ;
		String rootPath= System.getProperty("user.dir") ;
		String fileName= "TestSet0" ;
		int nri=200 ;
		int k = 175 ;

		DataModel model = null ;
		IRStatistics irstats = null ;

		try {
			model = new FileDataModel(new File(rootPath+"\\Datasets\\"+fileName+".csv"));
			RecommenderBuilder builder = new MyRecommenderBuilder1(model,threshold, k);
			KFoldRecommenderIRStatsEvaluator evaluatorIRStats = new KFoldRecommenderIRStatsEvaluator(model, 5); /* 5-fold */
			irstats = evaluatorIRStats.evaluate(builder, nri, threshold);

		} catch (IOException e) {
			e.printStackTrace();
		}catch (TasteException e) {
			e.printStackTrace();
			
		}
		

		System.out.println("****************** Test quality of brute force recommenders *******************");
		System.out.println("number of recommanded items: "+nri);
		System.out.println("Threshold: "+threshold);
		System.out.println("k : "+k);
		System.out.println();
		System.out.println("Precision: "+irstats.getPrecision());
		System.out.println("Recall: "+irstats.getRecall());


	}

}
