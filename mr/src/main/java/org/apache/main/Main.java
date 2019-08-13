package org.apache.main;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.PredictionStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.eval.KFoldRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.eval.KFoldRecommenderPredictionEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.recommender.ParseFile;
import org.apache.mahout.cf.taste.model.DataModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class Main {

	public static void main(String[] args) {

		
		ParseFile p;
		int nri= 1 ;
		float threshold=3 ;
		String rootPath= System.getProperty("user.dir") ;
		String fileName= "TestSet0_" ;
		
		try {
			System.setOut(new PrintStream(new FileOutputStream(rootPath+"\\logs\\resultLog.log")));
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		}
		
		try {
			File myFile= new File(rootPath+"\\Datasets\\"+fileName+".csv");
			p = new ParseFile();
			if(!myFile.exists()) {
				p.convertToCSV(fileName);
			}
			
			DataModel model = new FileDataModel(new File(rootPath+"\\Datasets\\"+fileName+".csv"));
//			List<RecommendedItem> recommendations = recommender.recommend(2, 3);
//			for (RecommendedItem recommendation : recommendations) {
//				System.out.println(recommendation);
//			}
			RecommenderBuilder builder = new MyRecommenderBuilder(); 
			KFoldRecommenderIRStatsEvaluator evaluatorIRStats = new KFoldRecommenderIRStatsEvaluator(model, 5); /* 5-fold */
		//	KFoldRecommenderPredictionEvaluator evaluatorPred = new KFoldRecommenderPredictionEvaluator(model, 5);
			int[] nra= new int[] {20,30,40,60,70,80,90} ;
			for(int i=0; i<nra.length; i++) {
			IRStatistics irstats = evaluatorIRStats.evaluate(builder, nra[i], threshold);
		//	PredictionStatistics prestats = evaluatorPred.evaluate(builder) ;


			System.out.println("****************** Test*******************");
			System.out.println("number of recommanded items: "+nra[i]);
			System.out.println("Threshold: "+threshold);
			System.out.println();
			System.out.println("Precision: "+irstats.getPrecision());
			System.out.println("Recall: "+irstats.getRecall());
//			System.out.println("MAE= "+prestats.getMAE());
//			System.out.println("RMSE= "+prestats.getRMSE());
//			System.out.println("More info= "+prestats.getMoreInfo());
			break ;
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (TasteException e) {
			e.printStackTrace();
		}
		
	}
	
	
}
