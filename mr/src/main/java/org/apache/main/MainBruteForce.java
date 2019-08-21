package org.apache.main;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Properties;

import org.apache.hadoop.yarn.webapp.example.MyApp.MyController;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.PredictionStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.eval.KFoldRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.eval.KFoldRecommenderPredictionEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.recommenders.MyRecommenderBuilder1;

public class MainBruteForce {

	public static void main(String[] args) {
		Properties config;
		config = new Properties();
		FileInputStream fis;
		float threshold=Float.parseFloat(args[3]) ;
		String rootPath= System.getProperty("user.dir") ;
		

		String fileName= args[0];
		int k = Integer.parseInt(args[2]) ;
		try {
			fis= new FileInputStream(rootPath+"/config.properties");
			config.load(fis);
			System.out.println("*********"+rootPath+config.getProperty("PATH_TO_BRUTE_LOG"));

			System.setOut(new PrintStream(new FileOutputStream(rootPath+config.getProperty("PATH_TO_BRUTE_LOG"))));
			
			//System.setOut(new PrintStream(new FileOutputStream("C:\\Users\\Moham\\git\\mahout\\mr\\logs\\resultLogBrute.log")));
				
		} catch (IOException e1) {
			e1.printStackTrace();
		}
		
		DataModel model = null ;
		IRStatistics irstats = null ;
		
		try {
			
			model = new FileDataModel(new File(rootPath+"/Datasets/"+fileName+".csv"));
			RecommenderBuilder builder = new MyRecommenderBuilder1(model,threshold, k);

			
			/* Measure of MAE and RMSE of recommenders */
			if(args[1].equals("error")) {
				System.out.println("****************** RMSE and MAE of brute force recommenders *******************");
					KFoldRecommenderPredictionEvaluator evaluatorPred = new KFoldRecommenderPredictionEvaluator(model, 5);
					PredictionStatistics prestats = evaluatorPred.evaluate(builder) ;
					System.out.println("MAE= "+prestats.getMAE());
					System.out.println("RMSE= "+prestats.getRMSE());
					System.out.println("More info= "+prestats.getMoreInfo());
			}
			
			/* Measure of Precision and Recall of recommenders */
			else if(args[1].equals("accuracy")) {
				int p=4 ;
				while(p<args.length) {
					KFoldRecommenderIRStatsEvaluator evaluatorIRStats = new KFoldRecommenderIRStatsEvaluator(model, 5); /* 5-fold */
					irstats = evaluatorIRStats.evaluate(builder, Integer.parseInt(args[p]), threshold);
					System.out.println("****************** Precision and Recall of brute force recommenders *******************");
					System.out.println("number of recommanded items: "+args[p]);
					System.out.println("Threshold: "+threshold);
					System.out.println("k : "+k);
					System.out.println();
					System.out.println("Precision: "+irstats.getPrecision());
					System.out.println("Recall: "+irstats.getRecall());
					p++ ;
				}
			}
			
			else {
				System.out.println("Please, you should pass in arguments what kind of measures you want to perform");
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		}catch (TasteException e) {
			e.printStackTrace();
			
		}
//
	}

}
