package org.apache.main;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Properties;

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



/* ********************************************
// you should pass argument to compile this main class: FILE_NAME  accuracy||error  threshold  (nbr_recommendation)* 
 * Example: Java Main.java TestSet0 accuracy 3 10 20
 * in config.properies you can specify the path of the log file 
******************************************** */
public class Main {

	 
	public static void main(String[] args) {

		
		ParseFile p;
		float threshold=Float.parseFloat(args[2]) ;
		String rootPath= System.getProperty("user.dir") ;
		Properties config;
		config = new Properties();
		FileInputStream fis;
		
		String fileName= args[0];
		try {
			fis= new FileInputStream(rootPath+"/config.properties");
			config.load(fis);
			// path to the log file named resultLogLBNN.log
			System.setOut(new PrintStream(new FileOutputStream(rootPath+config.getProperty("PATH_TO_LBNN_LOG"))));
			
				
		} catch (IOException e1) {
			e1.printStackTrace();

		}
		
		
			File myFile= new File(rootPath+"/Datasets/"+fileName+".csv");
			System.out.println("rootPath: "+rootPath);
			p = new ParseFile();
			if(!myFile.exists()) {
				p.convertToCSV(fileName);
			}
			
			IRStatistics irstats = null ;

//			List<RecommendedItem> recommendations = recommender.recommend(2, 3);
//			for (RecommendedItem recommendation : recommendations) {
//				System.out.println(recommendation);
//			}
			try {
			DataModel model = new FileDataModel(new File(rootPath+"/Datasets/"+fileName+".csv"));
			RecommenderBuilder builder = new MyRecommenderBuilder(config.getProperty("PATH_TO_LBNNG_BY_PYTHON"), args[3], threshold); 

			
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
				int pp=4 ;
				while(pp<args.length) {
					KFoldRecommenderIRStatsEvaluator evaluatorIRStats = new KFoldRecommenderIRStatsEvaluator(model, 5); /* 5-fold */
					irstats = evaluatorIRStats.evaluate(builder, Integer.parseInt(args[pp]), threshold);
					System.out.println("****************** Precision and Recall of brute force recommenders *******************");
					System.out.println("number of recommanded items: "+args[pp]);
					System.out.println("Threshold: "+threshold);
					System.out.println();
					System.out.println("Precision: "+irstats.getPrecision());
					System.out.println("Recall: "+irstats.getRecall());
					pp++ ;
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
		}
	}
