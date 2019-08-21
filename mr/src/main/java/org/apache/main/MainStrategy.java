package org.apache.main;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.PredictionStatistics;
import org.apache.mahout.cf.taste.impl.eval.KFoldRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.eval.KFoldRecommenderPredictionEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.recommender.ParseFile;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.recommenders.BinderRecommenderBuilder;

public class MainStrategy {
	
public static void main(String[] args) {

		List<Double> recalls = new ArrayList<Double>() ;
		List<Double> precisions = new ArrayList<Double>() ;
		ParseFile p;
		float threshold=Float.parseFloat(args[2]) ;
		String rootPath= System.getProperty("user.dir") ;
		Properties config;
		config = new Properties();
		FileInputStream fis;
		
		String strategy="allitems" ; /* testitems or trainingitems or testratings or relplusrandom */
		String knnOUlbnn="knn" ;
		String fileName= args[0]; // dataset name
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

			try {
			DataModel model = new FileDataModel(new File(rootPath+"/Datasets/"+fileName+".csv"));
			//args[3]: graph_learning or sym_graph_learning
			BinderRecommenderBuilder strategy_builder = new BinderRecommenderBuilder(strategy, threshold, knnOUlbnn, config.getProperty("PATH_TO_LBNNG_BY_PYTHON"), args[3], 50) ;
			
			/* Measure of MAE and RMSE of recommenders */
			if(args[1].equals("error")) {
				System.out.println("****************** RMSE and MAE of brute force recommenders *******************");
					KFoldRecommenderPredictionEvaluator evaluatorPred = new KFoldRecommenderPredictionEvaluator(model, 5);
					PredictionStatistics prestats = evaluatorPred.evaluate(strategy_builder) ;
					System.out.println("MAE= "+prestats.getMAE());
					System.out.println("RMSE= "+prestats.getRMSE());
					System.out.println("More info= "+prestats.getMoreInfo());
			}
			
			
			/* Measure of Precision and Recall of recommenders */
			else if(args[1].equals("accuracy")) {
				int pp=4 ;
				while(pp<args.length) {
					KFoldRecommenderIRStatsEvaluator evaluatorIRStats = new KFoldRecommenderIRStatsEvaluator(model, 5); /* 5-fold */
					irstats = evaluatorIRStats.evaluate(strategy_builder, Integer.parseInt(args[pp]), threshold);
					recalls.add(irstats.getRecall()) ;
					precisions.add(irstats.getPrecision()) ;
					pp++ ;
				}
				System.out.println("****************** Precision and Recall of brute force recommenders *******************");
				System.out.println("number of recommanded items: "+pp);
				System.out.println("Threshold: "+threshold);
				System.out.println("Recalls: ");
				for(int i=0;i<recalls.size();i++) {
					System.out.print(", "+recalls.get(i));
				}
				System.out.println("Precisions: ");

				for(int j=0;j<precisions.size();j++) {
					System.out.print(","+precisions.get(j));

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
