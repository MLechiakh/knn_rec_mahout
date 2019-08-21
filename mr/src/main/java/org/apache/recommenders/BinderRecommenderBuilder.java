package org.apache.recommenders;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.eval.Fold;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.ParseFile;
import org.apache.mahout.cf.taste.impl.recommender.RelPlusRandomCandidateItemStrategy;
import org.apache.mahout.cf.taste.impl.recommender.TestItemsCandidateItemStrategy;
import org.apache.mahout.cf.taste.impl.recommender.TestRatingsCandidateItemStrategy;
import org.apache.mahout.cf.taste.impl.recommender.TrainingItemsCandidateItemStrategy;
import org.apache.mahout.cf.taste.impl.similarity.JaccardUserSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhoodImpl;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.cf.taste.similarity.UserSimilarityIml;

public class BinderRecommenderBuilder implements RecommenderBuilder{

	private  String strategy=null;
	private  float threshold=3;
	private String  algorithm=null;
	private int k=30 ;
	
	private String path_lbnng=null;
	private String lbnn=null ;

		
	public BinderRecommenderBuilder(String strategy, float threshold, String algorithm, String path_lbnng, String lbnn, int k) {
		super();
		this.strategy = strategy;
		this.threshold = threshold;
		this.algorithm = algorithm;
		
		this.path_lbnng=path_lbnng ;
		this.lbnn= lbnn ;
		this.k = k ;
	}
	
	
	@Override
	public Recommender buildRecommender(DataModel dataModel) throws TasteException {
		return null;
	}

	@Override
	public Recommender buildRecommender(DataModel dataModel, Fold fold) throws TasteException {
		CandidateItemsStrategy s = null;
		if (this.strategy.equals("testratings")) {
			s = new TestRatingsCandidateItemStrategy(fold.getTesting());
		} else if (this.strategy.equals("testitems")) {
			s = new TestItemsCandidateItemStrategy(fold.getTesting());
		} else if (this.strategy.equals("trainingitems")) {
			s = new TrainingItemsCandidateItemStrategy(fold.getTraining());
		} else if (this.strategy.equals("allitems")) {
			s = new TrainingItemsCandidateItemStrategy(dataModel);
		} else if (this.strategy.equals("relplusrandom")) {
			s = new RelPlusRandomCandidateItemStrategy(fold.getTesting(), dataModel, this.threshold);
		} else {
			System.out.println("Invalid candidate item selection strategy "+ this.strategy);
			return null;
		}
		return buildRecommender(dataModel, s, fold);
	}
	
public Recommender buildRecommender(DataModel dataModel, CandidateItemsStrategy s, Fold fold) throws TasteException {
		
		/* Recommender algorithm */
		
		if (this.algorithm.equals("knn")) { /* User-Based K-Nearest-Neighbors */
			UserSimilarity similarity = new JaccardUserSimilarity(fold.getTraining(), threshold) ;
			UserNeighborhood neighborhood= new NearestNUserNeighborhood(k, similarity, fold.getTraining()) ;
			UserBasedRecommender recommender = new GenericUserBasedRecommender(dataModel, neighborhood, similarity, s);
			return recommender;
			
			
			
		} else if (this.algorithm.equals("lbnn")) { /* Learning-based-Nearest-Neighbors */
			MyRecommenderBuilder myRec = new MyRecommenderBuilder(path_lbnng,lbnn,threshold) ;
			ParseFile p= myRec.buildIt(fold) ;
			UserNeighborhood neighborhood= new UserNeighborhoodImpl(p.getJsonObject()) ;
			UserSimilarity similarity = new UserSimilarityIml(p.getJsonObject()) ;
			UserBasedRecommender recommender = new GenericUserBasedRecommender(dataModel, neighborhood, similarity, s);
			return recommender;
			
		} else {
			System.out.println("Invalid / unimplemented recommender config type");
			return null;
		}
}

}
