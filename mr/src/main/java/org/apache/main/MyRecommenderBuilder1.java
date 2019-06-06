package org.apache.main;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.eval.Fold;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.JaccardUserSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

public class MyRecommenderBuilder1 implements RecommenderBuilder{
	private DataModel data= null;
	private final float threshold;
	private int k ;
	
	public MyRecommenderBuilder1(DataModel data, float threshold, int n) {
		super();
		this.data = data;
		this.threshold = threshold ;
		this.k = n ;
	}

	@Override
	public Recommender buildRecommender(DataModel dataModel, Fold fold) throws TasteException {
		UserSimilarity similarity = new JaccardUserSimilarity(fold.getTraining(), threshold) ;
		UserNeighborhood neighborhood= new NearestNUserNeighborhood(k, similarity, fold.getTraining()) ;
		UserBasedRecommender recommender = new GenericUserBasedRecommender(dataModel, neighborhood, similarity);
		return recommender;
  
		
	}

	@Override
	public Recommender buildRecommender(DataModel dataModel) throws TasteException {
		// TODO Auto-generated method stub
		return null;
	}



}
