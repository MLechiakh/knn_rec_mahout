package org.apache.mahout.cf.taste.impl.eval;

import java.util.Iterator;
import java.util.List;

import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.FoldDataSplitter;
import org.apache.mahout.cf.taste.eval.PerUserStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

public final class KFoldRecommenderPerUserEvaluator {
	
	private static final double LOG2 = Math.log(2.0);

	private static final Logger log = LoggerFactory.getLogger(KFoldRecommenderPerUserEvaluator.class);

	private final DataModel dataModel;
	private final FoldDataSplitter folds;

	public KFoldRecommenderPerUserEvaluator(DataModel dataModel, int nbFolds) throws TasteException {
		Preconditions.checkArgument(dataModel != null, "dataModel is null");
		Preconditions.checkArgument(nbFolds > 1, "nbFolds must be > 1");

		this.dataModel = dataModel;
		this.folds = new KFoldDataSplitter(this.dataModel, nbFolds);
	}

	public KFoldRecommenderPerUserEvaluator(DataModel dataModel, FoldDataSplitter splitter) throws TasteException {
		Preconditions.checkArgument(dataModel != null, "dataModel is null");
		Preconditions.checkArgument(splitter != null, "splitter is null");

		this.dataModel = dataModel;
		this.folds = splitter;
	}

	public PerUserStatistics evaluate(RecommenderBuilder recommenderBuilder, int at, Double relevanceThreshold) throws TasteException {

		Preconditions.checkArgument(recommenderBuilder != null, "recommenderBuilder is null");
		Preconditions.checkArgument(at >= 1, "at must be at least 1");
		Preconditions.checkArgument(!relevanceThreshold.isNaN(), "relevanceThreshold is NaN");
		log.info("Beginning evaluation");

		int n = this.dataModel.getNumUsers();
		FastByIDMap<RunningAverage> mae = new FastByIDMap<RunningAverage>(n);
		FastByIDMap<RunningAverage> rmse = new FastByIDMap<RunningAverage>(n);
		FastByIDMap<RunningAverage> precision = new FastByIDMap<RunningAverage>(n);
		FastByIDMap<RunningAverage> recall = new FastByIDMap<RunningAverage>(n);
		FastByIDMap<RunningAverage> ndcg = new FastByIDMap<RunningAverage>(n);
		
		Iterator<Fold> itF = this.folds.getFolds();
		while (itF.hasNext()) {

			Fold fold = itF.next();

			DataModel trainingModel = fold.getTraining();
			FastByIDMap<PreferenceArray> testPrefs = fold.getTesting();
			LongPrimitiveIterator it = fold.getUserIDs().iterator();

			Recommender recommender = recommenderBuilder.buildRecommender(trainingModel, fold);

			while (it.hasNext()) {
				
				double smae = 0;
				double srmse = 0;
				int cnt = 0;

				long userID = it.nextLong();
				PreferenceArray prefs = testPrefs.get(userID);
				if (prefs == null || prefs.length() == 0) {
					log.debug("Ignoring user {}", userID);
					continue; // Oops we excluded all prefs for the user -- just move on
				}

				for (Preference pref : prefs) {
					long itemID = pref.getItemID();
					float truth = pref.getValue();
					try {
						Float pred = recommender.estimatePreference(userID, itemID);
						if (!pred.isNaN()) {
							double x = truth - pred;
							smae += Math.abs(x);
							srmse += x * x;
							cnt++;
						}
					} catch (NoSuchUserException nsee) {
						break;
					} catch (NoSuchItemException nsie) {
						continue;
					}
				}
				
				if (cnt > 0) {
					
					srmse = Math.sqrt(srmse / (double) (cnt));
					if (!rmse.containsKey(userID)) {
						rmse.put(userID, new FullRunningAverage());
					}
					rmse.get(userID).addDatum(srmse);
					
					smae = smae / (double) (cnt);
					if (!mae.containsKey(userID)) {
						mae.put(userID, new FullRunningAverage());
					}
					mae.get(userID).addDatum(smae);
				}
				
				try {
					recommender.getCandidateItems(userID);
				} catch (NoSuchUserException nsue) {
					continue;
				}

				FastIDSet relevantItemIDs = new FastIDSet(prefs.length());
				for (int i = 0; i < prefs.length(); i++) {
					if (prefs.getValue(i) >= relevanceThreshold) {
						relevantItemIDs.add(prefs.getItemID(i));
					}
				}

				int numRelevantItems = relevantItemIDs.size();
				if (numRelevantItems <= 0) {
					log.debug("Ignoring user {}", userID);
					continue;
				}

				try {
					trainingModel.getPreferencesFromUser(userID);
				} catch (NoSuchUserException nsee) {
					log.debug("Ignoring user {}", userID);
					continue; // Oops we excluded all prefs for the user -- just move on
				}

				int numRecommendedItems = 0;
				int intersectionSize = 0;
				List<RecommendedItem> recommendedItems = recommender.recommend(userID, at);
				for (RecommendedItem recommendedItem : recommendedItems) {
					if (relevantItemIDs.contains(recommendedItem.getItemID())) {
						intersectionSize++;
					}
					numRecommendedItems++;
				}

				// Precision
				double p = 0;
				if (numRecommendedItems > 0) {
					p = (double) intersectionSize / (double) numRecommendedItems;
					if (!precision.containsKey(userID)) {
						precision.put(userID, new FullRunningAverage());
					}
					precision.get(userID).addDatum(p);
				}

				// Recall
				double r = 0;
				if (numRelevantItems > 0) {
					r = (double) intersectionSize / (double) numRelevantItems;
					if (!recall.containsKey(userID)) {
						recall.put(userID, new FullRunningAverage());
					}
					recall.get(userID).addDatum(r);
				}

				// nDCG
				// In computing, assume relevant IDs have relevance 1 and others 0
				double cumulativeGain = 0.0;
				double idealizedGain = 0.0;
				for (int i = 0; i < numRecommendedItems; i++) {
					RecommendedItem item = recommendedItems.get(i);
					double discount = 1.0 / log2(i + 2.0); // Classical formulation says log(i+1), but i is 0-based here
					if (relevantItemIDs.contains(item.getItemID())) {
						cumulativeGain += discount;
					}
					// otherwise we're multiplying discount by relevance 0 so it doesn't do anything

					// Ideally results would be ordered with all relevant ones first, so this
					// theoretical
					// ideal list starts with number of relevant items equal to the total number of
					// relevant items
					if (i < numRelevantItems) {
						idealizedGain += discount;
					}
				}
				if (idealizedGain > 0.0) {
					if (!ndcg.containsKey(userID)) {
						ndcg.put(userID, new FullRunningAverage());
					}
					ndcg.get(userID).addDatum(cumulativeGain / idealizedGain);
				}

			}

		}

		PerUserStatisticsImpl results = new PerUserStatisticsImpl(n);
		LongPrimitiveIterator it;
		
		it = rmse.keySetIterator();
		while (it.hasNext()) {
			long userID = it.nextLong();
			results.addRMSE(userID, rmse.get(userID).getAverage());
		}
		
		it = mae.keySetIterator();
		while (it.hasNext()) {
			long userID = it.nextLong();
			results.addMAE(userID, mae.get(userID).getAverage());
		}
		
		it = precision.keySetIterator();
		while (it.hasNext()) {
			long userID = it.nextLong();
			results.addPrecision(userID, precision.get(userID).getAverage());
		}
		
		it = recall.keySetIterator();
		while (it.hasNext()) {
			long userID = it.nextLong();
			results.addRecall(userID, recall.get(userID).getAverage());
		}
		
		it = ndcg.keySetIterator();
		while (it.hasNext()) {
			long userID = it.nextLong();
			results.addNDCG(userID, ndcg.get(userID).getAverage());
		}
		
		return results;
	}
	
	private static double log2(double value) {
		return Math.log(value) / LOG2;
	}

}
