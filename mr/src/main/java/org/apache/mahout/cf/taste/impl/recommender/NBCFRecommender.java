package org.apache.mahout.cf.taste.impl.recommender;

import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.Callable;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.Bicluster;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.neighborhood.UserBiclusterNeighborhood;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.similarity.UserBiclusterSimilarity;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

public class NBCFRecommender extends AbstractRecommender {

	private static final Logger log = LoggerFactory.getLogger(NBCFRecommender.class);

	private final UserBiclusterNeighborhood neighborhood;
	private final UserBiclusterSimilarity similarity;
	private final RefreshHelper refreshHelper;
	private EstimatedPreferenceCapper capper;

	public NBCFRecommender(DataModel dataModel, UserBiclusterNeighborhood neighborhood,
			UserBiclusterSimilarity similarity) {
		super(dataModel);
		Preconditions.checkArgument(neighborhood != null, "neighborhood is null");
		this.neighborhood = neighborhood;
		this.similarity = similarity;
		this.refreshHelper = new RefreshHelper(new Callable<Void>() {
			@Override
			public Void call() {
				capper = buildCapper();
				return null;
			}
		});
		refreshHelper.addDependency(dataModel);
		refreshHelper.addDependency(similarity);
		refreshHelper.addDependency(neighborhood);
		capper = buildCapper();
	}
	
	public NBCFRecommender(DataModel dataModel, UserBiclusterNeighborhood neighborhood,
			UserBiclusterSimilarity similarity, CandidateItemsStrategy strategy) {
		super(dataModel, strategy);
		Preconditions.checkArgument(neighborhood != null, "neighborhood is null");
		this.neighborhood = neighborhood;
		this.similarity = similarity;
		this.refreshHelper = new RefreshHelper(new Callable<Void>() {
			@Override
			public Void call() {
				capper = buildCapper();
				return null;
			}
		});
		refreshHelper.addDependency(dataModel);
		refreshHelper.addDependency(similarity);
		refreshHelper.addDependency(neighborhood);
		capper = buildCapper();
	}

	public UserBiclusterSimilarity getSimilarity() {
		return similarity;
	}

	@Override
	public List<RecommendedItem> recommend(long userID, int howMany, IDRescorer rescorer, boolean includeKnownItems)
			throws TasteException {
		Preconditions.checkArgument(howMany >= 0, "howMany must be at least 0");

		log.debug("Recommending items for user ID '{}'", userID);

		List<Bicluster<Long>> theNeighborhood = neighborhood.getUserNeighborhood(userID);
		
		if (howMany == 0) {
			return Collections.emptyList();
		}

		if (theNeighborhood.size() == 0) {
			return Collections.emptyList();
		}

		FastIDSet allItemIDs = getCandidateItems(theNeighborhood, userID, includeKnownItems);

		TopItems.Estimator<Long> estimator = new Estimator(userID, theNeighborhood);

		List<RecommendedItem> topItems = TopItems.getTopItems(howMany, allItemIDs.iterator(), rescorer, estimator);

		log.debug("Recommendations are: {}", topItems);
		return topItems;
	}

	@Override
	public float estimatePreference(long userID, long itemID) throws TasteException {
		DataModel model = getDataModel();
		Float actualPref = model.getPreferenceValue(userID, itemID);
		if (actualPref != null) {
			return actualPref;
		}
		List<Bicluster<Long>> theNeighborhood = neighborhood.getUserNeighborhood(userID);
		return doEstimatePreference(userID, theNeighborhood, itemID);
	}

	protected float doEstimatePreference(long userID, List<Bicluster<Long>> theNeighborhood, long itemID)
			throws TasteException {
		if (theNeighborhood.size() == 0) {
			return Float.NaN;
		}
		DataModel dataModel = getDataModel();
		double preference = 0.0;
		double totalSimilarity = 0.0;
		double meanRating = 0;
		int meanRatingCnt = 0;
		for (Preference pref : dataModel.getPreferencesFromUser(userID)) {
			meanRating += pref.getValue();
			meanRatingCnt++;
		}
		if (meanRatingCnt > 0) {
			meanRating = meanRating / (double) meanRatingCnt;
		} else {
			return Float.NaN;
		}

		for (Bicluster<Long> bicluster : theNeighborhood) {
			if (bicluster.containsItem(itemID)) {
				double meanRatingBicluster = 0.0;
				int meanRatingBiclusterCnt = 0;
				Iterator<Long> users = bicluster.getUsers();
				while (users.hasNext()) {
					long theuserID = users.next();
					if (theuserID != userID) {
						Float pref = dataModel.getPreferenceValue(theuserID, itemID);
						if (pref != null) {
							meanRatingBicluster += pref;
							meanRatingBiclusterCnt++;
						}
					}
				}
				if (meanRatingBiclusterCnt > 0) {
					meanRatingBicluster = meanRatingBicluster / (double) meanRatingBiclusterCnt;
					double theSimilarity = similarity.userBiclusterSimilarity(userID, bicluster);
					if (!Double.isNaN(theSimilarity)) {
						preference += theSimilarity * (meanRatingBicluster - meanRating);
						totalSimilarity += Math.abs(theSimilarity);
					}
				}
			}
		}
		float estimate = (float) meanRating;
		if (totalSimilarity > 0) {
			estimate += (float) preference / totalSimilarity;
		}
		if (capper != null) {
			estimate = capper.capEstimate(estimate);
		}
		return estimate;
	}

	protected FastIDSet getCandidateItems(List<Bicluster<Long>> theNeighborhood, long theUserID,
			boolean includeKnownItems) throws TasteException {
		DataModel dataModel = getDataModel();
		FastIDSet possibleItemIDs = new FastIDSet();
		for (Bicluster<Long> bicluster : theNeighborhood) {
			Iterator<Long> it = bicluster.getItems();
			while (it.hasNext()) {
				long itemID = it.next();
				possibleItemIDs.add(itemID);
			}
		}
		if (!includeKnownItems) {
			possibleItemIDs.removeAll(dataModel.getItemIDsFromUser(theUserID));
		}
		return possibleItemIDs;
	}

	@Override
	public void refresh(Collection<Refreshable> alreadyRefreshed) {
		refreshHelper.refresh(alreadyRefreshed);
	}

	@Override
	public String toString() {
		return "NBCFRecommender[neighborhood:" + neighborhood + ']';
	}

	private EstimatedPreferenceCapper buildCapper() {
		DataModel dataModel = getDataModel();
		if (Float.isNaN(dataModel.getMinPreference()) && Float.isNaN(dataModel.getMaxPreference())) {
			return null;
		} else {
			return new EstimatedPreferenceCapper(dataModel);
		}
	}

	private final class Estimator implements TopItems.Estimator<Long> {

		private final long theUserID;
		private final List<Bicluster<Long>> theNeighborhood;

		Estimator(long theUserID, List<Bicluster<Long>> theNeighborhood) {
			this.theUserID = theUserID;
			this.theNeighborhood = theNeighborhood;
		}

		@Override
		public double estimate(Long itemID) throws TasteException {
			return doEstimatePreference(theUserID, theNeighborhood, itemID);
		}
	}
}
