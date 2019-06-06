package org.apache.mahout.cf.taste.impl.recommender;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Callable;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.neighborhood.ItemNeighborhood;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.ItemBasedRecommender;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Rescorer;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.common.LongPair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

public class NeighborhoodItemBasedRecommender extends AbstractRecommender implements ItemBasedRecommender {

	private static final Logger log = LoggerFactory.getLogger(NeighborhoodItemBasedRecommender.class);

	private final ItemNeighborhood neighborhood;
	private final ItemSimilarity similarity;
	private final RefreshHelper refreshHelper;
	private EstimatedPreferenceCapper capper;

	public NeighborhoodItemBasedRecommender(DataModel dataModel, ItemSimilarity similarity,
			ItemNeighborhood neighborhood, CandidateItemsStrategy candidateItemsStrategy) {
		super(dataModel, candidateItemsStrategy);
		Preconditions.checkArgument(similarity != null, "similarity is null");
		this.similarity = similarity;
		Preconditions.checkArgument(neighborhood != null, "neighborhood is null");
		this.neighborhood = neighborhood;
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

	public NeighborhoodItemBasedRecommender(DataModel dataModel, ItemSimilarity similarity,
			ItemNeighborhood neighborhood) {
		this(dataModel, similarity, neighborhood, AbstractRecommender.getDefaultCandidateItemsStrategy());
	}

	public ItemSimilarity getSimilarity() {
		return similarity;
	}

	@Override
	public List<RecommendedItem> recommend(long userID, int howMany, IDRescorer rescorer, boolean includeKnownItems)
			throws TasteException {
		Preconditions.checkArgument(howMany >= 0, "howMany must be at least 0");
		log.debug("Recommending items for user ID '{}'", userID);

		long[] theNeighborhood = neighborhood.getItemNeighborhood(userID);

		if (howMany == 0) {
			return Collections.emptyList();
		}

		if (theNeighborhood.length == 0) {
			return Collections.emptyList();
		}

		PreferenceArray preferencesFromUser = getDataModel().getPreferencesFromUser(userID);
		if (preferencesFromUser.length() == 0) {
			return Collections.emptyList();
		}

		FastIDSet possibleItemIDs = getAllOtherItems(userID, preferencesFromUser, includeKnownItems);

		TopItems.Estimator<Long> estimator = new Estimator(userID, preferencesFromUser, theNeighborhood);

		List<RecommendedItem> topItems = TopItems.getTopItems(howMany, possibleItemIDs.iterator(), rescorer, estimator);

		log.debug("Recommendations are: {}", topItems);
		return topItems;
	}

	@Override
	public float estimatePreference(long userID, long itemID) throws TasteException {
		PreferenceArray preferencesFromUser = getDataModel().getPreferencesFromUser(userID);
		Float actualPref = getPreferenceForItem(preferencesFromUser, itemID);
		if (actualPref != null) {
			return actualPref;
		}
		long[] theNeighborhood = neighborhood.getItemNeighborhood(itemID);
		return doEstimatePreference(userID, preferencesFromUser, itemID, theNeighborhood);
	}

	private static Float getPreferenceForItem(PreferenceArray preferencesFromUser, long itemID) {
		int size = preferencesFromUser.length();
		for (int i = 0; i < size; i++) {
			if (preferencesFromUser.getItemID(i) == itemID) {
				return preferencesFromUser.getValue(i);
			}
		}
		return null;
	}

	@Override
	public List<RecommendedItem> mostSimilarItems(long itemID, int howMany) throws TasteException {
		return mostSimilarItems(itemID, howMany, null);
	}

	@Override
	public List<RecommendedItem> mostSimilarItems(long itemID, int howMany, Rescorer<LongPair> rescorer)
			throws TasteException {
		TopItems.Estimator<Long> estimator = new MostSimilarEstimator(itemID, similarity, rescorer);
		return doMostSimilarItems(new long[] { itemID }, howMany, estimator);
	}

	@Override
	public List<RecommendedItem> mostSimilarItems(long[] itemIDs, int howMany) throws TasteException {
		TopItems.Estimator<Long> estimator = new MultiMostSimilarEstimator(itemIDs, similarity, null);
		return doMostSimilarItems(itemIDs, howMany, estimator);
	}

	@Override
	public List<RecommendedItem> mostSimilarItems(long[] itemIDs, int howMany, Rescorer<LongPair> rescorer)
			throws TasteException {
		TopItems.Estimator<Long> estimator = new MultiMostSimilarEstimator(itemIDs, similarity, rescorer);
		return doMostSimilarItems(itemIDs, howMany, estimator);
	}

	@Override
	public List<RecommendedItem> mostSimilarItems(long[] itemIDs, int howMany, boolean excludeItemIfNotSimilarToAll)
			throws TasteException {
		TopItems.Estimator<Long> estimator = new MultiMostSimilarEstimator(itemIDs, similarity, null);
		return doMostSimilarItems(itemIDs, howMany, estimator);
	}

	@Override
	public List<RecommendedItem> mostSimilarItems(long[] itemIDs, int howMany, Rescorer<LongPair> rescorer,
			boolean excludeItemIfNotSimilarToAll) throws TasteException {
		TopItems.Estimator<Long> estimator = new MultiMostSimilarEstimator(itemIDs, similarity, rescorer);
		return doMostSimilarItems(itemIDs, howMany, estimator);
	}

	@Override
	public List<RecommendedItem> recommendedBecause(long userID, long itemID, int howMany) throws TasteException {
		Preconditions.checkArgument(howMany >= 1, "howMany must be at least 1");

		DataModel model = getDataModel();
		TopItems.Estimator<Long> estimator = new RecommendedBecauseEstimator(userID, itemID);

		PreferenceArray prefs = model.getPreferencesFromUser(userID);
		int size = prefs.length();
		FastIDSet allUserItems = new FastIDSet(size);
		for (int i = 0; i < size; i++) {
			allUserItems.add(prefs.getItemID(i));
		}
		allUserItems.remove(itemID);

		return TopItems.getTopItems(howMany, allUserItems.iterator(), null, estimator);
	}

	private List<RecommendedItem> doMostSimilarItems(long[] itemIDs, int howMany, TopItems.Estimator<Long> estimator)
			throws TasteException {
		DataModel model = getDataModel();
		FastIDSet possibleItemIDs = new FastIDSet(model.getNumItems());
		LongPrimitiveIterator it = model.getItemIDs();
		while (it.hasNext()) {
			possibleItemIDs.add(it.nextLong());
		}
		return TopItems.getTopItems(howMany, possibleItemIDs.iterator(), null, estimator);
	}

	protected float doEstimatePreference(long userID, PreferenceArray preferencesFromUser, long itemID,
			long[] theNeighborhood) throws TasteException {
		double preference = 0.0;
		double totalSimilarity = 0.0;
		int count = 0;
		double[] similarities = similarity.itemSimilarities(itemID, preferencesFromUser.getIDs());
		for (int i = 0; i < similarities.length; i++) {
			double theSimilarity = similarities[i];
			if (!Double.isNaN(theSimilarity)) {

				boolean valid = false;
				for (long otherItemID : theNeighborhood) {
					if (valid) {
						break;
					}
					if (otherItemID == preferencesFromUser.getItemID(i)) {
						valid = true;
					}
				}

				if (valid) {
					// Weights can be negative!
					preference += theSimilarity * preferencesFromUser.getValue(i);
					totalSimilarity += theSimilarity;
					count++;
				}
			}
		}
		// Throw out the estimate if it was based on no data points, of course, but also
		// if based on
		// just one. This is a bit of a band-aid on the 'stock' item-based algorithm for
		// the moment.
		// The reason is that in this case the estimate is, simply, the user's rating
		// for one item
		// that happened to have a defined similarity. The similarity score doesn't
		// matter, and that
		// seems like a bad situation.
		if (count <= 1) {
			return Float.NaN;
		}
		float estimate = (float) (preference / totalSimilarity);
		if (capper != null) {
			estimate = capper.capEstimate(estimate);
		}
		return estimate;
	}

	@Override
	public void refresh(Collection<Refreshable> alreadyRefreshed) {
		refreshHelper.refresh(alreadyRefreshed);
	}

	@Override
	public String toString() {
		return "GenericItemBasedRecommender[similarity:" + similarity + ']';
	}

	private EstimatedPreferenceCapper buildCapper() {
		DataModel dataModel = getDataModel();
		if (Float.isNaN(dataModel.getMinPreference()) && Float.isNaN(dataModel.getMaxPreference())) {
			return null;
		} else {
			return new EstimatedPreferenceCapper(dataModel);
		}
	}

	public static class MostSimilarEstimator implements TopItems.Estimator<Long> {

		private final long toItemID;
		private final ItemSimilarity similarity;
		private final Rescorer<LongPair> rescorer;

		public MostSimilarEstimator(long toItemID, ItemSimilarity similarity, Rescorer<LongPair> rescorer) {
			this.toItemID = toItemID;
			this.similarity = similarity;
			this.rescorer = rescorer;
		}

		@Override
		public double estimate(Long itemID) throws TasteException {
			LongPair pair = new LongPair(toItemID, itemID);
			if (rescorer != null && rescorer.isFiltered(pair)) {
				return Double.NaN;
			}
			double originalEstimate = similarity.itemSimilarity(toItemID, itemID);
			return rescorer == null ? originalEstimate : rescorer.rescore(pair, originalEstimate);
		}
	}

	private final class Estimator implements TopItems.Estimator<Long> {

		private final long userID;
		private final PreferenceArray preferencesFromUser;
		private final long[] theNeighborhood;

		private Estimator(long userID, PreferenceArray preferencesFromUser, long[] theNeighborhood) {
			this.userID = userID;
			this.preferencesFromUser = preferencesFromUser;
			this.theNeighborhood = theNeighborhood;
		}

		@Override
		public double estimate(Long itemID) throws TasteException {
			return doEstimatePreference(userID, preferencesFromUser, itemID, theNeighborhood);
		}
	}

	private static final class MultiMostSimilarEstimator implements TopItems.Estimator<Long> {

		private final long[] toItemIDs;
		private final ItemSimilarity similarity;
		private final Rescorer<LongPair> rescorer;

		private MultiMostSimilarEstimator(long[] toItemIDs, ItemSimilarity similarity, Rescorer<LongPair> rescorer) {
			this.toItemIDs = toItemIDs;
			this.similarity = similarity;
			this.rescorer = rescorer;
		}

		@Override
		public double estimate(Long itemID) throws TasteException {
			RunningAverage average = new FullRunningAverage();
			double[] similarities = similarity.itemSimilarities(itemID, toItemIDs);
			for (int i = 0; i < toItemIDs.length; i++) {
				long toItemID = toItemIDs[i];
				LongPair pair = new LongPair(toItemID, itemID);
				if (rescorer != null && rescorer.isFiltered(pair)) {
					continue;
				}
				double estimate = similarities[i];
				if (rescorer != null) {
					estimate = rescorer.rescore(pair, estimate);
				}
				if (!Double.isNaN(estimate)) {
					average.addDatum(estimate);
				}
			}
			double averageEstimate = average.getAverage();
			return averageEstimate == 0 ? Double.NaN : averageEstimate;
		}
	}

	private final class RecommendedBecauseEstimator implements TopItems.Estimator<Long> {

		private final long userID;
		private final long recommendedItemID;

		private RecommendedBecauseEstimator(long userID, long recommendedItemID) {
			this.userID = userID;
			this.recommendedItemID = recommendedItemID;
		}

		@Override
		public double estimate(Long itemID) throws TasteException {
			Float pref = getDataModel().getPreferenceValue(userID, itemID);
			if (pref == null) {
				return Float.NaN;
			}
			double similarityValue = similarity.itemSimilarity(recommendedItemID, itemID);
			return (1.0 + similarityValue) * pref;
		}
	}

}
