package org.apache.mahout.cf.taste.impl.recommender;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;

import com.google.common.base.Preconditions;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.Average;
import org.apache.mahout.cf.taste.impl.common.BicaiNet;
import org.apache.mahout.cf.taste.impl.common.Bicluster;
import org.apache.mahout.cf.taste.impl.common.Biclustering;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.recommender.AbstractRecommender;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class BicaiNetRecommender extends AbstractRecommender {

	private final Biclustering<Long> biclusters;

	private static final Logger log = LoggerFactory.getLogger(BicaiNetRecommender.class);

	/**
	 * Create a recommender based on the Bic-aiNet algorithm
	 *
	 * @param dataModel
	 * @param wr        row weight
	 * @param wc        column weight
	 * @param lamnda
	 * @param maxNbIt   maximum number of iterations
	 * @param supIt     number of iterations between suppressions
	 *
	 * @throws TasteException
	 */
	public BicaiNetRecommender(DataModel dataModel, Biclustering<Long> biclusters) throws TasteException {

		super(dataModel);
		this.biclusters = biclusters;

	}
	
	public BicaiNetRecommender(DataModel dataModel, Biclustering<Long> biclusters, CandidateItemsStrategy strategy) throws TasteException {

		super(dataModel, strategy);
		this.biclusters = biclusters;

	}

	@Override
	public List<RecommendedItem> recommend(long userID, int howMany, IDRescorer rescorer, boolean includeKnownItems)
			throws TasteException {
		Preconditions.checkArgument(howMany >= 1, "howMany must be at least 1");
		log.debug("Recommending items for user ID '{}'", userID);

		PreferenceArray preferencesFromUser = getDataModel().getPreferencesFromUser(userID);
		FastIDSet possibleItemIDs = getAllOtherItems(userID, preferencesFromUser, includeKnownItems);

		List<RecommendedItem> topItems = TopItems.getTopItems(howMany, possibleItemIDs.iterator(), rescorer,
				new Estimator(userID));
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
		Bicluster<Long> bmin = null;
		double minResidue = Double.MAX_VALUE;
		Iterator<Bicluster<Long>> it = this.biclusters.iterator();
		while (it.hasNext()) {
			Bicluster<Long> b = it.next();
			if (b.containsUser(userID) && b.containsItem(itemID)) {
				double residue = BicaiNet.residue(b, model);
				if (residue <= minResidue) {
					minResidue = residue;
					bmin = b;
				}
			}
		}
		if (bmin == null) {
			return Float.NaN;
		} else {
			Average avg = new Average();
			Iterator<Long> itU = bmin.getUsers();
			while (itU.hasNext()) {
				long otherUserID = itU.next();
				Float rating = model.getPreferenceValue(otherUserID, itemID);
				float value = rating == null ? 0 : rating;
				avg.add(value);
			}
			return avg.compute();
		}

	}

	private final class Estimator implements TopItems.Estimator<Long> {

		private final long theUserID;

		private Estimator(long theUserID) {
			this.theUserID = theUserID;
		}

		@Override
		public double estimate(Long itemID) throws TasteException {
			return estimatePreference(theUserID, itemID);
		}
	}

	/**
	 * Refresh the data model and factorization.
	 */
	@Override
	public void refresh(Collection<Refreshable> alreadyRefreshed) {
	}

}
