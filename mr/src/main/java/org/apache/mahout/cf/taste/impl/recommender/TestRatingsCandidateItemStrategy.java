package org.apache.mahout.cf.taste.impl.recommender;

import java.util.Collection;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;

public class TestRatingsCandidateItemStrategy implements CandidateItemsStrategy {

	private final FastByIDMap<PreferenceArray> testRatings;

	public TestRatingsCandidateItemStrategy(FastByIDMap<PreferenceArray> testRatings) {
		this.testRatings = testRatings;
	}

	@Override
	public void refresh(Collection<Refreshable> alreadyRefreshed) {
	}

	@Override
	public FastIDSet getCandidateItems(long userID, PreferenceArray preferencesFromUser, DataModel dataModel,
			boolean includeKnownItems) throws TasteException {
		PreferenceArray prefs = this.testRatings.get(userID);
		if (prefs == null) {
			return new FastIDSet(0);
		}
		FastIDSet possibleItemIDs = new FastIDSet(prefs.length());
		for (Preference pref : prefs) {
			possibleItemIDs.add(pref.getItemID());
		}
		return possibleItemIDs;
	}

}
