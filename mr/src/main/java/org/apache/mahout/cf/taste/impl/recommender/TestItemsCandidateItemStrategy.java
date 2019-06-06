package org.apache.mahout.cf.taste.impl.recommender;

import java.util.Collection;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;

public class TestItemsCandidateItemStrategy implements CandidateItemsStrategy {

	private final FastIDSet possibleItemIDs;

	public TestItemsCandidateItemStrategy(FastByIDMap<PreferenceArray> testRatings) {
		possibleItemIDs = new FastIDSet(0);
		LongPrimitiveIterator it = testRatings.keySetIterator();
		while (it.hasNext()) {
			long userID = it.nextLong();
			PreferenceArray prefs = testRatings.get(userID);
			for (Preference pref : prefs) {
				possibleItemIDs.add(pref.getItemID());
			}
		}
	}

	@Override
	public void refresh(Collection<Refreshable> alreadyRefreshed) {
	}

	@Override
	public FastIDSet getCandidateItems(long userID, PreferenceArray preferencesFromUser, DataModel dataModel,
			boolean includeKnownItems) throws TasteException {
		return this.possibleItemIDs;
	}

}
