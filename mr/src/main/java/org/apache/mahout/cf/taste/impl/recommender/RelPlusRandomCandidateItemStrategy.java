package org.apache.mahout.cf.taste.impl.recommender;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;

public class RelPlusRandomCandidateItemStrategy implements CandidateItemsStrategy {

	private final FastByIDMap<PreferenceArray> testRatings;
	private final DataModel dataModel;
	private final float threshold;
	private final Random random;

	public RelPlusRandomCandidateItemStrategy(FastByIDMap<PreferenceArray> testRatings, DataModel dataModel,
			float threshold) {
		this.testRatings = testRatings;
		this.dataModel = dataModel;
		this.threshold = threshold;
		this.random = RandomUtils.getRandom();
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
			if (pref.getValue() >= this.threshold) {
				possibleItemIDs.add(pref.getItemID());
			}
		}
		List<Long> possibleRandomItemsIDs = new ArrayList<Long>(this.dataModel.getNumItems());
		LongPrimitiveIterator it = this.dataModel.getItemIDs();
		while (it.hasNext()) {
			long itemID = it.nextLong();
			if (!preferencesFromUser.hasPrefWithItemID(itemID)) {
				possibleRandomItemsIDs.add(itemID);
			}
		}
		Collections.shuffle(possibleRandomItemsIDs, this.random);
		int nbRandomItems = Math.min(10 * possibleItemIDs.size(), possibleRandomItemsIDs.size());
		for (int i = 0; i < nbRandomItems; i++) {
			possibleItemIDs.add(possibleRandomItemsIDs.get(i));
		}
		return possibleItemIDs;
	}

}
