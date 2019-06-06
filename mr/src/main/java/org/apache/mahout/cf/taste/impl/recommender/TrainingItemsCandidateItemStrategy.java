package org.apache.mahout.cf.taste.impl.recommender;

import java.util.Collection;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;

public class TrainingItemsCandidateItemStrategy implements CandidateItemsStrategy {

	private final FastIDSet possibleItemIDs;

	public TrainingItemsCandidateItemStrategy(DataModel trainingModel) throws TasteException {
		possibleItemIDs = new FastIDSet(0);
		LongPrimitiveIterator it = trainingModel.getItemIDs();
		while (it.hasNext()) {
			long itemID = it.nextLong();
			possibleItemIDs.add(itemID);
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
