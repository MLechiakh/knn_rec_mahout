package org.apache.mahout.cf.taste.impl.similarity;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;

public class JaccardItemSimilarity extends AbstractItemSimilarity {

	private final DataModel dataModel;
	private final float threshold;

	public JaccardItemSimilarity(DataModel dataModel, float threshold) throws TasteException {
		super(dataModel);
		this.dataModel = dataModel;
		this.threshold = threshold;
	}

	@Override
	public double itemSimilarity(long itemID1, long itemID2) throws TasteException {
		int commonUsers = 0;
		PreferenceArray prefs1 = dataModel.getPreferencesForItem(itemID1);
		PreferenceArray prefs2 = dataModel.getPreferencesForItem(itemID2);
		for (Preference pref : prefs1) {
			Float rating = dataModel.getPreferenceValue(pref.getUserID(), itemID2);
			if (rating != null && rating >= this.threshold) {
				commonUsers++;
			}
		}
		int cnt = prefs1.length() + prefs2.length() - commonUsers;
		double similarity = (double) commonUsers / (double) cnt;
		return similarity;
	}

	@Override
	public double[] itemSimilarities(long itemID1, long[] itemID2s) throws TasteException {
		int length = itemID2s.length;
		double[] result = new double[length];
		for (int i = 0; i < length; i++) {
			result[i] = itemSimilarity(itemID1, itemID2s[i]);
		}
		return result;
	}

}
