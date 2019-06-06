package org.apache.mahout.cf.taste.impl.eval;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.FoldDataSplitter;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

final public class KFoldDataSplitter implements FoldDataSplitter {

	private final List<Fold> folds;

	public KFoldDataSplitter(DataModel dataModel, int nbFolds) throws TasteException {
		Preconditions.checkArgument(dataModel != null, "dataModel is null");
		Preconditions.checkArgument(nbFolds > 1, "nbFolds must be > 1");

		this.folds = new ArrayList<Fold>(nbFolds);

		int numUsers = dataModel.getNumUsers();

		// Initialize buckets for the number of folds
		List<FastByIDMap<PreferenceArray>> folds = new ArrayList<FastByIDMap<PreferenceArray>>();
		for (int i = 0; i < nbFolds; i++) {
			folds.add(new FastByIDMap<PreferenceArray>(1 + (int) (i / nbFolds * numUsers)));
		}

		// Split the dataModel into K folds per user
		LongPrimitiveIterator it = dataModel.getUserIDs();
		while (it.hasNext()) {
			long userID = it.nextLong();
			splitOneUsersPrefs(nbFolds, folds, userID, dataModel);
		}

		// Rotate the folds. Each time only one is used for testing and the rest
		// k-1 folds are used for training
		for (int k = 0; k < nbFolds; k++) {

			// The testing fold
			FastByIDMap<PreferenceArray> testPrefs = folds.get(k);

			// Build the training set from the remaining folds
			FastByIDMap<PreferenceArray> trainingPrefs = new FastByIDMap<PreferenceArray>(1 + numUsers);
			for (int i = 0; i < folds.size(); i++) {
				if (i != k) {
					for (Map.Entry<Long, PreferenceArray> entry : folds.get(i).entrySet()) {
						if (!trainingPrefs.containsKey(entry.getKey())) {
							trainingPrefs.put(entry.getKey(), entry.getValue());
						} else {
							List<Preference> userPreferences = new ArrayList<Preference>();
							PreferenceArray existingPrefs = trainingPrefs.get(entry.getKey());
							for (int j = 0; j < existingPrefs.length(); j++) {
								userPreferences.add(existingPrefs.get(j));
							}
							PreferenceArray newPrefs = entry.getValue();
							for (int j = 0; j < newPrefs.length(); j++) {
								userPreferences.add(newPrefs.get(j));
							}
							trainingPrefs.remove(entry.getKey());
							trainingPrefs.put(entry.getKey(), new GenericUserPreferenceArray(userPreferences));
						}
					}
				}
			}
			
			// Register the final fold
			this.folds.add(new Fold(trainingPrefs, testPrefs));
		}

	}

	@Override
	public Iterator<Fold> getFolds() {
		return this.folds.iterator();
	}

	private void splitOneUsersPrefs(int k, List<FastByIDMap<PreferenceArray>> folds, long userID, DataModel dataModel)
			throws TasteException {

		List<List<Preference>> oneUserPrefs = Lists.newArrayListWithCapacity(k + 1);
		for (int i = 0; i < k; i++) {
			oneUserPrefs.add(null);
		}

		PreferenceArray prefs = dataModel.getPreferencesFromUser(userID);
		int size = prefs.length();

		List<Preference> userPrefs = new ArrayList<>();
		Iterator<Preference> it = prefs.iterator();
		while (it.hasNext()) {
			userPrefs.add(it.next());
		}

		// Shuffle the items
		Collections.shuffle(userPrefs);

		int currentBucket = 0;
		for (int i = 0; i < size; i++) {
			if (currentBucket == k) {
				currentBucket = 0;
			}

			Preference newPref = new GenericPreference(userID, userPrefs.get(i).getItemID(),
					userPrefs.get(i).getValue());

			if (oneUserPrefs.get(currentBucket) == null) {
				oneUserPrefs.set(currentBucket, new ArrayList<Preference>());
			}
			oneUserPrefs.get(currentBucket).add(newPref);
			currentBucket++;
		}

		for (int i = 0; i < k; i++) {
			if (oneUserPrefs.get(i) != null) {
				folds.get(i).put(userID, new GenericUserPreferenceArray(oneUserPrefs.get(i)));
			}
		}

	}

}
