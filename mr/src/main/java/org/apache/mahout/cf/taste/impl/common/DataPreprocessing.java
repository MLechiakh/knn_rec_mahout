package org.apache.mahout.cf.taste.impl.common;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;

public class DataPreprocessing {
	
	public static DataModel normalize(DataModel model) throws TasteException {
		
		FastByIDMap<PreferenceArray> userData = new FastByIDMap<PreferenceArray>(model.getNumUsers());
		LongPrimitiveIterator it;
		it = model.getUserIDs();
		while (it.hasNext()) {
			long userID = it.nextLong();
			float sum = 0;
			int cnt = 0;
			for (Preference pref : model.getPreferencesFromUser(userID)) {
				sum += pref.getValue();
				cnt++;
			}
			if (cnt > 0) {
				float mean = sum / (float) cnt;
				PreferenceArray a = new GenericUserPreferenceArray(cnt);
				int id = 0;
				for (Preference pref : model.getPreferencesFromUser(userID)) {
					a.set(id, new GenericPreference(userID, pref.getItemID(), pref.getValue() - mean));
					id++;
				}
				userData.put(userID, a);
			}
		}
		return new GenericDataModel(userData);
		
	}
	
	public static DataModel binarize(DataModel model, float threshold) throws TasteException {
		
		FastByIDMap<PreferenceArray> userData = new FastByIDMap<PreferenceArray>(model.getNumUsers());
		LongPrimitiveIterator it;
		it = model.getUserIDs();
		while (it.hasNext()) {
			long userID = it.nextLong();
			PreferenceArray a = new GenericUserPreferenceArray(model.getPreferencesFromUser(userID).length());
			int id = 0;
			for (Preference pref : model.getPreferencesFromUser(userID)) {
				a.set(id, new GenericPreference(userID, pref.getItemID(), pref.getValue() > threshold ? 1 : 0));
				id++;
			}
			userData.put(userID, a);
		}
		return new GenericDataModel(userData);
		
	}
	
public static DataModel discretize(DataModel model, int shift) throws TasteException {
		
		FastByIDMap<PreferenceArray> userData = new FastByIDMap<PreferenceArray>(model.getNumUsers());
		LongPrimitiveIterator it;
		it = model.getUserIDs();
		while (it.hasNext()) {
			long userID = it.nextLong();
			PreferenceArray a = new GenericUserPreferenceArray(model.getPreferencesFromUser(userID).length());
			int id = 0;
			for (Preference pref : model.getPreferencesFromUser(userID)) {
				a.set(id, new GenericPreference(userID, pref.getItemID(), Math.round(pref.getValue()) + shift));
				id++;
			}
			userData.put(userID, a);
		}
		return new GenericDataModel(userData);
		
	}

}
