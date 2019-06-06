package org.apache.mahout.cf.taste.impl.similarity;

import java.util.Collection;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;

public class JaccardUserSimilarity extends AbstractUserSimilarity{
	
	private final DataModel dataModel;
	private final float threshold;

	public JaccardUserSimilarity(DataModel dataModel, float threshold) throws TasteException {
		super(dataModel);
		this.dataModel = dataModel;
		this.threshold = threshold;
	}

	@Override
	public double userSimilarity(long userID1, long userID2) throws TasteException {
		FastIDSet items1 = dataModel.getItemIDsFromUser(userID1) ;
		FastIDSet items2 = dataModel.getItemIDsFromUser(userID2) ;
		long[] items1Array = items1.toArray() ;
		long[] items2Array = items2.toArray() ;
		
 
		int commonUsers=0 ;

		for(long id:items1Array) {
			Float rating = dataModel.getPreferenceValue(userID2, id);

			if (rating != null && rating >= this.threshold) {
				commonUsers++;
			}
		}
		
		int cnt = items2Array.length + items1Array.length - commonUsers;
		double similarity = (double) commonUsers / (double) cnt;
		return similarity;
	}

	@Override
	public void refresh(Collection<Refreshable> alreadyRefreshed) {
		throw new UnsupportedOperationException();	
		
	}

}
