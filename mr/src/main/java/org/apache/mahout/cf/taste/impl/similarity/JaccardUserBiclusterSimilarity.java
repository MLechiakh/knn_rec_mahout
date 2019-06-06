package org.apache.mahout.cf.taste.impl.similarity;

import java.util.Collection;
import java.util.Iterator;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.Bicluster;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.similarity.PreferenceInferrer;
import org.apache.mahout.cf.taste.similarity.UserBiclusterSimilarity;

public class JaccardUserBiclusterSimilarity implements UserBiclusterSimilarity {
	
	private final DataModel dataModel;
	private final double threshold;
	
	public JaccardUserBiclusterSimilarity(DataModel dataModel) throws TasteException {
		this.dataModel = dataModel;
		this.threshold = 0;
	}
	
	@Override
	public double userBiclusterSimilarity(long userID, Bicluster<Long> bicluster) throws TasteException {
		int commonItems = 0;
		int cnt = bicluster.getNbItems();
		Iterator<Long> it = bicluster.getItems();
		while (it.hasNext()) {
			long itemID = it.next();
			Float pref = this.dataModel.getPreferenceValue(userID, itemID);
			if (pref != null && pref > this.threshold) {
				commonItems++;
			}
		}
		double similarity = (double) commonItems / (double) cnt;
		return similarity;
	}

	@Override
	public void refresh(Collection<Refreshable> alreadyRefreshed) {
		// Do nothing
	}

	@Override
	public void setPreferenceInferrer(PreferenceInferrer inferrer) {
		throw new UnsupportedOperationException();
	}

}
