package org.apache.mahout.cf.taste.impl.similarity;


import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.similarity.PreferenceInferrer;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import com.google.common.base.Preconditions;

public abstract class AbstractUserSimilarity implements UserSimilarity {
	private final DataModel dataModel;
	private final RefreshHelper refreshHelper;

	  protected AbstractUserSimilarity(DataModel dataModel) {
		// TODO Auto-generated constructor stub
	    Preconditions.checkArgument(dataModel != null, "dataModel is null");
	    this.dataModel = dataModel;
	    this.refreshHelper = new RefreshHelper(null);
	    refreshHelper.addDependency(this.dataModel);
	  }

	  protected DataModel getDataModel() {
	    return dataModel;
	  }

	@Override
	public void setPreferenceInferrer(PreferenceInferrer inferrer) {
		throw new UnsupportedOperationException();
	}

}
