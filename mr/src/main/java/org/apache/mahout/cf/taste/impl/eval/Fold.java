package org.apache.mahout.cf.taste.impl.eval;

import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

public class Fold {
	
	private static final Logger log = LoggerFactory.getLogger(Fold.class);
	
	private final DataModel trainingSet;
	private final FastByIDMap<PreferenceArray> testingSet;
	private FastIDSet userIDs;
	
	public Fold(FastByIDMap<PreferenceArray> training, FastByIDMap<PreferenceArray> testing) {
		Preconditions.checkArgument(training != null, "training is null");
		Preconditions.checkArgument(testing != null, "testing is null");
		
		this.trainingSet = new GenericDataModel(training);
		this.testingSet = testing;
		this.userIDs = new FastIDSet();
		
		LongPrimitiveIterator it = this.testingSet.keySetIterator();
		while (it.hasNext()) {
			long userID = it.nextLong();
			this.userIDs.add(userID);
		}
	}
	
	public DataModel getTraining() {
		return this.trainingSet;
	}
	
	public FastByIDMap<PreferenceArray> getTesting() {
		return this.testingSet;
	}
	
	public FastIDSet getUserIDs() {
		return this.userIDs;
	}
	
	public void removeUserIDs(FastIDSet ids) {
		log.info("Has {} users registered in testing fold before removal", this.userIDs.size());
		this.userIDs.removeAll(ids);
		log.info("Has {} users registered in testing fold after removal", this.userIDs.size());
	}

}
