package org.apache.mahout.cf.taste.impl.eval;

import org.apache.mahout.cf.taste.eval.PerUserStatistics;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;

public class PerUserStatisticsImpl implements PerUserStatistics {
	
	private final FastByIDMap<Double> rmse;
	private final FastByIDMap<Double> mae;
	private final FastByIDMap<Double> precision;
	private final FastByIDMap<Double> recall;
	private final FastByIDMap<Double> ndcg;
	private final FastIDSet userIDs;

	public PerUserStatisticsImpl(int numUsers) {
		this.rmse = new FastByIDMap<Double>(numUsers);
		this.mae = new FastByIDMap<Double>(numUsers);
		this.precision = new FastByIDMap<Double>(numUsers);
		this.recall = new FastByIDMap<Double>(numUsers);
		this.ndcg = new FastByIDMap<Double>(numUsers);
		this.userIDs = new FastIDSet(numUsers);
	}

	@Override
	public double getRMSE(long userID) {
		if (!this.rmse.containsKey(userID)) {
			return Double.NaN;
		}
		return this.rmse.get(userID);
	}

	@Override
	public double getMAE(long userID) {
		if (!this.mae.containsKey(userID)) {
			return Double.NaN;
		}
		return this.mae.get(userID);
	}

	@Override
	public double getPrecision(long userID) {
		if (!this.precision.containsKey(userID)) {
			return Double.NaN;
		}
		return this.precision.get(userID);
	}

	@Override
	public double getRecall(long userID) {
		if (!this.recall.containsKey(userID)) {
			return Double.NaN;
		}
		return this.recall.get(userID);
	}

	@Override
	public double getNormalizedDiscountedCumulativeGain(long userID) {
		if (!this.ndcg.containsKey(userID)) {
			return Double.NaN;
		}
		return this.ndcg.get(userID);
	}

	@Override
	public LongPrimitiveIterator getUserIDs() {
		return this.userIDs.iterator();
	}

	@Override
	public void addRMSE(long userID, double rmse) {
		this.rmse.put(userID, rmse);
		this.userIDs.add(userID);
	}

	@Override
	public void addMAE(long userID, double mae) {
		this.mae.put(userID, mae);
		this.userIDs.add(userID);
	}

	@Override
	public void addPrecision(long userID, double precision) {
		this.precision.put(userID, precision);
		this.userIDs.add(userID);
	}

	@Override
	public void addRecall(long userID, double recall) {
		this.recall.put(userID, recall);
		this.userIDs.add(userID);
	}

	@Override
	public void addNDCG(long userID, double ndcg) {
		this.ndcg.put(userID, ndcg);
		this.userIDs.add(userID);
	}

}
