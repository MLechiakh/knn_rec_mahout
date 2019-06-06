package org.apache.mahout.cf.taste.impl.eval;

import org.apache.mahout.cf.taste.eval.PredictionStatistics;

import com.google.common.base.Preconditions;

public class PredictionStatisticsImpl implements PredictionStatistics {
	
	private final double mae;
	private final double rmse;
	private final double noEstimate;
	private final String info;

	PredictionStatisticsImpl(double mae, double rmse, double noEst) {
		this(mae, rmse, noEst, null);
	}
	
	PredictionStatisticsImpl(double mae, double rmse, double noEst, String info) {
		Preconditions.checkArgument(Double.isNaN(mae) || mae >= 0.0, "Illegal mae: " + mae + ". Must be >= 0.0 or NaN");
		Preconditions.checkArgument(Double.isNaN(rmse) || rmse >= 0.0, "Illegal rmse: " + rmse + ". Must be >= 0.0 or NaN");
		Preconditions.checkArgument(Double.isNaN(noEst) || noEst >= 0.0, "Illegal noEst: " + noEst + ". Must be >= 0.0 or NaN");
		this.mae = mae;
		this.rmse = rmse;
		this.noEstimate = noEst;
		this.info = info;
	}

	@Override
	public double getRMSE() {
		return this.rmse;
	}

	@Override
	public double getMAE() {
		return this.mae;
	}

	@Override
	public double getNoEstimate() {
		return this.noEstimate;
	}

	@Override
	public String getMoreInfo() {
		return info;
	}

}
