/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.cf.taste.impl.eval;

import java.io.Serializable;

import org.apache.mahout.cf.taste.eval.IRStatistics;

import com.google.common.base.Preconditions;

public final class IRStatisticsImpl implements IRStatistics, Serializable {

	private final double precision;
	private final double recall;
	private final double fallOut;
	private final double ndcg;
	private final double reachAtLeastOne;
	private final double reachAll;
	private final double itemCoverage;
	private final double perPrecision;
	private final double perRecall;

	IRStatisticsImpl(double precision, double recall, double fallOut, double ndcg, double reachAtLeastOne,
			double reachAll, double itemCoverage, double perPrecision, double perRecall) {
		Preconditions.checkArgument(Double.isNaN(precision) || (precision >= 0.0 && precision <= 1.0),
				"Illegal precision: " + precision + ". Must be: 0.0 <= precision <= 1.0 or NaN");
		Preconditions.checkArgument(Double.isNaN(recall) || (recall >= 0.0 && recall <= 1.0),
				"Illegal recall: " + recall + ". Must be: 0.0 <= recall <= 1.0 or NaN");
		Preconditions.checkArgument(Double.isNaN(fallOut) || (fallOut >= 0.0 && fallOut <= 1.0),
				"Illegal fallOut: " + fallOut + ". Must be: 0.0 <= fallOut <= 1.0 or NaN");
		Preconditions.checkArgument(Double.isNaN(ndcg) || (ndcg >= 0.0 && ndcg <= 1.0),
				"Illegal nDCG: " + ndcg + ". Must be: 0.0 <= nDCG <= 1.0 or NaN");
		Preconditions.checkArgument(Double.isNaN(reachAtLeastOne) || (reachAtLeastOne >= 0.0 && reachAtLeastOne <= 1.0),
				"Illegal reachAtLeastOne: " + reachAtLeastOne + ". Must be: 0.0 <= reachAtLeastOne <= 1.0 or NaN");
		Preconditions.checkArgument(Double.isNaN(reachAll) || (reachAll >= 0.0 && reachAll <= 1.0),
				"Illegal reachAll: " + reachAll + ". Must be: 0.0 <= reachAll <= 1.0 or NaN");
		Preconditions.checkArgument(Double.isNaN(itemCoverage) || (itemCoverage >= 0.0 && itemCoverage <= 1.0),
				"Illegal itemCoverage: " + itemCoverage + ". Must be: 0.0 <= itemCoverage <= 1.0 or NaN");
		Preconditions.checkArgument(Double.isNaN(perPrecision) || (perPrecision >= 0.0 && perPrecision <= 1.0),
				"Illegal perPrecision: " + perPrecision + ". Must be: 0.0 <= perPrecision <= 1.0 or NaN");
		Preconditions.checkArgument(Double.isNaN(perRecall) || (perRecall >= 0.0 && perRecall <= 1.0),
				"Illegal perRecall: " + perRecall + ". Must be: 0.0 <= perRecall <= 1.0 or NaN");
		this.precision = precision;
		this.recall = recall;
		this.fallOut = fallOut;
		this.ndcg = ndcg;
		this.reachAtLeastOne = reachAtLeastOne;
		this.reachAll = reachAll;
		this.itemCoverage = itemCoverage;
		this.perPrecision = perPrecision;
		this.perRecall = perRecall;
	}

	IRStatisticsImpl(double precision, double recall, double fallOut, double ndcg, double reach) {
		this(precision, recall, fallOut, ndcg, reach, Double.NaN, Double.NaN, Double.NaN, Double.NaN);
	}

	@Override
	public double getPrecision() {
		return precision;
	}

	@Override
	public double getRecall() {
		return recall;
	}

	@Override
	public double getFallOut() {
		return fallOut;
	}

	@Override
	public double getF1Measure() {
		return getFNMeasure(1.0);
	}

	@Override
	public double getFNMeasure(double b) {
		double b2 = b * b;
		double sum = b2 * precision + recall;
		return sum == 0.0 ? Double.NaN : (1.0 + b2) * precision * recall / sum;
	}

	@Override
	public double getNormalizedDiscountedCumulativeGain() {
		return ndcg;
	}

	@Override
	public double getReachAtLeastOne() {
		return reachAtLeastOne;
	}

	@Override
	public double getReachAll() {
		return reachAll;
	}

	@Override
	public double getItemCoverage() {
		return itemCoverage;
	}

	@Override
	public double getPerPrecision() {
		return perPrecision;
	}

	@Override
	public double getPerRecall() {
		return perRecall;
	}

	@Override
	public String toString() {
		return "IRStatisticsImpl[precision:" + precision + ",recall:" + recall + ",fallOut:" + fallOut + ",nDCG:" + ndcg
				+ ",reachAtLeastOne:" + reachAtLeastOne + ",reachAll:" + reachAll + ']';
	}

}
