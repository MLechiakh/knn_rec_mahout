package org.apache.mahout.cf.taste.impl.eval;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.FoldDataSplitter;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;

public final class AverageAbsoluteDifferenceRecommenderEvaluatorKFold extends
    AbstractKFoldRecommenderEvaluator {
  
  private RunningAverage average;
  
  public AverageAbsoluteDifferenceRecommenderEvaluatorKFold(DataModel dataModel, int nbFolds) throws TasteException {
	  super(dataModel, nbFolds);
  }
  
  public AverageAbsoluteDifferenceRecommenderEvaluatorKFold(DataModel dataModel, FoldDataSplitter splitter) throws TasteException {
	  super(dataModel, splitter);
  }
  
  @Override
  protected void reset() {
    average = new FullRunningAverage();
  }
  
  @Override
  protected void processOneEstimate(float estimatedPreference, Preference realPref) {
    average.addDatum(Math.abs(realPref.getValue() - estimatedPreference));
  }
  
  @Override
  protected double computeFinalEvaluation() {
    return average.getAverage();
  }
  
  @Override
  public String toString() {
    return "AverageAbsoluteDifferenceRecommenderEvaluatorKFold";
  }
  
}
