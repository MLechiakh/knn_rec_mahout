package org.apache.mahout.cf.taste.eval;


public interface PredictionStatistics {
  
  double getRMSE();
  double getMAE();
  double getNoEstimate();
  String getMoreInfo();
  
}
