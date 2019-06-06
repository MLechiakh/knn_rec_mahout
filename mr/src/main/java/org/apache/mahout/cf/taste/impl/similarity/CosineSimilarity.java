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

package org.apache.mahout.cf.taste.impl.similarity;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Weighting;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;

import com.google.common.base.Preconditions;

public final class CosineSimilarity extends AbstractSimilarity {

  /**
   * @throws IllegalArgumentException if {@link DataModel} does not have preference values
   */
  public CosineSimilarity(DataModel dataModel) throws TasteException {
    this(dataModel, Weighting.UNWEIGHTED);
  }

  /**
   * @throws IllegalArgumentException if {@link DataModel} does not have preference values
   */
  public CosineSimilarity(DataModel dataModel, Weighting weighting) throws TasteException {
    super(dataModel, weighting, false);
    Preconditions.checkArgument(dataModel.hasPreferenceValues(), "DataModel doesn't have preference values");
  }
  
  @Override
  public double userSimilarity(long userID1, long userID2) throws TasteException {
    DataModel dataModel = getDataModel();
    PreferenceArray xPrefs = dataModel.getPreferencesFromUser(userID1);
    PreferenceArray yPrefs = dataModel.getPreferencesFromUser(userID2);
    int xLength = xPrefs.length();
    int yLength = yPrefs.length();
    
    if (xLength == 0 || yLength == 0) {
      return Double.NaN;
    }
    
    double sumX2 = 0.0;
    double sumY2 = 0.0;
    double sumXY = 0.0;
    
    for (int i = 0; i < xLength; i++) {
      long xIndex = xPrefs.getItemID(i);
      double x = xPrefs.getValue(i);
      sumX2 += x * x;
      
      if (yPrefs.hasPrefWithItemID(xIndex)) {
        for (int j = 0; j < yLength; j++) {
          long yIndex = yPrefs.getItemID(j);
          if (xIndex == yIndex) {
            double y = yPrefs.getValue(j);
            sumXY += x * y;
            break;
          }
        }
      }
    }
    for (int j = 0; j < yLength; j++) {
      long yIndex = yPrefs.getItemID(j);
      double y = yPrefs.getValue(j);
      sumY2 += y * y;
    }
    
    int count = xLength + yLength;
    double result = computeResult(count, sumXY, sumX2, sumY2, 0);
    
    if (!Double.isNaN(result)) {
      result = normalizeWeightResult(result, count, count);
    }
    return result;
  }

  @Override
  double computeResult(int n, double sumXY, double sumX2, double sumY2, double sumXYdiff2) {
    if (n == 0) {
      return Double.NaN;
    }
    double denominator = Math.sqrt(sumX2) * Math.sqrt(sumY2);
    if (denominator == 0.0) {
      // One or both parties has -all- the same ratings;
      // can't really say much similarity under this measure
      return Double.NaN;
    }
    return sumXY / denominator;
  }

}
