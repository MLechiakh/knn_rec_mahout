package org.apache.mahout.cf.taste.impl.similarity;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Weighting;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.impl.common.AbstractCountMinSketch;
import org.apache.mahout.cf.taste.impl.common.DoubleCountMinSketch;
import org.apache.mahout.cf.taste.impl.common.HashFunctionBuilder;
import org.apache.mahout.cf.taste.impl.common.CountMinSketchConfig;

import com.google.common.base.Preconditions;

import java.util.HashMap;


public final class CosineCM extends AbstractSimilarity {
  
  private final HashMap<Long, DoubleCountMinSketch> sketches;
  private final HashFunctionBuilder hfBuilder;
  private final CountMinSketchConfig config;

  /**
   * @throws IllegalArgumentException if {@link DataModel} does not have preference values
   */
  public CosineCM(DataModel dataModel, CountMinSketchConfig conf, HashFunctionBuilder hfBuilder_) throws TasteException {
    this(dataModel, Weighting.UNWEIGHTED, conf, hfBuilder_);
  }

  /**
   * @throws IllegalArgumentException if {@link DataModel} does not have preference values
   */
  public CosineCM(DataModel dataModel, Weighting weighting, CountMinSketchConfig conf, HashFunctionBuilder hfBuilder_) throws TasteException {
    super(dataModel, weighting, false);
    config = conf;
    hfBuilder = hfBuilder_;
    sketches = new HashMap<Long, DoubleCountMinSketch>(dataModel.getNumUsers());
    Preconditions.checkArgument(dataModel.hasPreferenceValues(), "DataModel doesn't have preference values");
  }
  
  private DoubleCountMinSketch exportProfile(long userID, double delta, double epsilon) throws TasteException {
		DoubleCountMinSketch cm = null;
		try{
			cm = new DoubleCountMinSketch(delta, epsilon, hfBuilder);
		} catch(AbstractCountMinSketch.CMException ex) {
			throw new TasteException("CountMinSketch error:" + ex.getMessage());
		}
      
		DataModel dataModel = getDataModel();
		PreferenceArray prefs = dataModel.getPreferencesFromUser(userID);
		int length = prefs.length();
		for (int i = 0; i < length; i++) {
			long index = prefs.getItemID(i);
			double x = prefs.getValue(i);
			cm.update(index, x);
		}
		return cm;
	}
  
  public DoubleCountMinSketch getExportedCMProfile(long userID) throws TasteException {
    DoubleCountMinSketch cm = sketches.get(userID);
    if (cm == null) {
      cm = exportProfile(userID, config.getDelta(userID), config.getEpsilon(userID));
      sketches.put(userID, cm);
    }
    return cm;
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
  
  @Override
  public double userSimilarity(long userID1, long userID2) throws TasteException {
    
    DoubleCountMinSketch cm1 = exportProfile(userID1, config.getDelta(userID2), config.getEpsilon(userID2));
    DoubleCountMinSketch cm2 = getExportedCMProfile(userID2);
    
    double result = DoubleCountMinSketch.cosine(cm1, cm2);
    
    if (!Double.isNaN(result)) {
      result = normalizeWeightResult(result, 1, 0);
    }
    return result;
    
  }

}
