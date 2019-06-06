package org.apache.mahout.cf.taste.impl.neighborhood;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.SamplingLongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import com.google.common.base.Preconditions;

public final class NearestNItemNeighborhood extends AbstractItemNeighborhood {
  
  private final int n;
  private final double minSimilarity;
  
  public NearestNItemNeighborhood(int n, ItemSimilarity itemSimilarity, DataModel dataModel) throws TasteException {
    this(n, Double.NEGATIVE_INFINITY, itemSimilarity, dataModel, 1.0);
  }
  
  public NearestNItemNeighborhood(int n,
                                  double minSimilarity,
                                  ItemSimilarity itemSimilarity,
                                  DataModel dataModel) throws TasteException {
    this(n, minSimilarity, itemSimilarity, dataModel, 1.0);
  }
  
  public NearestNItemNeighborhood(int n,
                                  double minSimilarity,
                                  ItemSimilarity itemSimilarity,
                                  DataModel dataModel,
                                  double samplingRate) throws TasteException {
    super(itemSimilarity, dataModel, samplingRate);
    Preconditions.checkArgument(n >= 1, "n must be at least 1");
    int numUsers = dataModel.getNumUsers();
    this.n = n > numUsers ? numUsers : n;
    this.minSimilarity = minSimilarity;
  }
  
  @Override
  public long[] getItemNeighborhood(long userID) throws TasteException {
    
    DataModel dataModel = getDataModel();
    ItemSimilarity itemSimilarityImpl = getItemSimilarity();
    
    TopItems.Estimator<Long> estimator = new Estimator(itemSimilarityImpl, userID, minSimilarity);
    
    LongPrimitiveIterator itemIDs = SamplingLongPrimitiveIterator.maybeWrapIterator(dataModel.getItemIDs(),
      getSamplingRate());
    
    return TopItems.getTopItemsIDs(n, itemIDs, null, estimator);
  }
  
  @Override
  public String toString() {
    return "NearestNUserNeighborhood";
  }
  
  private static final class Estimator implements TopItems.Estimator<Long> {
    private final ItemSimilarity itemSimilarityImpl;
    private final long theUserID;
    private final double minSim;
    
    private Estimator(ItemSimilarity itemSimilarityImpl, long theUserID, double minSim) {
      this.itemSimilarityImpl = itemSimilarityImpl;
      this.theUserID = theUserID;
      this.minSim = minSim;
    }
    
    @Override
    public double estimate(Long userID) throws TasteException {
      if (userID == theUserID) {
        return Double.NaN;
      }
      double sim = itemSimilarityImpl.itemSimilarity(theUserID, userID);
      return sim >= minSim ? sim : Double.NaN;
    }
  }
}
