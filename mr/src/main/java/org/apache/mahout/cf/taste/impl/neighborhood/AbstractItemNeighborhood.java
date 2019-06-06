package org.apache.mahout.cf.taste.impl.neighborhood;

import java.util.Collection;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.ItemNeighborhood;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import com.google.common.base.Preconditions;

abstract class AbstractItemNeighborhood implements ItemNeighborhood {
  
  private final ItemSimilarity itemSimilarity;
  private final DataModel dataModel;
  private final double samplingRate;
  private final RefreshHelper refreshHelper;
  
  AbstractItemNeighborhood(ItemSimilarity itemSimilarity, DataModel dataModel, double samplingRate) {
    Preconditions.checkArgument(itemSimilarity != null, "userSimilarity is null");
    Preconditions.checkArgument(dataModel != null, "dataModel is null");
    Preconditions.checkArgument(samplingRate > 0.0 && samplingRate <= 1.0, "samplingRate must be in (0,1]");
    this.itemSimilarity = itemSimilarity;
    this.dataModel = dataModel;
    this.samplingRate = samplingRate;
    this.refreshHelper = new RefreshHelper(null);
    this.refreshHelper.addDependency(this.dataModel);
    this.refreshHelper.addDependency(this.itemSimilarity);
  }
  
  final ItemSimilarity getItemSimilarity() {
    return itemSimilarity;
  }
  
  final DataModel getDataModel() {
    return dataModel;
  }
  
  final double getSamplingRate() {
    return samplingRate;
  }
  
  @Override
  public final void refresh(Collection<Refreshable> alreadyRefreshed) {
    refreshHelper.refresh(alreadyRefreshed);
  }
  
}
