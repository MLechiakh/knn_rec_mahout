package org.apache.mahout.cf.taste.impl.neighborhood;

import java.util.Collection;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.impl.common.Biclustering;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserBiclusterNeighborhood;
import org.apache.mahout.cf.taste.similarity.UserBiclusterSimilarity;

import com.google.common.base.Preconditions;

/**
 * <p>
 * Contains methods and resources useful to all classes in this package.
 * </p>
 */
abstract class AbstractBiclusterNeighborhood implements UserBiclusterNeighborhood {
  
  private final UserBiclusterSimilarity userBiclusterSimilarity;
  private final DataModel dataModel;
  private final Biclustering<Long> biclustering;
  private final RefreshHelper refreshHelper;
  
  AbstractBiclusterNeighborhood(UserBiclusterSimilarity sim, DataModel dataModel, Biclustering<Long> biclustering) {
    Preconditions.checkArgument(sim != null, "userBiclusterSimilarity is null");
    Preconditions.checkArgument(dataModel != null, "dataModel is null");
    this.userBiclusterSimilarity = sim;
    this.dataModel = dataModel;
    this.biclustering = biclustering;
    this.refreshHelper = new RefreshHelper(null);
    this.refreshHelper.addDependency(this.dataModel);
    this.refreshHelper.addDependency(this.userBiclusterSimilarity);
  }
  
  final UserBiclusterSimilarity getUserBiclusterSimilarity() {
    return userBiclusterSimilarity;
  }
  
  final DataModel getDataModel() {
    return dataModel;
  }
  
  final Biclustering<Long> getBiclustering() {
	  return this.biclustering;
  }
   
  @Override
  public final void refresh(Collection<Refreshable> alreadyRefreshed) {
    refreshHelper.refresh(alreadyRefreshed);
  }
  
}
