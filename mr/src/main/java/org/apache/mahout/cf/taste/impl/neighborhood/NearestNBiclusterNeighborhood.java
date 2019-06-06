package org.apache.mahout.cf.taste.impl.neighborhood;

import java.util.Iterator;
import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.Bicluster;
import org.apache.mahout.cf.taste.impl.common.Biclustering;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.similarity.UserBiclusterSimilarity;

import com.google.common.base.Preconditions;

/**
 * <p>
 * Computes a neighborhood consisting of the nearest n biclusters to a given user. "Nearest" is defined by the
 * given {@link UserBiclusterSimilarity}.
 * </p>
 */
public final class NearestNBiclusterNeighborhood extends AbstractBiclusterNeighborhood {
  
  private final int n;
  private final double minSimilarity;
  
  /**
   * @param n neighborhood size; capped to the number of biclusters
   * @throws IllegalArgumentException
   *           if {@code n < 1}, or userBiclusterSimilarity or dataModel are {@code null}
   */
  public NearestNBiclusterNeighborhood(int n, UserBiclusterSimilarity userBiclusterSimilarity, 
		  							DataModel dataModel, Biclustering<Long> biclustering) throws TasteException {
    this(n, Double.NEGATIVE_INFINITY, userBiclusterSimilarity, dataModel, biclustering);
  }
  
  /**
   * @param n neighborhood size; capped to the number of biclusters
   * @param minSimilarity minimal similarity required for neighbors
   * @throws IllegalArgumentException
   *           if {@code n < 1}, or userBiclusterSimilarity or dataModel are {@code null}
   */
  public NearestNBiclusterNeighborhood(int n,
                                  double minSimilarity,
                                  UserBiclusterSimilarity userBiclusterSimilarity,
                                  DataModel dataModel, Biclustering<Long> biclustering) throws TasteException {
    super(userBiclusterSimilarity, dataModel, biclustering);
    Preconditions.checkArgument(n >= 1, "n must be at least 1");
    int N = getBiclustering().size();
    this.n = n < N ? n : N;
    this.minSimilarity = minSimilarity;
  }
  
  @Override
  public List<Bicluster<Long>> getUserNeighborhood(long userID) throws TasteException {
    
    UserBiclusterSimilarity userBiclusterSimilarity = getUserBiclusterSimilarity();
    
    TopItems.Estimator<Bicluster<Long>> estimator = new Estimator(userBiclusterSimilarity, userID, minSimilarity);
    Iterator<Bicluster<Long>> biclusters = getBiclustering().iterator();
    
    return TopItems.getTopBiclusters(n, biclusters, estimator);
    
  }
  
  @Override
  public String toString() {
    return "NearestNBiclusterNeighborhood";
  }
  
  private static final class Estimator implements TopItems.Estimator<Bicluster<Long>> {
    private final UserBiclusterSimilarity userBiclusterSimilarityImpl;
    private final long theUserID;
    private final double minSim;
    
    private Estimator(UserBiclusterSimilarity userBiclusterSimilarityImpl, long theUserID, double minSim) {
      this.userBiclusterSimilarityImpl = userBiclusterSimilarityImpl;
      this.theUserID = theUserID;
      this.minSim = minSim;
    }
    
    @Override
    public double estimate(Bicluster<Long> bicluster) throws TasteException {
      double sim = userBiclusterSimilarityImpl.userBiclusterSimilarity(this.theUserID, bicluster);
      return sim >= minSim ? sim : Double.NaN;
    }
  }
}
