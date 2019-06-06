package org.apache.mahout.cf.taste.neighborhood;

import java.util.List;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.Bicluster;

/**
 * <p>
 * Implementations of this interface compute a "neighborhood" of biclusters like a given user. This neighborhood
 * can be used to compute recommendations then.
 * </p>
 */
public interface UserBiclusterNeighborhood extends Refreshable {
  
  /**
   * @param userID
   *          ID of user for which a neighborhood will be computed
   * @return IDs of neighboring biclusters
   * @throws TasteException
   *           if an error occurs while accessing data
   */
  List<Bicluster<Long>> getUserNeighborhood(long userID) throws TasteException;
  
}
