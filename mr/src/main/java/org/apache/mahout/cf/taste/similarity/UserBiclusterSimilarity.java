package org.apache.mahout.cf.taste.similarity;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.Bicluster;

/**
 * <p>
 * Implementations of this interface define a notion of similarity between one user and a bicluster. Implementations should
 * return values in the range -1.0 to 1.0, with 1.0 representing perfect similarity.
 * </p>
 * 
 */
public interface UserBiclusterSimilarity extends Refreshable {
  
  /**
   * <p>
   * Returns the degree of similarity, of one user and one bicluster.
   * </p>
   * 
   * @param userID user ID
   * @param bicluster bicluster
   * @return similarity, in [-1,1] or {@link Double#NaN} similarity is unknown
   * @throws org.apache.mahout.cf.taste.common.NoSuchUserException
   *  if either user is known to be non-existent in the data
   * @throws TasteException if an error occurs while accessing the data
   */
  double userBiclusterSimilarity(long userID, Bicluster<Long> bicluster) throws TasteException;

  /**
   * <p>
   * Attaches a {@link PreferenceInferrer} to the {@link UserBiclusterSimilarity} implementation.
   * </p>
   * 
   * @param inferrer {@link PreferenceInferrer}
   */
  void setPreferenceInferrer(PreferenceInferrer inferrer);
  
}
