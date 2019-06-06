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

package org.apache.mahout.cf.taste.impl.recommender;

import org.apache.mahout.cf.taste.impl.common.Bicluster;
import org.apache.mahout.common.RandomUtils;

/** Simply encapsulates a user and a similarity value. */
public final class SimilarBicluster implements Comparable<SimilarBicluster> {
  
  private final int biclusterID;
  private final Bicluster<Long> bicluster;
  private final double similarity;
  
  public SimilarBicluster(int biclusterID, Bicluster<Long> bicluster, double similarity) {
    this.biclusterID = biclusterID;
    this.similarity = similarity;
    this.bicluster = bicluster;
  }
  
  int getBiclusterID() {
    return biclusterID;
  }
  
  double getSimilarity() {
    return similarity;
  }
  
  Bicluster<Long> getBicluster() {
	  return bicluster;
  }
  
  @Override
  public int hashCode() {
    return (int) biclusterID ^ RandomUtils.hashDouble(similarity);
  }
  
  @Override
  public boolean equals(Object o) {
    if (!(o instanceof SimilarBicluster)) {
      return false;
    }
    SimilarBicluster other = (SimilarBicluster) o;
    return biclusterID == other.getBiclusterID() && similarity == other.getSimilarity();
  }
  
  @Override
  public String toString() {
    return "SimilarUser[Bicluster:" + biclusterID + ", similarity:" + similarity + ']';
  }
  
  /** Defines an ordering from most similar to least similar. */
  @Override
  public int compareTo(SimilarBicluster other) {
    double otherSimilarity = other.getSimilarity();
    if (similarity > otherSimilarity) {
      return -1;
    }
    if (similarity < otherSimilarity) {
      return 1;
    }
    long otherBiclusterID = other.getBiclusterID();
    if (biclusterID < otherBiclusterID) {
      return -1;
    }
    if (biclusterID > otherBiclusterID) {
      return 1;
    }
    return 0;
  }
  
}
