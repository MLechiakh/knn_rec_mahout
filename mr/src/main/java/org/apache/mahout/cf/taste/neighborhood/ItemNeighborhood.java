package org.apache.mahout.cf.taste.neighborhood;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;

public interface ItemNeighborhood extends Refreshable {
  
  long[] getItemNeighborhood(long itemID) throws TasteException;
  
}
