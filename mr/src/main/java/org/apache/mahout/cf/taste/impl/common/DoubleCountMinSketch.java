package org.apache.mahout.cf.taste.impl.common;

import org.apache.mahout.cf.taste.impl.common.HashFunction;
import org.apache.mahout.cf.taste.impl.common.AbstractCountMinSketch;

import java.lang.Exception;
import java.lang.Math;

import gnu.trove.list.array.TDoubleArrayList;

import com.google.common.base.Preconditions;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class DoubleCountMinSketch extends AbstractCountMinSketch {
  
  private static final Logger log = LoggerFactory.getLogger(DoubleCountMinSketch.class);
  
  private final TDoubleArrayList count;
  
  /** Setup a new count-min sketch with parameters w and d
   * The parameters w and d control the accuracy of the estimates of the sketch
   * 
   * @param w           Width
   * @param d           Depth
   * @param hfBuilder   Hash functions builder
   * 
   * @throws  CountMinSketch.CMException  If delta or epsilon are not in the unit interval
   */
  public DoubleCountMinSketch(int width, int depth, HashFunctionBuilder hfBuilder) throws CMException {
    super(width, depth, hfBuilder);
    count = new TDoubleArrayList(w * d);
    for (int i = 0; i < w * d; i++) { count.insert(i, 0); }
  }

  /** Setup a new count-min sketch with parameters delta, epsilon, and k
   * The parameters delta,epsilon and k control the accuracy of the estimates of the sketch
   * 
   * @param delta       A value in the unit interval that sets the precision of the sketch
   * @param epsilon     A value in the unit interval that sets the precision of the sketch
   * @param hfBuilder   Hash functions builder
   * 
   * @throws  CountMinSketch.CMException  If delta or epsilon are not in the unit interval
   */
  public DoubleCountMinSketch(double delta, double epsilon, HashFunctionBuilder hfBuilder) throws CMException {
    super(delta, epsilon, hfBuilder);
    count = new TDoubleArrayList(w * d);
    for (int i = 0; i < w * d; i++) { count.insert(i, 0); }
  }
  
  
  /** Returns the value in a given cell of the sketch
   * 
   * @param i   Row index
   * @param j   Column index
   * 
   * @return  Value in cell at i-th row and j-th column
   * 
   */
  private double get(int i, int j) {
    return count.get(j + i * w);
  }
  
  /** Updates the sketch for the item with name of key by the amount specified in increment
   * 
   * @param key         The item to update the value of in the sketch
   * @param increment   The amount to update the sketch by for the given key
   * 
   */
  public void update(long key, double increment) {
    insertedKeys.add(key);
    for (int i = 0; i < d; i++) {
      int j = hashFunctions[i].hash(key);
      double value = get(i, j);
      count.set(j + i * w, value + increment);
      log.debug("Update value row {} column {}: previous value {}, new value {}", i, j, value, value + increment);
    }
  }
  
  /** Fetches the sketch estimate for the given key
   * 
   * @param key   The item to produce an estimate for
   * 
   * @return    The best estimate of the count for the given key based on the sketch
   * 
   * For an item i with count a_i, the estimate from the sketch a_i_hat will satisfy the relation
   *        a_hat_i <= a_i + epsilon * ||a||_1
   * with probability at least 1 - delta, where a is the the vector of
   * all counts and ||x||_1 is the L1 norm of a vector x
   * 
   */
  public double get(long key) {
    double estimate = Double.MAX_VALUE;
    for (int i = 0; i < d; i++) {
      int j = hashFunctions[i].hash(key);
      double value = get(i, j);
      if (value < estimate) { estimate = value; }
    }
    log.debug("Estimate value for key {} is {}", key, estimate);
    return estimate;
  }
  
  /** Computes an approximate cosine between two count-min sketches
   *  The two count-min sketches must have the same size
   * 
   * @param a   Count-min sketch (epsilon, delta)
   * @param b   Count-min sketch (epsilon, delta)
   * 
   * @return  Cosine approximation
   *
   */
  public static double cosine(DoubleCountMinSketch a, DoubleCountMinSketch b) {
    
    /* Check both count-min sketches have same size */
    Preconditions.checkArgument(a.w == b.w, "Widths of a (%s) and b (%s) must be the same", a.w, b.w);
    Preconditions.checkArgument(a.d == b.d, "Depths of a (%s) and b (%s) must be the same", a.d, b.d);
    
    double minCosine = Double.MAX_VALUE;
    double currentCosine = 0;
    
    for (int i = 0; i < a.d; i++) {
      
      double valueA = 0.0;
      double valueB = 0.0;
      double valueAB = 0.0;
      
      for (int j = 0; j < a.w; j++) {
        double xa = a.get(i, j);
        double xb = b.get(i, j);
        valueA += xa * xa;
        valueB += xb * xb;
        valueAB += xa * xb;
      }
      
      double denominator = Math.sqrt(valueA) * Math.sqrt(valueB);
      if (denominator != 0) {
        currentCosine = valueAB / denominator;
        minCosine = Math.min(minCosine, currentCosine);
      }
      
    }
    
    log.debug("Cosine is {}", minCosine);
    
    if (minCosine == Double.MAX_VALUE) { return Double.NaN; }
    return minCosine;
  }
  
  
  /** Returns a nice string representation of the count-min sketch content
   * 
   * @return matrix-like string representation of the sketch
   * 
   */
  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append(System.lineSeparator());
    for (int i = 0; i < d; i++) {
      builder.append("| ");
      for (int j = 0; j < w; j++) {
        builder.append(get(i, j));
        builder.append(" | ");
      }
      builder.append(System.lineSeparator());
    }
    return builder.toString();
  }
  
}
