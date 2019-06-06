package org.apache.mahout.cf.taste.impl.common;

import org.apache.mahout.cf.taste.impl.common.HashFunction;
import org.apache.mahout.cf.taste.impl.common.HashFunctionBuilder;

import java.lang.Exception;
import java.lang.Math;

import gnu.trove.list.array.TLongArrayList;

import com.google.common.base.Preconditions;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public abstract class AbstractCountMinSketch {
  
  private static final Logger log = LoggerFactory.getLogger(AbstractCountMinSketch.class);
  
  public class CMException extends Exception {
    
    public CMException(String message) {
      super("CountMinSketch: " + message);
    }
    
  }
  
  protected int w;
  protected int d;
  protected HashFunction[] hashFunctions;
  protected TLongArrayList insertedKeys;
  
  private void init(int width, int depth, HashFunctionBuilder hfBuilder) throws CMException {
    w = width;
    d = depth;
    
    log.debug("Creating count-min sketch with width {} and depth {}", w, d);
    
    hashFunctions = new HashFunction[d];
    for (int i = 0; i < d; i++) { hashFunctions[i] = hfBuilder.getHashFunction(i, w); }
    
    insertedKeys = new TLongArrayList();
  }
  
  
  /** Setup a new count-min sketch with parameters w and d
   * The parameters w and d control the accuracy of the estimates of the sketch
   * 
   * @param w           Width
   * @param d           Depth
   * @param hfBuilder   Hash functions builder
   * 
   * @throws  CountMinSketch.CMException  If delta or epsilon are not in the unit interval
   */
  public AbstractCountMinSketch(int width, int depth, HashFunctionBuilder hfBuilder) throws CMException {
    init(width, depth, hfBuilder);
  }
  
  /** Setup a new count-min sketch with parameters delta and epsilon
   * The parameters delta and epsilon control the accuracy of the estimates of the sketch
   * 
   * @param delta       A value in the unit interval that sets the precision of the sketch
   * @param epsilon     A value in the unit interval that sets the precision of the sketch
   * @param hfBuilder   Hash functions builder
   * 
   * @throws  CountMinSketch.CMException  If delta or epsilon are not in the unit interval
   */
  public AbstractCountMinSketch(double delta, double epsilon, HashFunctionBuilder hfBuilder) throws CMException {
    
    if (delta <= 0 || delta > Math.exp(-1)) {
      throw new CMException("delta must be between 0 and 1, exclusive");
    }
    if (epsilon <= 0 || epsilon > Math.exp(1)) {
      throw new CMException("epsilon must be between 0 and 1, exclusive");
    }
    
    int width = (int) (Math.ceil( Math.exp(1) / epsilon ));
    int depth = (int) (Math.ceil( Math.log(1 / delta) ));

    init(width, depth, hfBuilder);
    
  }
  
  
  /** Returns the list of keys inserted in the count-min sketch
   *  Useful to compute the error and configure the size parameters
   * 
   * @return  list of keys inserted
   */
  TLongArrayList getKeys() {
    return insertedKeys;
  }
  
  
  /** Returns a nice string representation of the count-min sketch content
   * 
   * @return matrix-like string representation of the sketch
   * 
   */
  abstract public String toString();
  
  
  
  
  
}
