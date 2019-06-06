package org.apache.mahout.cf.taste.impl.common;

import org.apache.mahout.cf.taste.impl.common.HashFunction;

import java.math.BigInteger;
import java.util.Random;

import gnu.trove.list.array.TLongArrayList;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class HashFunctionBuilder {
  
  private static final Logger log = LoggerFactory.getLogger(HashFunctionBuilder.class);
  
  private int index;
  private final BigInteger bigPrime;
  private final TLongArrayList randomParamA;
  private final TLongArrayList randomParamB;
  private final Random rand;
  
  public HashFunctionBuilder(long seed) {
    bigPrime = new BigInteger("9223372036854775783");
    randomParamA = new TLongArrayList();
    randomParamB = new TLongArrayList();
    rand = new Random(seed);
    index = -1;
  }
  
  public HashFunctionBuilder() {
    this(System.currentTimeMillis());
  }
  
  /** Return a hash function for a given iteration index and width
   * 
   *  @param  iteration   Used to choose the parameters of the hash function
   *  @param  width       Range: a key will be hashed into {0, .., width - 1}
   */
  HashFunction getHashFunction(int iteration, int size) {
    
    synchronized (this) {
      /* Check if enough random parameters already generated, otherwise generate some more */
      if (iteration > index) {
        for (int i = index + 1; i < (iteration + 1); i++) {
          long ra = Math.abs(rand.nextLong());
          long rb = Math.abs(rand.nextLong());
          randomParamA.add(ra);
          randomParamB.add(rb);
        }
        index = iteration;
      }
    }
    
    BigInteger a = BigInteger.valueOf(randomParamA.get(iteration));
    BigInteger b = BigInteger.valueOf(randomParamB.get(iteration));
    BigInteger w = BigInteger.valueOf(size);
    
    return new HashFunction(a, b, w, bigPrime);
    
  }
  
}
