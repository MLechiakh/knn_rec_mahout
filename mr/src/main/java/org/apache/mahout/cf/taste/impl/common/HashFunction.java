package org.apache.mahout.cf.taste.impl.common;

import java.math.BigInteger;
import java.util.Random;

import gnu.trove.list.array.TLongArrayList;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class HashFunction {
  
  private static final Logger log = LoggerFactory.getLogger(HashFunction.class);
  
  /* Store parameters for hash computations */
  private final BigInteger a;
  private final BigInteger b;
  private final BigInteger w;
  private final BigInteger bigPrime;
  
  
  HashFunction(BigInteger a_, BigInteger b_, BigInteger w_, BigInteger bigPrime_) {
    a = a_;
    b = b_;
    w = w_;
    bigPrime = bigPrime_;
  }
  
  
  /** Hashes a key and returns an integer */
  int hash(long key) {
    BigInteger k = BigInteger.valueOf(key);
    return a.multiply(k).add(b).mod(bigPrime).mod(w).intValue();
  }
  
}
