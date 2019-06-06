package org.apache.mahout.cf.taste.eval;

import java.util.Iterator;

import org.apache.mahout.cf.taste.impl.eval.Fold;

public interface FoldDataSplitter {
	
	Iterator<Fold> getFolds();

}
