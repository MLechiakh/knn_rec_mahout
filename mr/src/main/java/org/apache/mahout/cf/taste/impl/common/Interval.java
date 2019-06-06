package org.apache.mahout.cf.taste.impl.common;

public class Interval {
	
	private final double lower;
	private final boolean includeLower;
	private final double upper;
	private final boolean includeUpper;
	
	public Interval(double lower, double upper) {
		this(lower, true, upper, true);
	}
	
	public Interval(double lower, boolean includeLower, double upper, boolean includeUpper) {
		this.lower = lower;
		this.upper = upper;
		this.includeLower = includeLower;
		this.includeUpper = includeUpper;
	}
	
	double getLower() {
		return this.lower;
	}
	
	double getUpper() {
		return this.upper;
	}
	
	boolean hasIn(double x) {
		if (x < this.lower) {
			return false;
		} else if (x == this.lower) {
			if (this.includeLower) {
				return true;
			} else {
				return false;
			}
		} else if (x < this.upper) {
			return true;
		} else if (x == this.upper) {
			if (this.includeUpper) {
				return true;
			} else {
				return false;
			}
		} else {
			return false;
		}
	}

}
