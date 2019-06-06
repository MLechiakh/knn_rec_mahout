package org.apache.mahout.cf.taste.impl.common;

public class Average {

	private float sum;
	private int cnt;
	private float val;
	private boolean valValid;

	public Average() {
		this.sum = 0;
		this.cnt = 0;
		this.val = 0;
		this.valValid = false;
	}

	public Average(float value) {
		this.sum = value;
		this.cnt = 1;
		this.val = value;
		this.valValid = true;
	}

	public void add(float value) {
		this.sum += value;
		this.cnt++;
		this.valValid = false;
	}
	
	public int getCount() {
		return this.cnt;
	}

	public float compute() {
		if (!this.valValid) {
			if (this.cnt == 0) {
				return Float.NaN;
			}
			this.val = this.sum / (float) this.cnt;
			this.valValid = true;
		}
		return this.val;
	}

}
