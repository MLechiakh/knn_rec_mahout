package org.apache.mahout.cf.taste.impl.common;

import java.util.Iterator;
import java.util.PriorityQueue;
import java.util.Queue;

public class TopSizeBiclustering<E> extends Biclustering<E> {
	
	private Queue<Bicluster<E>> qbicl;
	private int size;
	
	TopSizeBiclustering(int size) {
		this.qbicl = new PriorityQueue<Bicluster<E>>(size);
		this.size = size;
	}
	
	@Override
	public void add(Bicluster<E> b) {
		if (this.size() == this.size) {
			this.qbicl.remove();
		}
		this.qbicl.add(b);
	}
	
	@Override
	public int size() {
		return this.qbicl.size();
	}
	
	@Override
	public Iterator<Bicluster<E>> iterator() {
		return this.qbicl.iterator();
	}
	
	@Override
	public String toString() {
		return this.qbicl.toString();
	}

}
