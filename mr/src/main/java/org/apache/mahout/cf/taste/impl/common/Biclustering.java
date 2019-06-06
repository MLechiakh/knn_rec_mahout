package org.apache.mahout.cf.taste.impl.common;

import java.util.ArrayList;
import java.util.Iterator;

public class Biclustering<E> {
	
	private ArrayList<Bicluster<E>> biclusters; 
	
	public Biclustering() {
		this.biclusters = new ArrayList<Bicluster<E>>();
	}
	
	public void add(Bicluster<E> b) {
		this.biclusters.add(b);
	}
	
	public void remove(Bicluster<E> b) {
		this.biclusters.remove(b);
	}
	
	public int size() {
		return this.biclusters.size();
	}
	
	public Iterator<Bicluster<E>> iterator() {
		return this.biclusters.iterator();
	}
	
	public String toString() {
		return this.biclusters.toString();
	}

}
