package org.apache.mahout.cf.taste.impl.common;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.model.DataModel;

public abstract class AbstractBiclusteringAlgorithm {

	protected DataModel dataModel;
	protected Biclustering<Long> bicl;

	AbstractBiclusteringAlgorithm(DataModel model) {
		this.dataModel = model;
		this.bicl = new Biclustering<Long>();
	}

	public abstract void run() throws TasteException;

	public Biclustering<Long> get() {
		return this.bicl;
	}

}
