package org.apache.mahout.cf.taste.impl.common;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.recommender.COCLUSTRecommender;
import org.apache.mahout.cf.taste.model.DataModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class COCLUSTBiclustering extends AbstractBiclusteringAlgorithm {
	
	private static Logger log = LoggerFactory.getLogger(COCLUSTBiclustering.class);

	private final int k;
	private final int l;
	private final int nbMaxIterations;

	public COCLUSTBiclustering(DataModel model, int nbUserClusters, int nbItemClusters, int maxIter) throws TasteException {
		super(model);
		this.k = nbUserClusters;
		this.l = nbItemClusters;
		this.nbMaxIterations = maxIter;
	}

	@Override
	public void run() throws TasteException {

		COCLUSTRecommender rec = new COCLUSTRecommender(this.dataModel, this.k, this.l, this.nbMaxIterations);
		this.bicl = rec.getBiclustering();

	}

}
