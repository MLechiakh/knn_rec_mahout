package org.apache.mahout.cf.taste.impl.common;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;
import java.util.Set;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.common.RandomUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BicaiNet extends AbstractBiclusteringAlgorithm {

	private static Logger log = LoggerFactory.getLogger(BicaiNet.class);

	private final float wr;
	private final float wc;
	private final float lambda;
	private final int maxNbIt;
	private final int supIt;
	private final int o;
	private final Random rand;
	private final int nbRatings;

	public BicaiNet(DataModel model, float wr, float wc, float lambda, int maxNbIt, int supIt, int o)
			throws TasteException {
		super(model);
		this.wr = wr;
		this.wc = wc;
		this.lambda = lambda;
		this.maxNbIt = maxNbIt;
		this.supIt = supIt;
		this.o = o;
		this.rand = RandomUtils.getRandom();

		/* Compute number of ratings */
		int n = 0;
		LongPrimitiveIterator itU = this.dataModel.getUserIDs();
		while (itU.hasNext()) {
			long userID = itU.nextLong();
			PreferenceArray prefs = this.dataModel.getPreferencesFromUser(userID);
			if (prefs != null) {
				n += prefs.length();
			}
		}
		this.nbRatings = n;
	}

	private void init(int exp) throws TasteException {

		/* Compute probability */
		double proba = (double) exp / (double) nbRatings;

		/* Randomly select the seeds */
		LongPrimitiveIterator itU = this.dataModel.getUserIDs();
		while (itU.hasNext()) {
			long userID = itU.nextLong();
			for (Preference pref : this.dataModel.getPreferencesFromUser(userID)) {
				double x = this.rand.nextDouble();
				if (x <= proba) {
					Bicluster<Long> b = new Bicluster<Long>();
					b.addUser(userID);
					b.addItem(pref.getItemID());
					this.bicl.add(b);
					log.debug("Added {}", b);
				}
			}
		}
	}

	public static double residue(Bicluster<Long> b, DataModel dataModel)
			throws TasteException {
		int N = b.getNbUsers();
		int M = b.getNbItems();
		FastByIDMap<Average> AR = new FastByIDMap<Average>(N);
		FastByIDMap<Average> AC = new FastByIDMap<Average>(M);
		Average ACC = new Average();
		Iterator<Long> itU = b.getUsers();
		while (itU.hasNext()) {
			long userID = itU.next();
			Iterator<Long> itI = b.getItems();
			while (itI.hasNext()) {
				long itemID = itI.next();
				Float rating = dataModel.getPreferenceValue(userID, itemID);
				float value = rating != null ? rating : 0;
				if (!AR.containsKey(userID)) {
					AR.put(userID, new Average());
				}
				AR.get(userID).add(value);
				if (!AC.containsKey(itemID)) {
					AC.put(itemID, new Average());
				}
				AC.get(itemID).add(value);
				ACC.add(value);
			}
		}
		double s = 0;
		itU = b.getUsers();
		while (itU.hasNext()) {
			long userID = itU.next();
			Iterator<Long> itI = b.getItems();
			while (itI.hasNext()) {
				long itemID = itI.next();
				Float rating = dataModel.getPreferenceValue(userID, itemID);
				if (rating != null) {
					double x = rating - AR.get(userID).compute() - AC.get(itemID).compute() + ACC.compute();
					s += x * x;
				}
			}
		}
		double R = s / ((double) N * M);
		return R;
	}
	
	public static double fitness(Bicluster<Long> b, DataModel dataModel, float wr, float wc, float lambda)
			throws TasteException {
		int N = b.getNbUsers();
		int M = b.getNbItems();
		double R = residue(b, dataModel);
		return R / lambda + (wc * lambda) / (double) M + (wr * lambda) / (double) N;
	}

	private void mutate(Bicluster<Long> b) throws TasteException {
		log.debug("Mutate {}", b);
		double x;
		x = this.rand.nextDouble();
		if (x < 0.5) { // Insert
			x = this.rand.nextDouble();
			if (x < 0.5) { // Row
				log.debug("Insert row");
				if (b.getNbUsers() == this.dataModel.getNumUsers()) {
					log.debug("Cannot insert row, already all of them in the bicluster");
					return;
				}
				LongPrimitiveIterator it = this.dataModel.getUserIDs();
				int idx = this.rand.nextInt(this.dataModel.getNumUsers());
				long userID = 0;
				for (int i = 0; i <= idx && it.hasNext(); i++) {
					userID = it.nextLong();
				}
				while (b.containsUser(userID)) {
					if (!it.hasNext()) {
						it = this.dataModel.getUserIDs();
					}
					userID = it.nextLong();
				}
				b.addUser(userID);
			} else { // Column
				log.debug("Insert column");
				if (b.getNbItems() == this.dataModel.getNumItems()) {
					log.debug("Cannot insert column, already all of them in the bicluster");
					return;
				}
				LongPrimitiveIterator it = this.dataModel.getItemIDs();
				int idx = this.rand.nextInt(this.dataModel.getNumItems());
				long itemID = 0;
				for (int i = 0; i <= idx && it.hasNext(); i++) {
					itemID = it.nextLong();
				}
				while (b.containsItem(itemID)) {
					if (!it.hasNext()) {
						it = this.dataModel.getItemIDs();
					}
					itemID = it.nextLong();
				}
				b.addItem(itemID);
			}
		} else { // Remove
			if (x < 0.5) { // Row
				log.debug("Remove row");
				Iterator<Long> it = b.getUsers();
				int idx = this.rand.nextInt(b.getNbUsers());
				long userID = 0;
				for (int i = 0; i <= idx && it.hasNext(); i++) {
					userID = it.next();
				}
				b.removeUser(userID);
			} else { // Column
				log.debug("Remove column");
				Iterator<Long> it = b.getItems();
				int idx = this.rand.nextInt(b.getNbItems());
				long itemID = 0;
				for (int i = 0; i <= idx && it.hasNext(); i++) {
					itemID = it.next();
				}
				b.removeItem(itemID);
			}
		}
	}

	private int suppress(double eps) throws TasteException {
		Set<Bicluster<Long>> toRemove = new HashSet<Bicluster<Long>>();
		Iterator<Bicluster<Long>> it1 = this.bicl.iterator();
		while (it1.hasNext()) {
			Bicluster<Long> b1 = it1.next();
			Iterator<Bicluster<Long>> it2 = this.bicl.iterator();
			while (it2.hasNext()) {
				Bicluster<Long> b2 = it2.next();
				if (b1 != b2) {
					if (b1.nbCommonCells(b2) > eps) {
						if (fitness(b1, this.dataModel, this.wr, this.wc, this.lambda) < fitness(b2, this.dataModel,
								this.wr, this.wc, this.lambda)) {
							toRemove.add(b2);
						} else {
							toRemove.add(b1);
						}
					}
				}
			}
		}
		for (Bicluster<Long> b : toRemove) {
			log.debug("Removed {}", b);
			this.bicl.remove(b);
		}
		return toRemove.size();
	}

	@Override
	public void run() throws TasteException {

		/* Random initialization */
		init(this.o);
		log.debug("Done with initialization, got {} seeds", this.bicl.size());

		/* Iterations */
		for (int k = 0; k < this.maxNbIt; k++) {
			log.debug("Starting iteration #{}", k);
			Set<Bicluster<Long>> toRemove = new HashSet<Bicluster<Long>>();
			Set<Bicluster<Long>> toAdd = new HashSet<Bicluster<Long>>();
			Iterator<Bicluster<Long>> it = this.bicl.iterator();
			while (it.hasNext()) {
				Bicluster<Long> b = it.next();
				Bicluster<Long> bb = b.copy();
				mutate(bb);
				if (bb.isEmpty()) {
					log.debug("Ignore mutation");
					continue;
				}
				double x1 = fitness(b, this.dataModel, this.wr, this.wc, this.lambda);
				double x2 = fitness(bb, this.dataModel, this.wr, this.wc, this.lambda);
				if (x2 <= x1) {
					log.debug("Replace bicluster {} by {} (fitness {} to {})", b, bb, x1, x2);
					toRemove.add(b);
					toAdd.add(bb);
				} else {
					log.debug("Ignore mutation");
				}
			}
			for (Bicluster<Long> b : toRemove) {
				this.bicl.remove(b);
			}
			for (Bicluster<Long> b : toAdd) {
				this.bicl.add(b);
			}
			if (k % this.supIt == 0) {
				log.debug("Let us remove and add new seeds", k);
				int nbRemoved = suppress(10);
				init(nbRemoved);
			}
		}

	}

}
