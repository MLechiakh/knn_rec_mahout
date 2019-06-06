package org.apache.mahout.cf.taste.impl.common;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.common.RandomUtils;

import com.google.common.base.Preconditions;

public class XMotif extends AbstractBiclusteringAlgorithm {

	private final int ns;
	private final int nd;
	private final int sd;
	private final List<Interval> intervals;
	private final Random rand;

	public XMotif(DataModel model, int ns, int nd, int sd, List<Interval> intervals) {
		super(model);
		Preconditions.checkArgument(ns > 0, "ns must be > 0: " + ns);
		Preconditions.checkArgument(nd > 0, "nd must be > 0: " + nd);
		Preconditions.checkArgument(sd > 0, "sd must be > 0: " + sd);
		Preconditions.checkArgument(intervals != null, "intervals is null");
		this.ns = ns;
		this.nd = nd;
		this.sd = sd;
		this.intervals = intervals;
		this.rand = RandomUtils.getRandom();
	}

	@Override
	public void run() throws TasteException {

		// Prepare
		int n = this.dataModel.getNumUsers();

		List<Long> users = new ArrayList<Long>(n);
		LongPrimitiveIterator it = this.dataModel.getUserIDs();
		while (it.hasNext()) {
			long userID = it.nextLong();
			users.add(userID);
		}

		// Iterate
		for (int i = 0; i < this.ns; i++) {
			long userC = users.get(this.rand.nextInt(n));
			for (int j = 0; j < this.nd; j++) {

				Bicluster<Long> b = new Bicluster<Long>();

				// Get random subset
				Collections.shuffle(users, this.rand);

				// Determine set of items
				FastByIDMap<Interval> states = new FastByIDMap<Interval>();
				it = this.dataModel.getItemIDs();
				while (it.hasNext()) {
					long itemID = it.nextLong();

					// Find interval
					Float rating = this.dataModel.getPreferenceValue(userC, itemID);
					if (rating == null) {
						continue;
					}
					List<Interval> indexes = new ArrayList<Interval>();
					for (Interval s : this.intervals) {
						if (s.hasIn(rating)) {
							indexes.add(s);
						}
					}
					if (indexes.isEmpty()) {
						continue;
					}
					Interval s = indexes.get(this.rand.nextInt(indexes.size()));
					boolean valid = true;
					int k = 0;
					while (valid && k < this.sd) {
						long userID = users.get(k);
						rating = this.dataModel.getPreferenceValue(userID, itemID);
						if (rating == null || !s.hasIn(rating)) {
							valid = false;
						}
						k++;
					}
					if (valid) {
						b.addItem(itemID);
						states.put(itemID, s);
					}
				}

				// No item found
				if (b.getNbItems() == 0) {
					continue;
				}

				// Determine set of users
				it = this.dataModel.getUserIDs();
				while (it.hasNext()) {
					long userID = it.nextLong();
					boolean valid = true;
					Iterator<Long> items = b.getItems();
					while (valid && items.hasNext()) {
						long itemID = items.next();
						Float rating = this.dataModel.getPreferenceValue(userID, itemID);
						if (rating == null || !states.get(itemID).hasIn(rating)) {
							valid = false;
						}
					}
					if (valid) {
						b.addUser(userID);
					}
				}

				this.bicl.add(b);

			}
		}
	}

}
