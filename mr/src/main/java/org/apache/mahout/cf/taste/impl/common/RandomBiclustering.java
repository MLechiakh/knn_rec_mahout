package org.apache.mahout.cf.taste.impl.common;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.common.RandomUtils;

import com.google.common.base.Preconditions;

public class RandomBiclustering extends AbstractBiclusteringAlgorithm {

	private final int cnt;
	private final Random rand;

	public RandomBiclustering(DataModel model, int cnt) {
		super(model);
		Preconditions.checkArgument(cnt > 0, "cnt must be > 0: " + cnt);
		this.cnt = cnt;
		this.rand = RandomUtils.getRandom();
	}

	@Override
	public void run() throws TasteException {

		LongPrimitiveIterator it;

		int n = this.dataModel.getNumUsers();
		int m = this.dataModel.getNumItems();

		List<Long> users = new ArrayList<Long>(n);
		it = this.dataModel.getUserIDs();
		while (it.hasNext()) {
			long userID = it.nextLong();
			users.add(userID);
		}

		List<Long> items = new ArrayList<Long>(m);
		it = this.dataModel.getItemIDs();
		while (it.hasNext()) {
			long itemID = it.nextLong();
			items.add(itemID);
		}

		for (int k = 0; k < this.cnt; k++) {

			int nb = this.rand.nextInt(n);
			int mb = this.rand.nextInt(m);
			Bicluster<Long> b = new Bicluster<Long>();
			Collections.shuffle(users, this.rand);
			Collections.shuffle(items, this.rand);
			for (int i = 0; i < nb; i++) {
				b.addUser(users.get(i));
			}
			for (int j = 0; j < mb; j++) {
				b.addItem(items.get(j));
			}
			this.bicl.add(b);

		}

	}

}
