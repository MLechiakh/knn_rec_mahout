package org.apache.mahout.cf.taste.impl.common;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

public class QUBIC extends AbstractBiclusteringAlgorithm {
	
	private static Logger log = LoggerFactory.getLogger(QUBIC.class);

	private final float consistency;
	private final float overlap;
	private final int bcnt;
	private final boolean bin;
	private final float th;

	public QUBIC(DataModel model, float c, int o, float f, Float th) throws TasteException {
		super(model);
		Preconditions.checkArgument(c > 0 && c <= 1,
				"Consistency level must be between 0 (exclusive) and 1 (inclusive): " + c);
		Preconditions.checkArgument(o > 0,
				"Number of biclusters output must be greater than 0: " + o);
		Preconditions.checkArgument(f >= 0 && f <= 1,
				"Overlap degree must be between 0 (inclusive) and 1 (inclusive): " + f);
		this.bicl = new TopSizeBiclustering<Long>(o);
		this.bcnt = o;
		this.consistency = c;
		this.overlap = f;
		if (th == null) {
			this.th = 0;
			this.bin = false;
		} else {
			this.th = th;
			this.bin = true;
		}
	}

	@Override
	public void run() throws TasteException {

		/* Construct the graph */
		log.debug("Building graph");
		int n = this.dataModel.getNumUsers();
		Queue<Seed> seeds = new PriorityQueue<Seed>(n * (n - 1));
		FastByIDMap<List<Bicluster<Long>>> inBcl = new FastByIDMap<List<Bicluster<Long>>>(n);
		LongPrimitiveIterator it1, it2;
		it1 = this.dataModel.getUserIDs();
		while (it1.hasNext()) {
			long userID1 = it1.nextLong();
			inBcl.put(userID1, new ArrayList<Bicluster<Long>>());
			it2 = this.dataModel.getUserIDs();
			while (it2.hasNext()) {
				long userID2 = it2.nextLong();
				if (userID1 < userID2) {
					double value = 0;
					for (Preference pref : this.dataModel.getPreferencesFromUser(userID1)) {
						Float rating = getValueFromRating(this.dataModel.getPreferenceValue(userID2, pref.getItemID()));
						if (rating != null && rating != 0 && rating == pref.getValue()) {
							value += 1;
						}
					}
					Seed s = new Seed(userID1, userID2, value);
					seeds.add(s);
				}
			}
		}

		/* Consider all possible seeds */
		while (seeds.size() != 0) {
			
			if (this.bicl.size() >= this.bcnt) {
				break;
			}
			
			Seed s = seeds.remove();
			long userID1 = s.getUserID1();
			long userID2 = s.getUserID2();
			double value = s.getValue();

			/* Check if seed is valid */
			log.debug("Checking seed ({} remaining then)", seeds.size());
			boolean isSeed = false;
			if (!inBcl.get(userID1).isEmpty() && !inBcl.get(userID2).isEmpty()) {
				for (Bicluster<Long> b1 : inBcl.get(userID1)) {
					if (isSeed) {
						break;
					}
					for (Bicluster<Long> b2 : inBcl.get(userID2)) {
						if (isSeed) {
							break;
						}
						if (value >= b1.getNbUsers() && value >= b2.getNbUsers()) {
							boolean disjoint = true;
							Iterator<Long> it = b1.getUsers();
							while (disjoint && it.hasNext()) {
								long userID = it.next();
								if (b2.containsUser(userID)) {
									disjoint = false;
								}
							}
							if (disjoint) {
								isSeed = true;
							}
						}
					}
				}
			} else {
				isSeed = true;
			}

			/* If seed is valid, then build bicluster */
			if (isSeed) {
				
				log.debug("Seed {} -- {} valid!", userID1, userID2);

				/* Step 1: Initial bicluster */
				log.debug("Building initial bicluster");
				Bicluster<Long> b = new Bicluster<Long>();
				FastByIDMap<Float> dominatingElements = new FastByIDMap<Float>();
				b.addUser(userID1);
				b.addUser(userID2);
				
				for (Preference pref : this.dataModel.getPreferencesFromUser(userID1)) {
					Float rating = getValueFromRating(this.dataModel.getPreferenceValue(userID2, pref.getItemID()));
					if (rating != null && rating != 0 && rating == pref.getValue()) {
						b.addItem(pref.getItemID());
						dominatingElements.put(pref.getItemID(), rating);
					}
				}
				if (b.isEmpty()) {
					/* Nothing in common, let's go to next seed */
					continue;
				}
				
				/* Step 2: Expand but maintain global consistency */
				log.debug("Expanding while maintaining global consistency");
				LongPrimitiveIterator itU = this.dataModel.getUserIDs();
				while (itU.hasNext()) {
					long userID = itU.nextLong();
					if (userID != userID1 && userID != userID2) {
						log.debug("Trying to expend {} by adding user {}", b, userID);
						Bicluster<Long> bb = b.copy();
						bb.addUser(userID);
						Iterator<Long> itI= b.getItems();
						while (itI.hasNext()) {
							long itemID = itI.next();
							Float r1 = getValueFromRating(this.dataModel.getPreferenceValue(userID1, itemID));
							Float r = getValueFromRating(this.dataModel.getPreferenceValue(userID, itemID));
							if (r1 == null || r == null || !r1.equals(r)) {
								bb.removeItem(itemID);
								log.debug("Removing item {} ({} != {})", itemID, r, r1);
							}
						}
						log.debug("After item removal, bicluster is {}", bb);
						if (Math.min(bb.getNbUsers(), bb.getNbItems()) >= Math.min(b.getNbUsers(), b.getNbItems())) {
							log.debug("Expanded by adding user {}", userID);
							b = bb;
						}
					}
				}
				
				/* Step 3: Expansion with less than total consistency */
				log.debug("Expanding as much as possible");
				if (this.consistency < 1) {
					LongPrimitiveIterator itI = this.dataModel.getItemIDs();
					while (itI.hasNext()) {
						long itemID = itI.nextLong();
						if (!b.containsItem(itemID)) {
							int max = 0;
							Float maxRating = (float) 0;
							Map<Float, Counter> cnts = new HashMap<Float, Counter>(b.getNbUsers());
							Iterator<Long> it = b.getUsers();
							while (it.hasNext()) {
								long userID = it.next();
								Float rating = getValueFromRating(this.dataModel.getPreferenceValue(userID, itemID));
								if (rating != null) {
									if (!cnts.containsKey(rating)) {
										cnts.put(rating, new Counter());
									}
									Counter c = cnts.get(rating);
									c.incr();
									if (c.get() > max) {
										max = c.get();
										maxRating = rating;
									}
								}
							}
							double cons = (double) max / (double) b.getNbUsers();
							if (cons >= this.consistency) {
								b.addItem(itemID);
								dominatingElements.put(itemID, maxRating);
							}
							itU = this.dataModel.getUserIDs();
							while (itU.hasNext()) {
								long userID = itU.nextLong();
								if (!b.containsUser(userID)) {
									int common = 0;
									it = b.getItems();
									while (it.hasNext()) {
										long otherItemID = it.next();
										Float rating = getValueFromRating(this.dataModel.getPreferenceValue(userID, otherItemID));
										if (rating != null && rating != 0 && rating.equals(dominatingElements.get(otherItemID))) {
											common++;
										}
									}
									float x = (float) common / (float) b.getNbItems();
									if (x >= this.consistency) {
										b.addUser(userID);
									}
								}
							}
						}
					}
					
				}
				/* Overlap post processing */
				boolean doesOverlap = false;
				Iterator<Bicluster<Long>> itb = this.bicl.iterator();
				while (!doesOverlap && itb.hasNext()) {
					Bicluster<Long> bb = itb.next();
					if (b.overlap(bb) > this.overlap) {
						doesOverlap = true;
					}
				}
				if (!doesOverlap) {
					this.bicl.add(b);
					Iterator<Long> it = b.getUsers();
					while (it.hasNext()) {
						long userID = it.next();
						if (!inBcl.containsKey(userID)) {
							inBcl.put(userID, new ArrayList<Bicluster<Long>>());
						}
						inBcl.get(userID).add(b);
					}
				}
				
			}

		}

	}
	
	private Float getValueFromRating(Float rating) {
		if (rating == null || !this.bin) {
			return rating;
		} else {
			return (float) (rating >= this.th ? 1 : 0);
		}
	}
	
	class Counter {
		
		private int value = 0;
		
		Counter() {
			this.value = 0;
		}
		
		void incr() {
			this.value++;
		}
		
		int get() {
			return this.value;
		}
		
	}

	class Seed implements Comparable<Seed> {

		private long userID1;
		private long userID2;
		private double value;

		long getUserID1() {
			return this.userID1;
		}

		long getUserID2() {
			return this.userID2;
		}

		double getValue() {
			return this.value;
		}

		Seed(long userID1, long userID2, double value) {
			this.userID1 = userID1;
			this.userID2 = userID2;
			this.value = value;
		}

		@Override
		public int compareTo(Seed s) {
			double y = s.getValue();
			if (this.value == y) {
				return 0;
			} else if (this.value > y) {
				return -1;
			} else {
				return 1;
			}
		}

	}

}
