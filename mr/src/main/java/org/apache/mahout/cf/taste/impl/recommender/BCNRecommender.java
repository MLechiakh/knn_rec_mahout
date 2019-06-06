package org.apache.mahout.cf.taste.impl.recommender;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import com.google.common.base.Preconditions;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.Bicluster;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.recommender.AbstractRecommender;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.impl.similarity.JaccardItemSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.thegalactic.context.Context;

public final class BCNRecommender extends AbstractRecommender {

	private final ItemSimilarity sim;
	private final float threshold;
	private final FastByIDMap<Bicluster<Long>> smallers;
	private final FastByIDMap<List<Bicluster<Long>>> neighborhoods;
	private final Context ctx;
	private final int level;

	private static final Logger log = LoggerFactory.getLogger(BCNRecommender.class);

	public BCNRecommender(DataModel dataModel, float threshold, int level, CandidateItemsStrategy strategy) throws TasteException {
		super(dataModel, strategy);
		Preconditions.checkArgument(level > 0, "level must be at least 1");
		this.sim = new JaccardItemSimilarity(dataModel, threshold);
		this.threshold = threshold;
		this.smallers = new FastByIDMap<Bicluster<Long>>(dataModel.getNumUsers());
		this.neighborhoods = new FastByIDMap<List<Bicluster<Long>>>(dataModel.getNumUsers());
		this.ctx = new Context();
		this.level = level;
		initLattice();
	}

	public BCNRecommender(DataModel dataModel, float threshold, int level) throws TasteException {
		super(dataModel);
		Preconditions.checkArgument(level > 0, "level must be at least 1");
		this.sim = new JaccardItemSimilarity(dataModel, threshold);
		this.threshold = threshold;
		this.smallers = new FastByIDMap<Bicluster<Long>>(dataModel.getNumUsers());
		this.neighborhoods = new FastByIDMap<List<Bicluster<Long>>>(dataModel.getNumUsers());
		this.ctx = new Context();
		this.level = level;
		initLattice();
	}

	private void initLattice() throws TasteException {
		log.debug("Add attributes to lattice");
		DataModel model = getDataModel();
		LongPrimitiveIterator it;
		it = model.getItemIDs();
		while (it.hasNext()) {
			long itemID = it.nextLong();
			this.ctx.addToAttributes(itemID);
		}
		log.debug("Add obsevations and intent/extend to lattice");
		it = model.getUserIDs();
		while (it.hasNext()) {
			long userID = it.nextLong();
			this.ctx.addToObservations(userID);
			for (Preference pref : model.getPreferencesFromUser(userID)) {
				if (pref.getValue() >= this.threshold) {
					long itemID = pref.getItemID();
					boolean b = this.ctx.addExtentIntent(userID, itemID);
					log.debug("Added {},{}, worked? {}", userID, itemID, b);
				}
			}
		}
		log.debug("Final context is {}", this.ctx.toString());
		
		this.ctx.setBitSets();
	}

	@Override
	public List<RecommendedItem> recommend(long userID, int howMany, IDRescorer rescorer, boolean includeKnownItems)
			throws TasteException {
		Preconditions.checkArgument(howMany >= 0, "howMany must be at least 0");
		log.debug("Recommending items for user ID '{}'", userID);

		if (howMany == 0) {
			return Collections.emptyList();
		}

		PreferenceArray preferencesFromUser = getDataModel().getPreferencesFromUser(userID);
		FastIDSet possibleItemIDs = getAllOtherItems(userID, preferencesFromUser, includeKnownItems);

		List<RecommendedItem> topItems = TopItems.getTopItems(howMany, possibleItemIDs.iterator(), rescorer,
				new Estimator(userID));
		log.debug("Recommendations are: {}", topItems);

		return topItems;
	}

	@Override
	public float estimatePreference(long userID, long itemID) throws TasteException {
		long t1, t2;
		
		DataModel model = getDataModel();
		Float actualPref = model.getPreferenceValue(userID, itemID);
		if (actualPref != null) {
			return actualPref;
		}

		Bicluster<Long> sb = getSmallestBicluster(userID);
		if (sb.isEmpty()) {
			return Float.NaN;
		}

		double g = 0;
		int cnt = 0;
		Iterator<Long> it = sb.getItems();
		while (it.hasNext()) {
			long j = it.next();
			if (j != itemID) {
				g += this.sim.itemSimilarity(itemID, j);
				cnt++;
			}
		}
		if (cnt > 0) {
			g = g / (double) cnt;
		}
		
		log.debug("Global similarity of user {} and item {} is {}", userID, itemID, g);

		t1 = System.currentTimeMillis();
		List<Bicluster<Long>> neighbors = getBiclusterNeighborhood(sb, userID);
		t2 = System.currentTimeMillis();
		log.debug("candidate biclusters: {} ms", t2 - t1);
		
		double l = 0;
		for (Bicluster<Long> bb : neighbors) {
			if (bb.containsItem(itemID)) {
				l += biSim(sb, bb);
			}
		}
		if (l == 0) {
			return Float.NaN;
		}

		log.debug("Local similarity of user {} and item {} is {}", userID, itemID, l);

		return (float) (g * l);
	}

	@SuppressWarnings("rawtypes")
	private Bicluster<Long> getSmallestBicluster(long userID) throws TasteException {
		if (!this.smallers.containsKey(userID)) {
			TreeSet<Comparable> itemSet = this.ctx.getIntent(userID);
			TreeSet<Comparable> userSet = this.ctx.getExtent(itemSet);
			Bicluster<Long> sb = getBiclusterFromComparable(userSet, itemSet);
			log.debug("Computed smallest bicluster for user {} is {}", userID, sb);
			this.smallers.put(userID, sb);
			return sb;
		} else {
			Bicluster<Long> sb = this.smallers.get(userID);
			log.debug("Cached smallest bicluster for user {} is {}", userID, sb);
			return sb;
		}
	}

	@SuppressWarnings("rawtypes")
	private List<Bicluster<Long>> getBiclusterNeighborhood(Bicluster<Long> sb, long userID) throws TasteException {
		if (!this.neighborhoods.containsKey(userID)) {
			
			TreeSet<Comparable> userSet = new TreeSet<Comparable>();
			for (long otherUserID : sb.getSetUsers()) {
				userSet.add(otherUserID);
			}
			
			List<Bicluster<Long>> candidates = getLowers(sb, null);
			log.debug("Lower biclusters of {} are {}", sb, candidates);
			List<Bicluster<Long>> uppers = getUppers(sb);
			log.debug("Upper biclusters of {} are {}", sb, uppers);
			for (Bicluster<Long> b : uppers) {
				candidates.addAll(getLowers(b, userSet));
			}
			log.debug("Computed candidate biclusters of {} are {}", sb, candidates);
			
			int r = this.level;
			List<Bicluster<Long>> newCandidates = new ArrayList<Bicluster<Long>>(candidates);
			while (r > 1) {
				List<Bicluster<Long>> newCandidates2 = new ArrayList<Bicluster<Long>>();
				for (Bicluster<Long> otherB : newCandidates) {
					List<Bicluster<Long>> otherLowers = getLowers(otherB, null);
					List<Bicluster<Long>> otherUppers = getUppers(otherB);
					for (Bicluster<Long> b : otherUppers) {
						otherLowers.addAll(getLowers(b, userSet));
					}	
					newCandidates2.addAll(otherLowers);
				}
				candidates.addAll(newCandidates2);
				newCandidates = newCandidates2;
				r--;
			}

			this.neighborhoods.put(userID, candidates);
			return candidates;
		} else {
			List<Bicluster<Long>> candidates = this.neighborhoods.get(userID);
			log.debug("Cached candidate biclusters of {} are {}", sb, candidates);
			return candidates;
		}
	}

	@SuppressWarnings({ "rawtypes" })
	private Bicluster<Long> getBiclusterFromComparable(TreeSet<Comparable> userSet, TreeSet<Comparable> itemSet) {
		Set<Long> users = new HashSet<Long>();
		for (Comparable comp : userSet) {
			users.add((Long) comp);
		}
		Set<Long> items = new HashSet<Long>();
		for (Comparable comp : itemSet) {
			items.add((Long) comp);
		}
		return new Bicluster<Long>(users, items);
	}

	@SuppressWarnings("rawtypes")
	private List<Bicluster<Long>> getLowers(Bicluster<Long> b, TreeSet<Comparable> cur) throws TasteException {
		
		TreeSet<Comparable> itemSet = new TreeSet<Comparable>();
		for (long itemID : b.getSetItems()) {
			itemSet.add(itemID);
		}
		
		List<Bicluster<Long>> lowers = new ArrayList<Bicluster<Long>>();
		Set<TreeSet<Comparable>> validUserSets = new HashSet<TreeSet<Comparable>>();
		DataModel model = getDataModel();
		LongPrimitiveIterator it = model.getItemIDs();
		while (it.hasNext()) {
			long itemID = it.nextLong();
			if (!b.containsItem(itemID)) {
				int nbZeros = 0;
				Iterator<Long> itU = b.getUsers();
				while (itU.hasNext()) {
					long userID = itU.next();
					Float rating = model.getPreferenceValue(userID, itemID);
					if (rating == null || rating < this.threshold) {
						nbZeros++;
					} else {
						break;
					}
				}
				if (nbZeros == b.getNbUsers()) {
					continue;
				}
				TreeSet<Comparable> items = new TreeSet<Comparable>(itemSet);
				items.add(itemID);
				TreeSet<Comparable> theUsers = this.ctx.getExtent(items);
				if (!theUsers.isEmpty() && (cur == null || !theUsers.equals(cur))) {
					boolean valid = true;
					List<TreeSet<Comparable>> toRemove = new ArrayList<TreeSet<Comparable>>();
					for (TreeSet<Comparable> otherSet : validUserSets) {
						if (!valid) {
							break;
						}
						if (otherSet.containsAll(theUsers) || (cur != null && cur.containsAll(theUsers))) {
							valid = false;
						} else if (theUsers.containsAll(otherSet)) {
							toRemove.add(otherSet);
						}
					}
					if (valid) {
						validUserSets.add(theUsers);
						for (TreeSet<Comparable> otherSet : toRemove) {
							validUserSets.remove(otherSet);
						}
					}
				}
			}
		}
		for (TreeSet<Comparable> theUsers : validUserSets) {
			TreeSet<Comparable> theItems = this.ctx.getIntent(theUsers);
			Bicluster<Long> theB = getBiclusterFromComparable(theUsers, theItems);
			lowers.add(theB);
		}
		return lowers;
	}

	@SuppressWarnings("rawtypes")
	private List<Bicluster<Long>> getUppers(Bicluster<Long> b) throws TasteException {
		
		TreeSet<Comparable> userSet = new TreeSet<Comparable>();
		for (long userID : b.getSetUsers()) {
			userSet.add(userID);
		}
		
		List<Bicluster<Long>> uppers = new ArrayList<Bicluster<Long>>();
		Set<TreeSet<Comparable>> validItemSets = new HashSet<TreeSet<Comparable>>();
		DataModel model = getDataModel();
		LongPrimitiveIterator it = model.getUserIDs();
		while (it.hasNext()) {
			long userID = it.nextLong();
			if (!b.containsUser(userID)) {
				int nbZeros = 0;
				Iterator<Long> itI = b.getItems();
				while (itI.hasNext()) {
					long itemID = itI.next();
					Float rating = model.getPreferenceValue(userID, itemID);
					if (rating == null || rating < this.threshold) {
						nbZeros++;
					} else {
						break;
					}
				}
				if (nbZeros == b.getNbItems()) {
					continue;
				}
				TreeSet<Comparable> users = new TreeSet<Comparable>(userSet);
				users.add(userID);
				TreeSet<Comparable> theItems = this.ctx.getIntent(users);
				if (!theItems.isEmpty()) {
					boolean valid = true;
					List<TreeSet<Comparable>> toRemove = new ArrayList<TreeSet<Comparable>>();
					for (TreeSet<Comparable> otherSet : validItemSets) {
						if (!valid) {
							break;
						}
						if (otherSet.containsAll(theItems)) {
							valid = false;
						} else if (theItems.containsAll(otherSet)) {
							toRemove.add(otherSet);
						}
					}
					if (valid) {
						validItemSets.add(theItems);
						for (TreeSet<Comparable> otherSet : toRemove) {
							validItemSets.remove(otherSet);
						}
					}
				}
			}
		}
		for (TreeSet<Comparable> theItems : validItemSets) {
			TreeSet<Comparable> theUsers = this.ctx.getExtent(theItems);
			Bicluster<Long> theB = getBiclusterFromComparable(theUsers, theItems);
			uppers.add(theB);
		}
		return uppers;
	}

	private double biSim(Bicluster<Long> b1, Bicluster<Long> b2) throws TasteException {
		DataModel model = getDataModel();
		Bicluster<Long> b = b1.copy();
		b.merge(b2);
		int zeros = 0;
		Iterator<Long> itU = b.getUsers();
		while (itU.hasNext()) {
			long userID = itU.next();
			Iterator<Long> itI = b.getItems();
			while (itI.hasNext()) {
				long itemID = itI.next();
				Float rating = model.getPreferenceValue(userID, itemID);
				if (rating == null || rating < this.threshold) {
					zeros++;
				}
			}
		}
		int cnt = b.getNbUsers() * b.getNbItems();
		double x = (double) zeros / (double) cnt;
		return 1 - x;
	}

	private final class Estimator implements TopItems.Estimator<Long> {

		private final long theUserID;

		private Estimator(long theUserID) {
			this.theUserID = theUserID;
		}

		@Override
		public double estimate(Long itemID) throws TasteException {
			return estimatePreference(theUserID, itemID);
		}
	}

	@Override
	public void refresh(Collection<Refreshable> alreadyRefreshed) {
	}

}
