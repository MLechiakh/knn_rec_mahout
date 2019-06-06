package org.apache.mahout.cf.taste.impl.recommender;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.Callable;

import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.Bicluster;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.impl.recommender.svd.RatingSGDFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.SVDRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.neighborhood.UserBiclusterNeighborhood;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.UserBiclusterSimilarity;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

public class BBCFRecommender extends AbstractRecommender {

	private static final Logger log = LoggerFactory.getLogger(BBCFRecommender.class);

	private final UserBiclusterNeighborhood neighborhood;
	private final UserBiclusterSimilarity similarity;
	private final RefreshHelper refreshHelper;
	private FastByIDMap<Recommender> subrecs;
	private Recommender backup;

	public BBCFRecommender(DataModel dataModel, UserBiclusterNeighborhood neighborhood,
			UserBiclusterSimilarity similarity) throws TasteException {
		super(dataModel);
		Preconditions.checkArgument(neighborhood != null, "neighborhood is null");
		this.neighborhood = neighborhood;
		this.similarity = similarity;
		try {
			this.subrecs = new FastByIDMap<Recommender>(dataModel.getNumUsers());
		} catch (TasteException e) {
			this.subrecs = new FastByIDMap<Recommender>();
		}
		this.refreshHelper = new RefreshHelper(new Callable<Void>() {
			@Override
			public Void call() {
				return null;
			}
		});
		refreshHelper.addDependency(dataModel);
		refreshHelper.addDependency(similarity);
		refreshHelper.addDependency(neighborhood);
		this.backup = new SVDRecommender(dataModel, new RatingSGDFactorizer(dataModel, 18, 30),
				this.candidateItemsStrategy);
	}

	public BBCFRecommender(DataModel dataModel, UserBiclusterNeighborhood neighborhood,
			UserBiclusterSimilarity similarity, CandidateItemsStrategy strategy) throws TasteException {
		super(dataModel, strategy);
		Preconditions.checkArgument(neighborhood != null, "neighborhood is null");
		this.neighborhood = neighborhood;
		this.similarity = similarity;
		try {
			this.subrecs = new FastByIDMap<Recommender>(dataModel.getNumUsers());
		} catch (TasteException e) {
			this.subrecs = new FastByIDMap<Recommender>();
		}
		this.refreshHelper = new RefreshHelper(new Callable<Void>() {
			@Override
			public Void call() {
				return null;
			}
		});
		refreshHelper.addDependency(dataModel);
		refreshHelper.addDependency(similarity);
		refreshHelper.addDependency(neighborhood);
		this.backup = new SVDRecommender(dataModel, new RatingSGDFactorizer(dataModel, 18, 30),
				this.candidateItemsStrategy);
	}

	public UserBiclusterSimilarity getSimilarity() {
		return similarity;
	}

	@Override
	public List<RecommendedItem> recommend(long userID, int howMany, IDRescorer rescorer, boolean includeKnownItems)
			throws TasteException {
		Preconditions.checkArgument(howMany >= 0, "howMany must be at least 0");

		log.debug("Recommending items for user ID '{}'", userID);

		List<Bicluster<Long>> theNeighborhood = neighborhood.getUserNeighborhood(userID);

		if (howMany == 0) {
			return Collections.emptyList();
		}

		Recommender rec = getSubRec(userID, theNeighborhood);
		if (rec != null) {
			try {
				return rec.recommend(userID, howMany, rescorer, includeKnownItems);
			} catch (NoSuchUserException nsue) {
				return addBackupRecommendations(userID, howMany, rescorer, includeKnownItems, Collections.<RecommendedItem>emptyList());
			} catch (NoSuchItemException nsie) {
				return addBackupRecommendations(userID, howMany, rescorer, includeKnownItems, Collections.<RecommendedItem>emptyList());
			}
		} else {
			return addBackupRecommendations(userID, howMany, rescorer, includeKnownItems, Collections.<RecommendedItem>emptyList());
		}
	}

	@Override
	public float estimatePreference(long userID, long itemID) throws TasteException {
		DataModel model = getDataModel();
		Float actualPref = model.getPreferenceValue(userID, itemID);
		if (actualPref != null) {
			return actualPref;
		}
		List<Bicluster<Long>> theNeighborhood = neighborhood.getUserNeighborhood(userID);
		return doEstimatePreference(userID, theNeighborhood, itemID);
	}

	private Recommender getSubRec(long userID, List<Bicluster<Long>> theNeighborhood) throws TasteException {

		if (!subrecs.containsKey(userID)) {

			if (theNeighborhood.size() == 0) {
				return null;
			}

			DataModel model = getDataModel();
			FastByIDMap<PreferenceArray> userData = new FastByIDMap<PreferenceArray>(model.getNumUsers());
			Bicluster<Long> b = new Bicluster<Long>();
			for (Bicluster<Long> otherb : theNeighborhood) {
				b.merge(otherb);
			}
			Iterator<Long> itU = b.getUsers();
			while (itU.hasNext()) {
				long theuserID = itU.next();
				PreferenceArray a = new GenericUserPreferenceArray(b.getNbItems());
				int id = 0;
				Iterator<Long> itI = b.getItems();
				while (itI.hasNext()) {
					long theitemID = itI.next();
					Float rating = model.getPreferenceValue(theuserID, theitemID);
					if (rating != null) {
						a.set(id, new GenericPreference(theuserID, theitemID, rating));
						id++;
					}
				}
				userData.put(theuserID, a);
			}
			DataModel submodel = new GenericDataModel(userData);

			ItemSimilarity sim = new PearsonCorrelationSimilarity(submodel);
			Recommender rec = new GenericItemBasedRecommender(submodel, sim,
					new PreferredItemsNeighborhoodCandidateItemsStrategy(),
					new PreferredItemsNeighborhoodCandidateItemsStrategy());
			
			subrecs.put(userID, rec);
			return rec;
		} else {
			return subrecs.get(userID);
		}
	}

	protected float doEstimatePreference(long userID, List<Bicluster<Long>> theNeighborhood, long itemID)
			throws TasteException {
		Recommender rec = getSubRec(userID, theNeighborhood);
		if (rec != null) {
			try {
				return rec.estimatePreference(userID, itemID);
			} catch (NoSuchUserException nsue) {
				return doBackupEstimatePreference(userID, itemID);
			} catch (NoSuchItemException nsie) {
				return doBackupEstimatePreference(userID, itemID);
			}
		} else {
			return doBackupEstimatePreference(userID, itemID);
		}
	}

	private float doBackupEstimatePreference(long userID, long itemID) throws TasteException {
//		return Float.NaN;
		return this.backup.estimatePreference(userID, itemID);
	}
	
	private List<RecommendedItem> addBackupRecommendations(long userID, int howMany, IDRescorer rescorer, boolean includeKnownItems, List<RecommendedItem> l) throws TasteException {
		int missing = howMany - l.size();
		List<RecommendedItem> recommendations = new ArrayList<RecommendedItem>(howMany);
		List<RecommendedItem> more = this.backup.recommend(userID, missing, rescorer, includeKnownItems);
		for (RecommendedItem item : more) {
			long itemID = item.getItemID();
			boolean found = false;
			for (RecommendedItem otherItem : l) {
				long otherItemID = otherItem.getItemID();
				if (itemID == otherItemID) {
					found = true;
					break;
				}
			}
			if (!found) {
				recommendations.add(item);
			}
		}
		recommendations.addAll(l);
		return recommendations;
	}

	@Override
	public void refresh(Collection<Refreshable> alreadyRefreshed) {
		refreshHelper.refresh(alreadyRefreshed);
	}

	@Override
	public String toString() {
		return "BBCFRecommender[neighborhood:" + neighborhood + ']';
	}

}
