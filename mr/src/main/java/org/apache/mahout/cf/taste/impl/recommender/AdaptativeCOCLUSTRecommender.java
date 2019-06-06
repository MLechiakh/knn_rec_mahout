package org.apache.mahout.cf.taste.impl.recommender;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import com.google.common.base.Preconditions;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.eval.Fold;
import org.apache.mahout.cf.taste.impl.eval.KFoldRecommenderPredictionEvaluator;
import org.apache.mahout.cf.taste.impl.recommender.AbstractRecommender;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.common.RandomUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class AdaptativeCOCLUSTRecommender extends AbstractRecommender {

	
	private Recommender curRec;
	private final int nbMaxIterations;
	private static final Logger log = LoggerFactory.getLogger(AdaptativeCOCLUSTRecommender.class);

	public AdaptativeCOCLUSTRecommender(DataModel dataModel, int maxIter, CandidateItemsStrategy strategy)
			throws TasteException {
		super(dataModel, strategy);
		this.nbMaxIterations = maxIter;
		init();
	}

	public AdaptativeCOCLUSTRecommender(DataModel dataModel, int maxIter) throws TasteException {
		super(dataModel);
		this.nbMaxIterations = maxIter;
		init();
	}

	private void init() throws TasteException {
		
		Parameters params = new Parameters();
		
		DataModel dataModel = this.getDataModel();
		KFoldRecommenderPredictionEvaluator evaluator = new KFoldRecommenderPredictionEvaluator(dataModel, 5);

		RecommenderBuilder builder = new COCLUSTBuilder(params.getK(), params.getL(), this.nbMaxIterations);
		double rmse = evaluator.evaluate(builder).getRMSE();
		
		for (int s = 0; s < this.nbMaxIterations; s++) {
			
			if (params.giveUp()) {
				break;
			}

			log.info("Current parameters are k={} and l={}, rmse on validation set is {}", params.getK(), params.getL(), rmse);

			params.next();
			builder = new COCLUSTBuilder(params.getK(), params.getL(), this.nbMaxIterations);
			double newRmse = evaluator.evaluate(builder).getRMSE();
			if (newRmse >= rmse) {
				params.revert();
				log.info("New rmse is {}, let's revert", newRmse);
			} else {
				params.commit();
				rmse = newRmse;
				log.info("It's better, new parameters are {} and {}", params.getK(), params.getL());
			}
		}
		
		builder = new COCLUSTBuilder(params.getK(), params.getL(), this.nbMaxIterations);
		this.curRec = builder.buildRecommender(dataModel);
		log.info("Final parameters are k={} and l={}", params.getK(), params.getL());
	}

	class COCLUSTBuilder implements RecommenderBuilder {
		
		private final int k;
		private final int l;
		private final int ni;

		COCLUSTBuilder(int nbUserClusters, int nbItemClusters, int nbIter) {
			this.k = nbUserClusters;
			this.l = nbItemClusters;
			this.ni = nbIter;
		}

		@Override
		public Recommender buildRecommender(DataModel dataModel) throws TasteException {
			return new COCLUSTRecommender(dataModel, this.k, this.l, this.ni);
		}

		@Override
		public Recommender buildRecommender(DataModel dataModel, Fold fold) throws TasteException {
			return this.buildRecommender(dataModel);
		}
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

	/**
	 * a preference is estimated by considering the chessboard biclustering computed
	 */
	@Override
	public float estimatePreference(long userID, long itemID) throws TasteException {
		DataModel model = getDataModel();
		Float actualPref = model.getPreferenceValue(userID, itemID);
		if (actualPref != null) {
			return actualPref;
		}
		return this.curRec.estimatePreference(userID, itemID);
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

	/**
	 * Refresh the data model and factorization.
	 */
	@Override
	public void refresh(Collection<Refreshable> alreadyRefreshed) {
	}
	
	class Parameters {
		
		private int k;
		private int l;
		private int kBak;
		private int lBak;
		private final Random random;
		private int sinceLastCommit;
		
		Parameters() {
			this.k = 1;
			this.l = 1;
			this.kBak = this.k;
			this.lBak = this.l;
			this.random = RandomUtils.getRandom();
			this.sinceLastCommit = 0;
		}
		
		int getK() {
			return this.k;
		}
		
		int getL() {
			return this.l;
		}
		
		void next() {
			this.kBak = this.k;
			this.lBak = this.l;
			int opt = random.nextInt(2);
			int step = random.nextInt(3) + 1;
			if (opt == 0) {
				this.k += step;
			} else {
				this.l += step;
			}
		}
		
		void revert() {
			this.k = this.kBak;
			this.l = this.lBak;
			this.sinceLastCommit++;
		}
		
		void commit() {
			this.kBak = this.k;
			this.lBak = this.l;
			this.sinceLastCommit = 0;
		}
		
		boolean giveUp() {
			return this.sinceLastCommit > 4;
		}
		
	}

}
