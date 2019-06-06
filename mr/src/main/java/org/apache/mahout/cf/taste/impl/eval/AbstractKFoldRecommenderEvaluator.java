package org.apache.mahout.cf.taste.impl.eval;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Iterator;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.common.RunningAverageAndStdDev;

import org.apache.mahout.cf.taste.eval.FoldDataSplitter;

import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public abstract class AbstractKFoldRecommenderEvaluator {

	private final DataModel dataModel;
	private final FoldDataSplitter folds;
	public double noEstimateCounterAverage = 0.0;
	public double totalEstimateCount = 0.0;
	public double totalEstimateCountAverage = 0.0;

	private static final Logger log = LoggerFactory.getLogger(AbstractKFoldRecommenderEvaluator.class);

	public AbstractKFoldRecommenderEvaluator(DataModel dataModel, int nbFolds) throws TasteException {
		Preconditions.checkArgument(dataModel != null, "dataModel is null");
		Preconditions.checkArgument(nbFolds > 1, "nbFolds must be > 1");

		this.dataModel = dataModel;
		this.folds = new KFoldDataSplitter(dataModel, nbFolds);
	}
	
	public AbstractKFoldRecommenderEvaluator(DataModel dataModel, FoldDataSplitter splitter) throws TasteException {
		Preconditions.checkArgument(dataModel != null, "dataModel is null");
		Preconditions.checkArgument(splitter != null, "splitter is null");

		this.dataModel = dataModel;
		this.folds = splitter;
	}

	public double getNoEstimateCounterAverage() {
		return noEstimateCounterAverage;
	}

	public double getTotalEstimateCount() {
		return totalEstimateCount;
	}

	public double getTotalEstimateCountAverage() {
		return totalEstimateCountAverage;
	}

	/**
	 * We use the same evaluate function from the RecommenderEvaluator interface the
	 * trainingPercentage is used as the number of folds, so it can have values
	 * bigger than 0 to the number of folds.
	 */
	public double evaluate(RecommenderBuilder recommenderBuilder) throws TasteException {
		Preconditions.checkNotNull(recommenderBuilder);

		log.info("Beginning evaluation using of {}", dataModel);

		double result = Double.NaN;
		List<Double> intermediateResults = new ArrayList<>();
		List<Integer> unableToRecoomend = new ArrayList<>();
		List<Integer> averageEstimateCounterIntermediate = new ArrayList<>();

		noEstimateCounterAverage = 0.0;
		totalEstimateCount = 0.0;
		totalEstimateCountAverage = 0.0;
		int totalEstimateCounter = 0;

		Iterator<Fold> itF = this.folds.getFolds();
		int k = 0;
		while (itF.hasNext()) {

			Fold fold = itF.next();

			DataModel trainingModel = fold.getTraining();
			FastByIDMap<PreferenceArray> testPrefs = fold.getTesting();

			Recommender recommender = recommenderBuilder.buildRecommender(trainingModel, fold);

			Double[] retVal = getEvaluation(testPrefs, recommender);
			double intermediate = retVal[0];
			int noEstimateCounter = ((Double) retVal[1]).intValue();
			totalEstimateCounter += ((Double) retVal[2]).intValue();
			averageEstimateCounterIntermediate.add(((Double) retVal[2]).intValue());

			log.info("Evaluation result from fold {} : {}", k, intermediate);
			log.info("Average Unable to recommend  for fold {} in: {} cases out of {}", k++, noEstimateCounter,
					((Double) retVal[2]).intValue());
			intermediateResults.add(intermediate);
			unableToRecoomend.add(noEstimateCounter);

		}

		double sum = 0;
		double noEstimateSum = 0;
		double totalEstimateSum = 0;
		// Sum the results in each fold
		for (int i = 0; i < intermediateResults.size(); i++) {
			if (!Double.isNaN(intermediateResults.get(i))) {
				sum += intermediateResults.get(i);
				noEstimateSum += unableToRecoomend.get(i);
				totalEstimateSum += averageEstimateCounterIntermediate.get(i);
			}
		}

		if (sum > 0) {
			// Get an average for the folds
			result = sum / intermediateResults.size();
		}

		double noEstimateCount = 0;
		if (noEstimateSum > 0) {
			noEstimateCount = noEstimateSum / unableToRecoomend.size();
		}

		double avgEstimateCount = 0;
		if (totalEstimateSum > 0) {
			avgEstimateCount = totalEstimateSum / averageEstimateCounterIntermediate.size();
		}

		log.info("Average Evaluation result: {} ", result);
		log.info("Average Unable to recommend in: {} cases out of avg. {} cases or total {} ", noEstimateCount,
				avgEstimateCount, totalEstimateCounter);

		noEstimateCounterAverage = noEstimateCount;
		totalEstimateCount = totalEstimateCounter;
		totalEstimateCountAverage = avgEstimateCount;
		return result;
	}

	private Double[] getEvaluation(FastByIDMap<PreferenceArray> testPrefs, Recommender recommender)
			throws TasteException {
		reset();
		Collection<Callable<Void>> estimateCallables = Lists.newArrayList();
		AtomicInteger noEstimateCounter = new AtomicInteger();
		AtomicInteger totalEstimateCounter = new AtomicInteger();
		for (Map.Entry<Long, PreferenceArray> entry : testPrefs.entrySet()) {
			estimateCallables.add(
					new PreferenceEstimateCallable(recommender, entry.getKey(), entry.getValue(), noEstimateCounter));
		}
		log.info("Beginning evaluation of {} users", estimateCallables.size());
		RunningAverageAndStdDev timing = new FullRunningAverageAndStdDev();
		execute(estimateCallables, noEstimateCounter, timing);

		Double[] retVal = new Double[3];
		retVal[0] = computeFinalEvaluation();
		retVal[1] = (double) noEstimateCounter.get();
		retVal[2] = (double) totalEstimateCounter.get();
		// retVal.put(computeFinalEvaluation(), noEstimateCounter.get());
		// return computeFinalEvaluation();
		return retVal;
	}

	abstract protected void reset();

	abstract protected void processOneEstimate(float estimatedPreference, Preference realPref);

	abstract protected double computeFinalEvaluation();

	public final class PreferenceEstimateCallable implements Callable<Void> {

		private final Recommender recommender;
		private final long testUserID;
		private final PreferenceArray prefs;
		private final AtomicInteger noEstimateCounter;

		public PreferenceEstimateCallable(Recommender recommender, long testUserID, PreferenceArray prefs,
				AtomicInteger noEstimateCounter) {
			this.recommender = recommender;
			this.testUserID = testUserID;
			this.prefs = prefs;
			this.noEstimateCounter = noEstimateCounter;
		}

		@Override
		public Void call() throws TasteException {
			for (Preference realPref : prefs) {
				float estimatedPreference = Float.NaN;
				try {
					estimatedPreference = recommender.estimatePreference(testUserID, realPref.getItemID());
				} catch (NoSuchUserException nsue) {
					// It's possible that an item exists in the test data but not training data in
					// which case
					// NSEE will be thrown. Just ignore it and move on.
					log.info("User exists in test data but not training data: {}", testUserID);
				} catch (NoSuchItemException nsie) {
					log.info("Item exists in test data but not training data: {}", realPref.getItemID());
				}
				if (Float.isNaN(estimatedPreference)) {
					noEstimateCounter.incrementAndGet();
				} else {
					processOneEstimate(estimatedPreference, realPref);
				}
			}
			return null;
		}

	}

	protected static void execute(Collection<Callable<Void>> callables, AtomicInteger noEstimateCounter,
			RunningAverageAndStdDev timing) throws TasteException {

		Collection<Callable<Void>> wrappedCallables = wrapWithStatsCallables(callables, noEstimateCounter, timing);
		int numProcessors = Runtime.getRuntime().availableProcessors();
		ExecutorService executor = Executors.newFixedThreadPool(numProcessors);
		log.info("Starting timing of {} tasks in {} threads", wrappedCallables.size(), numProcessors);
		try {
			List<Future<Void>> futures = executor.invokeAll(wrappedCallables);
			// Go look for exceptions here, really
			for (Future<Void> future : futures) {
				future.get();
			}

		} catch (InterruptedException ie) {
			throw new TasteException(ie);
		} catch (ExecutionException ee) {
			throw new TasteException(ee.getCause());
		}

		executor.shutdown();
		try {
			executor.awaitTermination(10, TimeUnit.SECONDS);
		} catch (InterruptedException e) {
			throw new TasteException(e.getCause());
		}
	}

	private static Collection<Callable<Void>> wrapWithStatsCallables(Iterable<Callable<Void>> callables,
			AtomicInteger noEstimateCounter, RunningAverageAndStdDev timing) {
		Collection<Callable<Void>> wrapped = new ArrayList<>();
		int count = 0;
		for (Callable<Void> callable : callables) {
			boolean logStats = count++ % 1000 == 0; // log every 1000 or so iterations
			wrapped.add(new StatsCallable(callable, logStats, timing, noEstimateCounter));
		}
		return wrapped;
	}
}
