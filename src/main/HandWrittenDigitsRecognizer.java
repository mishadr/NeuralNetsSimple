package main;

import static main.Utils.argmax;
import static main.Utils.plotResults;
import static main.Utils.saveResult;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

public class HandWrittenDigitsRecognizer {

	private Collection<Example> trainingSet;
	private Collection<Example> validationSet;
	private Collection<Example> testSet;
	private int inputSize;
	private int outputSize;
	private SimpleNeuralNetwork network;

	public HandWrittenDigitsRecognizer() {

		int trainSize= 450;
		int validationSize = 1000;
		int testSize = 0;
		loadData(trainSize, testSize, validationSize);
		
		saveResult(
				"Using %d for train, %d for validation, %d for test:",
				trainSize, validationSize, testSize);
		
		int numberOfEpochs = 30;
		int minibatchSize = 10;
		double learnRate = 0.03;
		double momentum = 0.0;

		double[] log = new double[]{};
		double[] learnRates = new double[] { 0.0001, 0.0003, 0.001, 0.003,
				0.01, 0.03, 0.1, 0.3, 1.0 };
		double[] momentums = new double[] { 0.0, 0.2, 0.5, 0.7, 0.9, 0.95, 0.99 };

//		for (double x : learnRates) {
			network = new SimpleNeuralNetwork(inputSize, 1000, outputSize);
			log = trainAndTest(numberOfEpochs, minibatchSize, learnRate,
					momentum, ICost.CROSS_ENTROPY);
//		}

		// plotting errors during network training
		plotResults(log);
	}

	/**
	 * Train and test a network using given parameters.
	 * 
	 * @param numberOfEpochs
	 * @param minibatchSize
	 * @param learnRate
	 * @param momentum
	 * @param cost
	 * @return
	 */
	private double[] trainAndTest(int numberOfEpochs, int minibatchSize,
			double learnRate, double momentum, ICost cost) {
		network.setCostFunction(cost);
		network.setMomentum(momentum);

		// before training
		makeMeasures();

		// training
		long time = System.nanoTime();
		
		double[] log = network.trainGradientDecsent(trainingSet, validationSet,
				numberOfEpochs, learnRate, 0.0, minibatchSize);
		time = System.nanoTime() - time;
		System.out.printf("Time spent on training: %f ms\n", 0.000001 * time);

		// after training
		double accuracy = makeMeasures();
		
		double maxValidationAccuracy = Utils.max(log);
		int maxStep = Utils.argmax(log);

		Utils.saveResult(
				"  NN %s; LR: %f; momentum: %.3f; epochs: %d; batch size: %d.\t acc = %.3f %% at %d",
				Arrays.toString(network.getLayers()), learnRate, momentum,
				numberOfEpochs, minibatchSize, 100 * maxValidationAccuracy, maxStep);

		return log;
	}

	private double makeMeasures() {
		double trainError = network.computeError(trainingSet, ICost.QUADRATIC);
		double testError = network.computeError(testSet, ICost.QUADRATIC);
		System.out.printf(
				"Average square error on training set: %f,\t on test set: %f\n",
				trainError, testError);
		double accuracy = computeAccuracy();
		System.out.printf("Accuracy: %.2f %%\n", 100 * accuracy);
		return accuracy;
	}

	/**
	 * Load data, assign training set, test set and validation set, also input
	 * and output sizes.
	 * 
	 */
	private void loadData(int trainNumber, int testNumber, int validationNumber) {
		trainingSet = new ArrayList<>(trainNumber);
		validationSet = new ArrayList<>(validationNumber);
		Collection<Example> set = trainingSet;
		for (Example ex: MNISTDataReader.read("data/train-labels.idx1-ubyte",
				"data/train-images.idx3-ubyte", trainNumber + validationNumber)) {
			if(--trainNumber < 0) {
				set = validationSet;
			}
			set.add(ex);
		}
		testSet = MNISTDataReader.read("data/t10k-labels.idx1-ubyte",
				"data/t10k-images.idx3-ubyte", testNumber);

		inputSize = trainingSet.iterator().next().getInput().length;
		outputSize = trainingSet.iterator().next().getOutput().length;
	}

	private double computeAccuracy() {
		int correct = 0;
		int total = testSet.size();
		for (Example ex : testSet) {
			if (argmax(network.compute(ex.getInput())) == argmax(ex.getOutput())) {
				correct++;
			}
		}
		return 1.0 * correct / total;
	}

}
