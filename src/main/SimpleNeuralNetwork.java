package main;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import static main.Utils.*;

/**
 * Multi-layer neural network. Can be trained by backpropagation.
 * 
 * @author misha
 *
 */
public class SimpleNeuralNetwork {

	private int nLayers;
	private int[] layers;

	// (Nlayers-1) matrices of sizes: (|layer_i| + 1) X |layer_(i+1)|
	private double[][][] weights;
	private double[][][] weightsDeltas;

	// (Nlayers-1) vectors of sizes: |layer_i|
	private double[][] activations;

	private ICost costFunction;
	private double momentum;

	public SimpleNeuralNetwork(int... sizes) {
		this.nLayers = sizes.length;
		layers = sizes;
		weights = new double[nLayers - 1][][];
		weightsDeltas = new double[nLayers - 1][][];
		activations = new double[nLayers][];
		activations[0] = new double[layers[0]];
		for (int l = 0; l < nLayers - 1; ++l) {
			int in = layers[l] + 1;
			int out = layers[l + 1];
			weights[l] = new double[in][out];
			weightsDeltas[l] = new double[in][out];
			for (int i = 0; i < in; ++i) {
				weights[l][i] = new double[out];
				weightsDeltas[l][i] = new double[out];
			}
			activations[l + 1] = new double[out];
		}

		initWeights();
		costFunction = ICost.QUADRATIC;
	}

	SimpleNeuralNetwork(double[][][] newWeights) {
		nLayers = newWeights.length + 1;
		layers = new int[nLayers];
		weights = new double[nLayers - 1][][];
		weightsDeltas = new double[nLayers - 1][][];
		activations = new double[nLayers][];
		activations[0] = new double[layers[0]];
		for (int l = 0; l < nLayers - 1; ++l) {
			layers[l] = newWeights[l].length - 1;
			int in = layers[l] + 1;
			int out = newWeights[l][0].length;
			weights[l] = new double[in][out];
			weightsDeltas[l] = new double[in][out];
			for (int i = 0; i < in; ++i) {
				weights[l][i] = new double[out];
				weightsDeltas[l][i] = new double[out];
				for (int j = 0; j < out; ++j) {
					weights[l][i][j] = newWeights[l][i][j];
					weightsDeltas[l][i][j] = newWeights[l][i][j];
				}
			}
			activations[l + 1] = new double[out];
		}
		layers[nLayers - 1] = newWeights[nLayers - 2][0].length;
	}

	public void setCostFunction(ICost costFunction) {
		this.costFunction = costFunction;
	}

	public void setMomentum(double momentum) {
		this.momentum = momentum;
	}

	public int[] getLayers() {
		return layers;
	}

	public double getMomentum() {
		return momentum;
	}

	private void initWeights() {
		for (int l = 0; l < nLayers - 1; ++l) {
			for (int i = 0; i < weights[l].length; ++i) {
				for (int j = 0; j < weights[l][i].length; ++j) {
					// XXX seed used
					weights[l][i][j] = Math.random() - 0.5f;
				}
			}
		}
	}

	/**
	 * Forward propagation of the input. Activations at each layer will be
	 * updated.
	 * 
	 * @param input
	 * @return
	 */
	public double[] compute(double[] input) {
		activations[0] = input;
		for (int l = 0; l < nLayers - 1; ++l) {
			int in = activations[l].length;
			int out = activations[l + 1].length;
			// computing next activations vector by applying weights to current
			// activations
			Arrays.fill(activations[l + 1], 0);
			for (int i = 0; i < in; ++i) {
				for (int j = 0; j < out; ++j) {
					activations[l + 1][j] += weights[l][i][j]
							* activations[l][i];
				}
			}
			// adding extra 1 and applying sigma function
			for (int j = 0; j < out; ++j) {
				activations[l + 1][j] = sigma(weights[l][in][j]
						+ activations[l + 1][j]);
			}
		}
		return activations[nLayers - 1];
	}

	/**
	 * Train the neural net with stochastic gradient descent using
	 * backpropagation. Specified number of steps times do the following: pick a
	 * minibatch from given set of examples, perform backpropagation and
	 * regularization (if needed).
	 * 
	 * @param examples
	 * @param validationSet 
	 * @param nSteps
	 *            - steps number
	 * @param learnRate
	 *            - learning rate
	 * @param lambda
	 *            - L2 regularisation factor (must be < 1)
	 * @param batchSize
	 *            - size of minibatch used by stochastic gradient descent
	 * @return cost function values during training
	 */
	public double[] trainGradientDecsent(Collection<Example> examples,
			Collection<Example> validationSet, int nSteps, double learnRate, double lambda, int batchSize) {
		double[] costLog = new double[nSteps];
		Iterator<Example> iterator = examples.iterator();
		
		for (int step = 0; step < nSteps; ++step) {
			double cost = 0;
			for(int i=0; i<examples.size()/batchSize; ++i)  {
				// picking a minibatch from example set
				Collection<Example> miniBatch = new ArrayList<>(batchSize);
				for (int size = 0; size < batchSize; ++size) {
					if (!iterator.hasNext()) {
						iterator = examples.iterator();
					}
					miniBatch.add(iterator.next());
				}
	
				// performing backpropagation using the minibatch
				multWeightsDeltasByMomentum();
				cost += backpropagationStep(miniBatch, learnRate / miniBatch.size());
				updateWeights();
			}

//			// L2-regularisation: decreasing all weights by a factor of
//			// lambda<1
//			for (int l = 0; l < nLayers - 1; ++l) {
//				for (int i = 0; i < weights[l].length; ++i) {
//					for (int j = 0; j < weights[l][i].length; ++j) {
//						weightsDeltas[l][i][j] += -lambda * (weights[l][i][j]);
//					}
//				}
//			}
			
			// // computing value of gradient just for stats
			double grad = 0;
//			for (int l = 0; l < nLayers - 1; ++l) {
//				for (int i = 0; i < weights[l].length; ++i) {
//					for (int j = 0; j < weights[l][i].length; ++j) {
//						grad += weightsDeltas[l][i][j] * weightsDeltas[l][i][j];
//					}
//				}
//			}

			// evaluating on validation set
			double acc = computeAccuracy(validationSet);			
			
			// print weights
			System.out.printf("Step %d. ", step);
			// System.out.print(Arrays.deepToString(weights) + ", \t");
			costLog[step] = acc;
			System.out.printf("Error = %f. |grad| = %f, validation acc = %f",
					cost, Math.sqrt(grad), acc);
			System.out.println();
		}
		return costLog;
	}

	/**
	 * Recalculate weights deltas computed using given mini batch of examples
	 * and learning rate.
	 * 
	 * @param miniBatch
	 * @param eta
	 *            - learning rate
	 * @return total cost on given minibatch
	 */
	private double backpropagationStep(Collection<Example> miniBatch, double eta) {
		// SimpleNeuralNetwork sameNet = clone();
		double cost = 0;
		for (Example ex : miniBatch) {
			// forward propagating, which inits activations at each layer
			compute(ex.getInput());
			double[] correct = ex.getOutput();
			// computing deltas for output layer
			int l = nLayers - 1;
			int out = layers[l];
			double[] delta = new double[out];
			double[] dJ_do = costFunction.derivative(correct, activations[l]);
			for (int j = 0; j < out; ++j) {
				delta[j] = activations[l][j] * (1 - activations[l][j])
						* dJ_do[j];
			}
			// updating weights deltas and computing deltas for other layers
			for (; l > 0; --l) {
				int in = layers[l - 1];
				out = layers[l];
				// deltas at layer (l-1)
				double[] nextDelta = new double[in];
				for (int j = 0; j < in; ++j) {
					double sum = 0;
					for (int k = 0; k < out; ++k) {
						sum += weights[l - 1][j][k] * delta[k];
					}
					nextDelta[j] = activations[l - 1][j]
							* (1 - activations[l - 1][j]) * sum;
				}
				// updating weights deltas
				for (int j = 0; j < in; ++j) {
					for (int i = 0; i < out; ++i) {
						/*
						 * // gradient checking double epsilon = 0.0001f;
						 * SimpleNeuralNetwork netMinus = sameNet.clone();
						 * netMinus.weights[l - 1][j][i] -= epsilon;
						 * netMinus.compute(ex.getInput()); double costMinus =
						 * cost(correct, netMinus.activations[nLayers - 1]);
						 * SimpleNeuralNetwork netPlus = sameNet.clone();
						 * netPlus.weights[l - 1][j][i] += epsilon;
						 * netPlus.compute(ex.getInput()); double costPlus =
						 * cost(correct, netPlus.activations[nLayers - 1]);
						 * double der_delta = -eta*(costPlus - costMinus) / 2 /
						 * epsilon; double bp_delta = -eta * delta[i] *
						 * activations[l - 1][j]; System.out.print(der_delta);
						 * System.out.println("  " + bp_delta); weights[l -
						 * 1][j][i] += der_delta;
						 */
						weightsDeltas[l - 1][j][i] += -eta * delta[i]
								* activations[l - 1][j];
					}
				}
				for (int i = 0; i < out; ++i) {
					weightsDeltas[l - 1][in][i] += -eta * delta[i] * 1;
				}
				delta = nextDelta;
			}
			cost += costFunction.cost(correct, activations[nLayers - 1]);
		}
		return cost;
	}

	/**
	 * Compute average squared error on given example set.
	 * 
	 * @param examples
	 * @return
	 */
	public double computeError(Collection<Example> examples) {
		return computeError(examples, costFunction);
	}

	/**
	 * Compute average error on given example set using specified cost function.
	 * 
	 * @param examples
	 * @param cost
	 * @return
	 */
	public double computeError(Collection<Example> examples, ICost cost) {
		double res = 0;
		for (Example ex : examples) {
			res += cost.cost(ex.getOutput(), compute(ex.getInput()));
		}
		return res / examples.size();
	}

	private double computeAccuracy(Collection<Example> testSet) {
		int correct = 0;
		int total = testSet.size();
		for (Example ex : testSet) {
			if (argmax(compute(ex.getInput())) == argmax(ex.getOutput())) {
				correct++;
			}
		}
		return 1.0 * correct / total;
	}

	/**
	 * Multiply all weights deltas by momentum or assign to zero if momentum is
	 * zero.
	 */
	private void multWeightsDeltasByMomentum() {
		for (int l = 0; l < nLayers - 1; ++l) {
			for (int i = 0; i < weights[l].length; ++i) {
				// Arrays.fill(weightsDeltas[l][i], 0);
				for (int j = 0; j < weights[l][i].length; ++j) {
					weightsDeltas[l][i][j] = momentum * weightsDeltas[l][i][j];
				}
			}
		}
	}

	/**
	 * Update all weights with weight deltas.
	 */
	private void updateWeights() {
		for (int l = 0; l < nLayers - 1; ++l) {
			for (int i = 0; i < weights[l].length; ++i) {
				for (int j = 0; j < weights[l][i].length; ++j) {
					weights[l][i][j] += weightsDeltas[l][i][j];
				}
			}
		}
	}

	public SimpleNeuralNetwork clone() {
		return new SimpleNeuralNetwork(weights);
	}

}
