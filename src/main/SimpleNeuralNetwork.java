package main;

import java.util.Arrays;
import java.util.List;

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

	public SimpleNeuralNetwork(int ... sizes) {
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

	private void initWeights() {
		for (int l = 0; l < nLayers - 1; ++l) {
			for (int i = 0; i < weights[l].length; ++i) {
				for (int j = 0; j < weights[l][i].length; ++j) {
					// XXX seed used
					weights[l][i][j] = (double) (Math.random()) - 0.5f;
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
		activations[0] = Arrays.copyOf(input, input.length);
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
	 * Activation function (sigma).
	 * 
	 * @param x
	 * @return
	 */
	private static double sigma(double x) {
		return (double) (1.0 / (1.0 + Math.exp(-x)));
	}

	/**
	 * Trains with backpropagation, specified number of steps times.
	 * 
	 * @param exampleSet
	 * @param nSteps
	 *            - steps number
	 * @param eta
	 *            - learning rate
	 * @param lambda
	 *            - L2 regularisation factor (must be < 1)
	 * @return cost function values during training
	 */
	public double[] trainBackpropagation(List<Example> exampleSet, int nSteps,
			double eta, double lambda) {
//		int m = exampleSet.size();
		double[] costLog = new double[nSteps];
		for (int step = 0; step < nSteps; ++step) {
			double cost = 0;
			for (Example ex : exampleSet) {
//				SimpleNeuralNetwork sameNet = clone();
				
				multWeightsDeltasByMomentum();
				// forward propagating, which inits activations at each layer
				compute(ex.getInput());
				double[] correct = ex.getOutput();
				// computing deltas for output layer
				int l = nLayers - 1;
				int out = layers[l];
				double[] delta = new double[out];
				double[] dJ_do = costFunction.derivative(correct, activations[l]);
				for (int j = 0; j < out; ++j) {
					delta[j] = activations[l][j] * (1 - activations[l][j]) * dJ_do[j];
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
							// gradient checking
							double epsilon = 0.0001f;
							SimpleNeuralNetwork netMinus = sameNet.clone();
							netMinus.weights[l - 1][j][i] -= epsilon;
							netMinus.compute(ex.getInput());
							double costMinus = cost(correct,
									netMinus.activations[nLayers - 1]);
							SimpleNeuralNetwork netPlus = sameNet.clone();
							netPlus.weights[l - 1][j][i] += epsilon;
							netPlus.compute(ex.getInput());
							double costPlus = cost(correct,
									netPlus.activations[nLayers - 1]);
							double der_delta = -eta*(costPlus - costMinus) / 2 / epsilon;
							double bp_delta = -eta * delta[i] * activations[l - 1][j];
							System.out.print(der_delta);
							System.out.println("  " + bp_delta);
							weights[l - 1][j][i] += der_delta;
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
				
				// updating weights after whole example set propagation
				updateWeights();
			}
			
			// L2-regularisation: decreasing all weights by a factor of lambda<1
			for (int l = 0; l < nLayers - 1; ++l) {
				for (int i = 0; i < weights[l].length; ++i) {
					for (int j = 0; j < weights[l][i].length; ++j) {
						weightsDeltas[l][i][j] += -lambda * (weights[l][i][j]);
					}
				}
			}
			
			// computing value of gradient just for stats
			double grad = 0;
			for (int l = 0; l < nLayers - 1; ++l) {
				for (int i = 0; i < weights[l].length; ++i) {
					for (int j = 0; j < weights[l][i].length; ++j) {
						grad += weightsDeltas[l][i][j] * weightsDeltas[l][i][j];
					}
				}
			}

			// print weights
			System.out.printf("Step %d. ", step);
//			System.out.print(Arrays.deepToString(weights) + ", \t");
			costLog[step] = cost;
			System.out.printf("Error = %f. |grad| = %f", cost, Math.sqrt(grad));
			System.out.println();
		}
		return costLog;
	}

	/**
	 * Multiply all weights deltas by momentum or assign to zero if momentum is zero.
	 */
	private void multWeightsDeltasByMomentum() {
		for (int l = 0; l < nLayers - 1; ++l) {
			for (int i = 0; i < weights[l].length; ++i) {
//				Arrays.fill(weightsDeltas[l][i], 0);
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
