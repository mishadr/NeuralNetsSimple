package main;

import java.util.Arrays;
import java.util.List;

public class SimpleNeuralNetwork {

	private int nInput;
	private int nOutput;
	// size: (input+1) X (output)
	private float[][] weights;

	public SimpleNeuralNetwork(int inputSize, int outputSize) {
		nInput = inputSize;
		nOutput = outputSize;
		weights = new float[nInput + 1][nOutput];
		for (int i = 0; i < nInput + 1; ++i) {
			weights[i] = new float[nOutput];
		}
		initWeights();
	}

	private void initWeights() {
		for (int i = 0; i <= nInput; ++i) {
			for (int j = 0; j < nOutput; ++j) {
				weights[i][j] = (float) Math.random()-0.5f;
			}
		}
	}

	public float[] compute(float[] input) {
		float[] res = new float[nOutput];

		for (int i = 0; i < nInput; ++i) {
			for (int j = 0; j < nOutput; ++j) {
				res[j] += weights[i][j]*input[i];
			}
		}
		for (int j = 0; j < nOutput; ++j) {
			res[j] = sigma(weights[nInput][j] + res[j]);
		}

//		for (int i = 0; i < nOutput; ++i) {
//			res[i] = sigma(scalar(input, weights[i]));
//		}
		return res;
	}

	/**
	 * Activation function (sigma).
	 * 
	 * @param x
	 * @return
	 */
	private float sigma(float x) {
		return (float) (1.0 / (1.0 + Math.exp(-x)));
	}

	/**
	 * Scalar product of input vector and weights vector of the same size or 1
	 * more (for additional 1)
	 * 
	 * @param x
	 * @param w
	 */
	private float scalar(float[] x, float[] w) {
		float res = 0;
		int i = 0;
		for (; i < x.length; ++i) {
			res += x[i] * w[i];
		}
		if (w.length > i) {
			res += w[i];
		}
		return res;
	}

	/**
	 * Trains with backpropagation, specified number of steps times.
	 * 
	 * @param exampleSet
	 * @param n - steps number
	 * @param eta - learning rate
	 */
	public void trainBackpropagation(List<Example> exampleSet, int n, float eta) {
		for(int step=0; step<n; ++step) {
			float cost = 0;
			for(Example ex: exampleSet) {
				float[] input = ex.getInput();
				// forward propagation of the input
				float[] answer = compute(input);
				float[] correct = ex.getOutput();
				// computing deltas
				// for output layer
				float[] delta = new float[nOutput];
				for (int k = 0; k < nOutput; ++k) {
					delta[k] = answer[k]*(1-answer[k])*(answer[k]-correct[k]);
				}
				// for other layers
	//			float[] c = new float[nOutput];
				for (int k = 0; k < nInput; ++k) {
	//				c[k] = answer[k] *(1-answer[k])*scalar(b, weights[k]);
					// updating weights
					for(int i=0; i<nOutput; ++i) {
						weights[k][i] += -eta*delta[i]*input[k];
					}
				}
				for(int i=0; i<nOutput; ++i) {
					weights[nInput][i] += -eta*delta[i]*1;
				}
				cost += cost(correct, answer);
			}
			// print weights
			System.out.println(Arrays.deepToString(weights) + ", \tError = " + cost);
		}
	}

	/**
	 * Quadratic cost function value computed for a single example.
	 * 
	 * @param correct
	 * @param answer
	 * @return
	 */
	private float cost(float[] correct, float[] answer) {
		float res = 0;
		for (int i=0; i<correct.length; ++i) {
			res += (correct[i] - answer[i])*(correct[i] - answer[i]);
		}
		return res;
	}

}
