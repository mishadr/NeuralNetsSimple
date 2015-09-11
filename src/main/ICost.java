package main;

import static java.lang.Math.log;

/**
 * Provides computing of cost function and its derivative.
 * 
 * @author misha
 *
 */
public interface ICost {

	/**
	 * Computes error cost given neural net output and correct output vectors.
	 * 
	 * @param correct
	 *            - expected outputs
	 * @param output
	 *            - neural net's output
	 * @return
	 */
	public double cost(double[] correct, double[] output);

	/**
	 * Computes vector of cost function dirivatives by (vector of) outputs of.
	 * 
	 * @param correct
	 *            - expected outputs
	 * @param output
	 *            - neural net's output
	 * @return
	 */
	public double[] derivative(double[] correct, double[] output);

	/**
	 * Quadratic error function.
	 */
	public static final ICost QUADRATIC = new ICost() {

		/**
		 * @return half of L2 distance between given vectors
		 */
		@Override
		public double cost(double[] correct, double[] output) {
			double res = 0;
			for (int i = 0; i < correct.length; ++i) {
				res += (correct[i] - output[i]) * (correct[i] - output[i]);
			}
			return res / 2;
		}

		@Override
		public double[] derivative(double[] correct, double[] output) {
			double[] res = new double[correct.length];
			for (int i = 0; i < correct.length; ++i) {
				res[i] = (output[i] - correct[i]);
			}
			return res;
		}
	};

	/**
	 * Cross-entropy error function with natural logarithms. NOTE: when correct
	 * net's outputs are not only 0's and 1's this error does not converge to
	 * zero. Use lower learning rate than that with square error.
	 */
	public static final ICost CROSS_ENTROPY = new ICost() {

		@Override
		public double cost(double[] correct, double[] output) {
			double res = 0;
			for (int i = 0; i < correct.length; ++i) {
				res -= correct[i] * log(output[i]) + (1 - correct[i])
						* log(1 - output[i]);
			}
			return res;
		}

		@Override
		public double[] derivative(double[] correct, double[] output) {
			double[] res = new double[correct.length];
			for (int i = 0; i < correct.length; ++i) {
				res[i] = (output[i] - correct[i]) / output[i] / (1 - output[i]);
			}
			return res;
		}
	};
}
