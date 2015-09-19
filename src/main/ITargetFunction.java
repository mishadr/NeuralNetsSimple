package main;

import java.util.HashSet;
import java.util.Set;

/**
 * Function neural net is to learn. It has multiple inputs (n) and multiple
 * outputs (m). Function must be determined on [0;1]^n --> [0;1]^m
 * 
 * @author misha
 *
 */
public interface ITargetFunction {

	/**
	 * Compute result on given inputs.
	 * 
	 * @param input
	 * @return
	 */
	public double[] getResult(double[] input);

	/**
	 * Generate random examples of this function.
	 * 
	 * @param n
	 *            - number of examples if function is not descrete
	 * @return
	 */
	public Set<Example> getExamples(int n);

	/**
	 * Binary AND function. When input is less than 0.5 it is tracted as 0,
	 * otherwise as 1.
	 */
	public static final ITargetFunction AND = new ITargetFunction() {

		@Override
		public double[] getResult(double[] input) {
			return new double[] { input[0] >= 0.5 & input[1] >= 0.5 ? 1.0 : 0.0 };
		}

		@Override
		public Set<Example> getExamples(int n) {
			Set<Example> exampleSet = new HashSet<>();
			exampleSet.add(new Example(new double[] { 0, 0 },
					new double[] { 0 }));
			exampleSet.add(new Example(new double[] { 0, 1 },
					new double[] { 0 }));
			exampleSet.add(new Example(new double[] { 1, 0 },
					new double[] { 0 }));
			exampleSet.add(new Example(new double[] { 1, 1 },
					new double[] { 1 }));
			return exampleSet;
		}
	};

	/**
	 * Square function on [0;1]
	 */
	public static final ITargetFunction SQUARE = new ITargetFunction() {

		@Override
		public double[] getResult(double[] input) {
			return new double[] { input[0] * input[0] };
		}

		@Override
		public Set<Example> getExamples(int n) {
			Set<Example> exampleSet = new HashSet<>();
			for (int i = 0; i < n; ++i) {
				double x = Math.random();
				exampleSet.add(new Example(new double[] { x }, new double[] { x
						* x }));
			}
			return exampleSet;
		}
	};
	
	/**
	 * Half sum of two inputs.
	 */
	public static final ITargetFunction HALF_SUM_2 = new ITargetFunction() {
		
		@Override
		public double[] getResult(double[] input) {
			return new double[]{(input[0] + input[1]) / 2};
		}
		
		@Override
		public Set<Example> getExamples(int n) {
			Set<Example> exampleSet = new HashSet<>();
			for (int i = 0; i < n; ++i) {
				double[] input = new double[] { Math.random(), Math.random() };
				exampleSet.add(new Example(input, getResult(input)));
			}
			return exampleSet;
		}
	};
}
