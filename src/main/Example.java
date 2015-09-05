package main;

public class Example {

	private final double[] in;
	private final double[] out;
	
	public Example(double[] in, double[] out) {
		this.in = in;
		this.out = out;
	}
	
	public double[] getInput() {
		return in;
	}
	
	public double[] getOutput() {
		return out;
	}
}
