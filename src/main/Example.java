package main;

public class Example {

	private final float[] in;
	private final float[] out;
	
	public Example(float[] in, float[] out) {
		this.in = in;
		this.out = out;
	}
	
	public float[] getInput() {
		return in;
	}
	
	public float[] getOutput() {
		return out;
	}
}
