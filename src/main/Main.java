package main;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class Main {

	public static void main(String[] args) {
		List<Example> exampleSet = new LinkedList<>();
		exampleSet.add(new Example(new float[]{0, 0}, new float[]{0}));
		exampleSet.add(new Example(new float[]{0, 1}, new float[]{1}));
		exampleSet.add(new Example(new float[]{1, 0}, new float[]{1}));
		exampleSet.add(new Example(new float[]{1, 1}, new float[]{1}));
		
		SimpleNeuralNetwork net = new SimpleNeuralNetwork(2, 1);
		
		for(Example e: exampleSet) {
			System.out.print(Arrays.toString(net.compute(e.getInput())));
		}
		System.out.println();
		
		net.trainBackpropagation(exampleSet, 40, 10.9f);
		for(Example e: exampleSet) {
			System.out.print(Arrays.toString(net.compute(e.getInput())));
		}
		System.out.println();
}
}
