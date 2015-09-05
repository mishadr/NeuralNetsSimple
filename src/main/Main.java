package main;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class Main {

	public static void main(String[] args) {
		List<Example> exampleSet = new LinkedList<>();
		exampleSet.add(new Example(new double[]{0, 0}, new double[]{1}));
		exampleSet.add(new Example(new double[]{0, 1}, new double[]{0}));
		exampleSet.add(new Example(new double[]{1, 0}, new double[]{0}));
		exampleSet.add(new Example(new double[]{1, 1}, new double[]{0}));
		
		SimpleNeuralNetwork net = new SimpleNeuralNetwork(2, 1);
		
		for(Example e: exampleSet) {
			System.out.print(Arrays.toString(net.compute(e.getInput())));
		}
		System.out.println();
		
		net.trainBackpropagation(exampleSet, 100, 10.9f);
		for(Example e: exampleSet) {
			System.out.print(Arrays.toString(net.compute(e.getInput())));
		}
		System.out.println();
}
}
