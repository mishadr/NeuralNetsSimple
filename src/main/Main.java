package main;

import java.util.Arrays;
import java.util.Set;

import javax.swing.JFrame;

import de.erichseifert.gral.data.DataTable;
import de.erichseifert.gral.plots.XYPlot;
import de.erichseifert.gral.plots.lines.DefaultLineRenderer2D;
import de.erichseifert.gral.plots.lines.LineRenderer;
import de.erichseifert.gral.ui.InteractivePanel;

public class Main {

	public static void main(String[] args) {
		ITargetFunction target = ITargetFunction.HALF_SUM_2;
		Set<Example> trainingSet = target.getExamples(50);
		Set<Example> testSet = target.getExamples(10);
		
		new HandWrittenDigitsRecognizer();
		
//		SimpleNeuralNetwork net = new SimpleNeuralNetwork(2, 512, 512, 1);
//		net.setCostFunction(ICost.CROSS_ENTROPY);
//		net.setMomentum(0.9);
//		
//		double trainError = net.computeError(trainingSet, ICost.QUADRATIC);
//		double testError = net.computeError(testSet, ICost.QUADRATIC);
//		System.out.printf("Train error: %f\tTest error: %f\n", trainError, testError);
//
//		for(Example e: trainingSet) {
//			System.out.print(Arrays.toString(net.compute(e.getInput())));
//		}
//		System.out.println();
//		
//		double[] log = net.trainBackpropagation(trainingSet, 50, 0.1, 0.0);
//		for(Example e: testSet) {
//			String input = Arrays.toString(e.getInput());
//			String output = Arrays.toString(net.compute(e.getInput()));
//			String correct = Arrays.toString(e.getOutput());
//			System.out.printf("Input: %s\tCorrect: %s\tOutput: %s\n", input, correct, output);
//		}
//		System.out.println();
//		
//		trainError = net.computeError(trainingSet, ICost.QUADRATIC);
//		testError = net.computeError(testSet, ICost.QUADRATIC);
//		System.out.printf("Train error: %G\tTest error: %g\n", trainError, testError);
//
//		// plotting errors during net training
//		DataTable data = new DataTable(Double.class, Double.class);
//		int len = log.length;
//		for (int i = 0; i < len; ++i) {
//			data.add(1.0*i, log[i]);
//		}
//		XYPlot plot = new XYPlot(data);
//		LineRenderer lines = new DefaultLineRenderer2D();
//		plot.setLineRenderer(data, lines);
//
//		// put the PlotPanel in a JFrame, as a JPanel
//		JFrame frame = new JFrame("a plot panel");
//		frame.getContentPane().add(new InteractivePanel(plot));
//		frame.setSize(800, 600);
//		frame.setVisible(true);
//		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

	}
}
