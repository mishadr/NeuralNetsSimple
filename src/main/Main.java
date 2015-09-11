package main;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import javax.swing.JFrame;

import de.erichseifert.gral.data.DataTable;
import de.erichseifert.gral.plots.XYPlot;
import de.erichseifert.gral.plots.lines.DefaultLineRenderer2D;
import de.erichseifert.gral.plots.lines.LineRenderer;
import de.erichseifert.gral.ui.InteractivePanel;

public class Main {

	public static void main(String[] args) {
		List<Example> exampleSet = new LinkedList<>();
		exampleSet.add(new Example(new double[]{0, 0}, new double[]{1}));
		exampleSet.add(new Example(new double[]{0, 1}, new double[]{0}));
		exampleSet.add(new Example(new double[]{1, 0}, new double[]{0}));
		exampleSet.add(new Example(new double[]{1, 1}, new double[]{0.4}));
		
		SimpleNeuralNetwork net = new SimpleNeuralNetwork(2, 4, 1);
		net.setCostFunction(ICost.CROSS_ENTROPY);
		net.setMomentum(0.95);
		
		for(Example e: exampleSet) {
			System.out.print(Arrays.toString(net.compute(e.getInput())));
		}
		System.out.println();
		
		double[] log = net.trainBackpropagation(exampleSet, 400, 0.1, 0.00);
		for(Example e: exampleSet) {
			System.out.print(Arrays.toString(net.compute(e.getInput())));
		}
		System.out.println();

		// plotting errors during net training
		DataTable data = new DataTable(Double.class, Double.class);
		int len = log.length;
		for (int i = 0; i < len; ++i) {
			data.add(1.0*i, log[i]);
		}
		XYPlot plot = new XYPlot(data);
		LineRenderer lines = new DefaultLineRenderer2D();
		plot.setLineRenderer(data, lines);

		// put the PlotPanel in a JFrame, as a JPanel
		JFrame frame = new JFrame("a plot panel");
		frame.getContentPane().add(new InteractivePanel(plot));
		frame.setSize(800, 600);
		frame.setVisible(true);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

	}
}
