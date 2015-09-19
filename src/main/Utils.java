package main;

import java.io.FileWriter;
import java.io.IOException;

import javax.swing.JFrame;

import de.erichseifert.gral.data.DataTable;
import de.erichseifert.gral.plots.XYPlot;
import de.erichseifert.gral.plots.lines.DefaultLineRenderer2D;
import de.erichseifert.gral.plots.lines.LineRenderer;
import de.erichseifert.gral.ui.InteractivePanel;

public class Utils {

	/**
	 * argmax function for an array of doubles.
	 * 
	 * @param array
	 * @return
	 */
	public static int argmax(double[] array) {
		int res = -1;
		double max = array[0];
		for (int i = 1; i < array.length; ++i) {
			if (array[i] > max) {
				max = array[i];
				res = i;
			}
		}
		return res;
	}

	/**
	 * Activation function (sigma).
	 * 
	 * @param x
	 * @return
	 */
	public static double sigma(double x) {
		return 1.0 / (1.0 + Math.exp(-x));
	}

	/**
	 * Write to a file with results given string.
	 * 
	 * @param content
	 */
	public static void saveResult(String content, Object ... args) {
		String filename = "results.txt";
		try (FileWriter fw = new FileWriter(filename, true);) {
			fw.write(String.format(content, args));
			fw.write("\n");
		} catch (IOException ioe) {
			ioe.printStackTrace();
		}
	}
	
	/**
	 * Make a 2d plot.
	 * 
	 * @param log 
	 * 
	 */
	public static void plotResults(double[] log) {
		DataTable data = new DataTable(Double.class, Double.class);
		int len = log.length;
		for (int i = 0; i < len; ++i) {
			data.add(1.0 * i, log[i]);
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

	/**
	 * Compute maximum over an array.
	 * 
	 * @param array
	 * @return
	 */
	public static double max(double[] array) {
		double max = array[0];
		for (int i = 1; i < array.length; ++i) {
			if (array[i] > max) {
				max = array[i];
			}
		}
		return max;
	}

}
