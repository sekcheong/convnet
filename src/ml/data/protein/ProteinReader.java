package ml.data.protein;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import ml.convnet.Volume;
import ml.data.DataSet;
import ml.data.Example;
import ml.io.DataReader;
import ml.utils.Console;

public class ProteinReader extends DataReader {

	private List<Amino> _paddings;
	private int _windowSize = 17;
	private int _size = 0;

	private BufferedReader _reader;


	public ProteinReader(String fileName) {
		try {
			FileReader freader = new FileReader(fileName);
			_reader = new BufferedReader(freader);
		}
		catch (Exception ex) {

		}

	}


	private List<List<Amino>> readProteins(BufferedReader reader) throws Exception {

		List<List<Amino>> proteins = new ArrayList<List<Amino>>();
		List<Amino> protein = null;

		if (reader == null) return proteins;
		
		// creates the padding vector
		int paddingCount = _windowSize / 2;
		_paddings = new ArrayList<Amino>();
		for (int i = 0; i < paddingCount; i++) {
			_paddings.add(Amino.padding);
		}

		int lineNo = 0;

		while (true) {

			String line = reader.readLine();
			lineNo++;

			if (line == null) {
				if (protein != null) {
					proteins.add(protein);
				}
				break;
			}

			line = line.trim();

			// skip comment line
			if (line.startsWith("#") || line.startsWith("//")) {
				continue;
			}

			// starts of a new protein sequence
			if (line.startsWith("<>")) {
				if (protein != null) {
					proteins.add(protein);
				}
				protein = new ArrayList<Amino>();
				continue;
			}

			// skip the end of protein mark
			if (line.toUpperCase().startsWith("END") || line.toUpperCase().startsWith("<END") || line.toUpperCase().startsWith("<END>")) {
				continue;
			}

			// parse the primary and secondary structure line
			if (line.length() > 0) {
				try {
					Amino amino = new Amino(line);
					protein.add(amino);
				}
				catch (Exception ex) {
					Console.writeLine("Unable to process sequence at line ", lineNo, ":", line);
				}
			}
		}

		int count = 0;
		for (List<Amino> p : proteins) {
			count += p.size();
		}

		return proteins;
	}


	private Example createInstance(Amino[] window) {

		int cols = Amino.primaryLabelCount();
		double[] features = new double[window.length * cols];
		double[] target = new double[Amino.secondaryLabelCount()];
		Amino center = window[window.length / 2];

		for (int i = 0; i < window.length; i++) {
			int[] v = window[i].primaryOneHot();
			for (int j = 0; j < cols; j++) {
				features[i * cols + j] = v[j];
			}
		}

		for (int i = 0; i < target.length; i++) {
			target[i] = center.secondaryOneHot()[i];
		}

		Example inst = new Example(new Volume(features), new Volume(target));

		return inst;
	}


	private List<Example> createInstances(List<List<Amino>> proteins) {
		Amino[] window = new Amino[_windowSize];
		List<Example> insts = new ArrayList<Example>();

		for (List<Amino> p : proteins) {
			List<Amino> m = new ArrayList<Amino>();
			int size = p.size();
			m.addAll(_paddings);
			m.addAll(p);
			m.addAll(_paddings);
			for (int i = 0; i < size; i++) {
				for (int j = 0; j < _windowSize; j++) {
					window[j] = m.get(i + j);
				}
				Example inst = createInstance(window);
				insts.add(inst);
			}
		}
		return insts;
	}


	private Example[] convertToExamples(List<List<Amino>> proteins) {
		Example[] ret;
		List<Example> insts = createInstances(proteins);
		ret = insts.toArray(new Example[insts.size()]);
		return ret;
	}


	private DataSet[] splitDataSet(List<List<Amino>> proteins) {
		List<List<Amino>> tune = new ArrayList<List<Amino>>();
		List<List<Amino>> test = new ArrayList<List<Amino>>();
		List<List<Amino>> train = new ArrayList<List<Amino>>();

		DataSet[] ret = new DataSet[3];

		int i = 1;
		for (List<Amino> p : proteins) {
			if ((i % 5) == 0) {
				tune.add(p);
			}
			else if ((i % 5) == 1 && (i > 5)) {
				test.add(p);
			}
			else {
				train.add(p);
			}
			i++;
		}

		try {
			ret[0] = new DataSet(convertToExamples(train));
			ret[1] = new DataSet(convertToExamples(tune));
			ret[2] = new DataSet(convertToExamples(test));
		}
		catch (Exception ex) {
			ex.printStackTrace();
		}
		return ret;
	}


	@Override
	public DataSet readDataSet() {
		try {
			List<List<Amino>> proteins = readProteins(_reader);
			DataSet[] ds = splitDataSet(proteins);
			return new ProteinDataSet(ds[0].examples(), ds[1].examples(), ds[2].examples());
		}
		catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}

}
