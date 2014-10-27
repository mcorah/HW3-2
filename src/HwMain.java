import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.FastVector;
import weka.core.Attribute;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.*;
import weka.classifiers.rules.*;
import weka.classifiers.bayes.*;
import weka.classifiers.functions.*;
import weka.classifiers.lazy.IBk;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.filters.Filter;

public class HwMain {
	public static void main(String[] args) throws Exception{
		// Options and java attributes
		String pathFile = "/home/micah/courses/affective_computing/hw3-2/postureData.txt";
		pathFile = "/Users/theopak/Dropbox/classes/csci-4974_affective-computing/postureData.arff";
		int seed  = 4;		// chosen by fair dice roll, guaranteed to be random.
		int folds = 10;		// number of folds in cross-validation
		
		// Load data set from file using the recommended method for 
		//   ARFF files in Weka > 3.5.5. For details refer to:
		//   http://weka.wikispaces.com/Use+WEKA+in+your+Java+code
		DataSource source = new DataSource(pathFile);
		Instances data = source.getDataSet();
		if (data.classIndex() == -1)
		   data.setClassIndex(data.numAttributes() - 1);
		
		// Print the data set USING THE WEKA API THAT ALREADY DOES THE WORK
		//  FOR YOU WHY WOULD YOU DO IT ANY OTHER WAY !?
		System.out.println(data.toSummaryString());
		System.out.print("\n");
		
		
		//------Creating a classifier------//
		System.out.println("Classifying and validating using 10-fold CV\n");

		// Classifier
		String[] tmpOptions;
		String classname;
		tmpOptions     = Utils.splitOptions(Utils.getOption("W", args));
		classname      = "functions.MultilayerPerceptron";// tmpOptions[0];
		//tmpOptions[0]  = "";
		Classifier cls = (Classifier) Utils.forName(Classifier.class, "weka.classifiers." + classname, tmpOptions);

		Instances data_features;
		data_features = generateFeatures(data);
		if (data_features.classIndex() == -1)
			   data_features.setClassIndex(data_features.numAttributes() - 1);
		// Randomize data
		Random rand = new Random(seed);
		Instances randData = new Instances(data_features);
		randData.randomize(rand);
		if (randData.classAttribute().isNominal())
			randData.stratify(folds);

		// Perform cross-validation
		Evaluation eval = new Evaluation(randData);
		for (int n = 0; n < folds; n++) {
			Instances train = randData.trainCV(folds, n);
			Instances test = randData.testCV(folds, n);
			// the above code is used by the StratifiedRemoveFolds filter, the
			// code below by the Explorer/Experimenter:
			// Instances train = randData.trainCV(folds, n, rand);

			// build and evaluate classifier
			Classifier clsCopy = Classifier.makeCopy(cls);
			clsCopy.buildClassifier(train);
			eval.evaluateModel(clsCopy, test);
		}

		// output evaluation
		System.out.println();
		System.out.println("=== Setup ===");
		System.out.println("Classifier: " + cls.getClass().getName() + " " + Utils.joinOptions(cls.getOptions()));
		System.out.println("Dataset: " + data.relationName());
		System.out.println("Folds: " + folds);
		System.out.println("Seed: " + seed);
		System.out.println();
		System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation ===", false));
		System.out.println();
		System.out.println("== Confusion ==");
		System.out.println(eval.toMatrixString());
	}
	
	public static Instances generateFeatures(Instances data) throws Exception{
		SimpleBatchFilter filter = new QuaternionFilter();
		filter.setInputFormat(data);
		Instances new_data = Filter.useFilter(data,  filter);
		System.out.println("num_instances");
		System.out.println(new_data.numInstances());
		return new_data;
	}
}
