package src;

import java.util.Random;

import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.attributeSelection.*;
import weka.filters.*;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.*;

public class HwMain {
	public static void main(String[] args) throws Exception {
		
		// Options and java attributes
		String pathFile;
		//pathFile = "/home/micah/courses/affective_computing/hw3-2/postureData.arff";
		pathFile = "/Users/theopak/Dropbox/classes/csci-4974_affective-computing/postureData.arff";
		//pathFile = "/home/andrew/affectiveComputing/HW3-2/postureData.arff";
		//int seed = 4; // chosen by fair dice roll, guaranteed to be random.
		int folds = 10; // number of folds in cross-validation

		// Load data set from file using the recommended method for
		// ARFF files in Weka > 3.5.5. For details refer to:
		// http://weka.wikispaces.com/Use+WEKA+in+your+Java+code
		DataSource source = new DataSource(pathFile);
		Instances data = source.getDataSet();
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);

		// Print the data set using the Weka API
		System.out.println(data.toSummaryString());
		System.out.print("\n");

		// Classifier
		String[] tmpOptions = {};//{"-N", "100", "-M", "0.1", "-L", "0.2", "-B", "-H", "70,20,20,20,10"};
		//String[] tmpOptions = {"-H", "70,20,20,20"};
		String classname;
		classname      = "bayes.NaiveBayes";
		//classname      = "functions.MultilayerPerceptron";
		//classname      = "meta.RandomCommittee";
		//classname      = "functions.SMO";
		//classname      = "trees.J48";
		//classname      = "trees.RandomForest";
		Classifier cls = (Classifier) Utils.forName(Classifier.class,
				"weka.classifiers." + classname, tmpOptions);

		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		
		// Randomize data
		Random rand = new Random();
		Instances randData = new Instances(data);
		randData.randomize(rand);
		if (randData.classAttribute().isNominal())
			randData.stratify(folds);

		// Perform cross-validation
		Evaluation eval = new Evaluation(randData);
		for (int n = 0; n < folds; n++) {
			
			// Useful
			System.out.println("=== Fold " + (n + 1) + " of " + folds + " ===");
			
			// Generate features in a training set and a testing set
			Instances train = randData.trainCV(folds, n);
			train = generateFeatures(train);
			if (train.classIndex() == -1)
				train.setClassIndex(train.numAttributes() - 1);
			Instances test = randData.testCV(folds, n);
			test = generateFeatures(test);
			if (test.classIndex() == -1)
				test.setClassIndex(test.numAttributes() - 1);

			AttributeSelection selection = new AttributeSelection();
			CfsSubsetEval subs = new CfsSubsetEval();
			//FilteredSubsetEval subs = new FilteredSubsetEval();
			//ReliefFAttributeEval subs = new ReliefFAttributeEval();
			
			GreedyStepwise search = new GreedyStepwise();
			search.setSearchBackwards(true);
			//Ranker search = new Ranker();
			
			//search.setNumToSelect(0);
			
			//System.out.println("feature selection");
			//System.out.println(Filter.useFilter(train, selection).toSummaryString());
			//System.out.print("\n");
			// the above code is used by the StratifiedRemoveFolds filter, the
			// code below by the Explorer/Experimenter:
			// Instances train = randData.trainCV(folds, n, rand);

			// Build and evaluate classifier
			Classifier clsCopy = Classifier.makeCopy(cls);
			clsCopy.buildClassifier(train);
			eval.evaluateModel(clsCopy, test);
		}

		// Print the results
		System.out.println();
		System.out.println("=== Setup ===");
		System.out.println("Classifier: " + cls.getClass().getName() + " "
				+ Utils.joinOptions(cls.getOptions()));
		System.out.println("Dataset: " + data.relationName());
		System.out.println("Folds: " + folds);
		//System.out.println("Seed: " + seed);
		System.out.println();
		System.out.println(eval.toSummaryString("=== " + folds
				+ "-fold Cross-validation ===", false));
		System.out.println();
		System.out.println("== Confusion ==");
		System.out.println(eval.toMatrixString());
	}

	public static Instances generateFeatures(Instances data) throws Exception {
		SimpleBatchFilter filter;
		Instances new_data = new Instances(data);

		DifferenceGenerator diffGen = new DifferenceGenerator();
		//new_data = diffGen.process(new_data);
		
		filter = new SimilarFilter();
		filter.setInputFormat(new_data);
		// new_data = Filter.useFilter(new_data, filter);
		
		filter = new QuaternionFilter();
		filter.setInputFormat(new_data);
		//new_data = Filter.useFilter(new_data,  filter);
		
		filter = new PropagateFilter();
		filter.setInputFormat(new_data);
		//new_data = Filter.useFilter(new_data,  filter);
		
		Normalize norm = new Normalize();
		norm.setInputFormat(new_data);
		//new_data = Filter.useFilter(new_data, norm);
		
		System.out.println("num_instances");
		System.out.println(new_data.numInstances());
		return new_data;
	}
}
