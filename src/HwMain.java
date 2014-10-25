import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.FastVector;
import weka.core.Attribute;

import weka.classifiers.trees.*;
import weka.classifiers.rules.*;
import weka.classifiers.bayes.*;
import weka.classifiers.functions.*;
import weka.classifiers.lazy.IBk;
import weka.classifiers.Evaluation;

public class HwMain {
	public static void main(String[] args) throws Exception{
		System.out.println("I hate java");
		
		//loading the file
		String pathFile = "/home/micah/courses/affective_computing/hw3-2/postureData.txt";
		pathFile = "/Users/theopak/Dropbox/classes/csci-4974_affective-computing/postureData.csv";
		BufferedReader br = new BufferedReader(new FileReader(pathFile));
		String line;
		int nInstance = 0;
		ArrayList<ArrayList<Double>> data = new ArrayList<ArrayList<Double> >();
		String[] attributes = null;
		while ((line = br.readLine()) != null) {
			if (nInstance == 0) { //The first line is the name of attributes
				attributes = line.split(",");
			} else {
				String[] value = line.split(",");
				ArrayList<Double> temp = new ArrayList<Double>();
				for (int i = 0; i < value.length; i++) {
					temp.add(Double.parseDouble(value[i]));
				}
				data.add(temp);
			}
			nInstance++;
		}
		nInstance--; //This is the number of instance
		int nAttribute = attributes.length;
		br.close();

		//Print out the whole data file.
		System.out.println("Number of attributes: " + nAttribute);
		System.out.println("Number of Instance: " + nInstance);
		for (int i = 0; i < nAttribute; i++) {
			System.out.print(attributes[i] + " ");
		}
		System.out.print("\n");
		for (int i = 0; i < nInstance; i++) {
			for (int j = 0; j < nAttribute; j++) {
				System.out.print((data.get(i)).get(j) + " ");
			}
			System.out.print("\n");
		}
		System.out.print("\n");
		
		/////////////////////////////////////////////////////////////////////////
		/*
		* The last attribute is label which is what we want to predict
		* There are 4 labels - 1 2 3 4
		* 1 = Triumphant
		* 2 = Concentrated
		* 3 = Defeated
		* 4 = Frustrated
		*/
		
		//---Building features in weka format---//
		
		//Declare numeric attributes (weka.core.Attribute)
		Attribute Attribute1 = new Attribute("XpositionHip");
		Attribute Attribute2 = new Attribute("YpositionHip");
		Attribute Attribute3 = new Attribute("ZpositionHip");
		//You should come up with any new feature
		//For example, the size of this vector, sqrt(x^2+y^2+z^2), name it sumHip.
		
		//Declare the class attribute
		//Declare a nominal attribute along with its value 
		//(weka.core.FastVector)
		FastVector fvClassVal = new FastVector(4); //4 different class
		//you can also use the name of emotion
		fvClassVal.addElement("1");
		fvClassVal.addElement("2");
		fvClassVal.addElement("3");
		fvClassVal.addElement("4");
		Attribute ClassAttribute = new Attribute("Emotion", fvClassVal);
		
		//Declare the feature vector
		FastVector fvWekaAttributes = new FastVector(4);
		fvWekaAttributes.addElement(Attribute1);
		fvWekaAttributes.addElement(Attribute2);
		fvWekaAttributes.addElement(Attribute3);
		fvWekaAttributes.addElement(ClassAttribute);
		
		//Create an empty training set (weka.core.Instances)
		//						"name the relation",feature vector,initial set capacity
		Instances train = new Instances("Posture",fvWekaAttributes,nInstance);
		//set class index (the thing that we want to predict)
		//index starts at 0.
		train.setClassIndex(3);
		
		//adding instances to training set
		for(int i=0;i<nInstance;i++){
			//Create the instance (weka.core.Instance)
			Instance temp = new Instance(4);
			//setValue( Attribute, value)
			temp.setValue( (Attribute)fvWekaAttributes.elementAt(0), data.get(i).get(0) );
			temp.setValue( (Attribute)fvWekaAttributes.elementAt(1), data.get(i).get(1) );
			temp.setValue( (Attribute)fvWekaAttributes.elementAt(2), data.get(i).get(2) );
			 
			int classValue = (data.get(i).get(nAttribute-1)).intValue();
			temp.setValue( (Attribute)fvWekaAttributes.elementAt(3), 
					Integer.toString( classValue ) );
			//add the instance
			//classValue when print out is shown as index,not the value
			//System.out.println(temp); 
			train.add(temp);
		}
		
		
		//------Creating a classifier------//
		System.out.println("The result of the classifier\n");
	}
}
