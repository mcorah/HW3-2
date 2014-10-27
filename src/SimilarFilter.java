package src;

import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.*;
import weka.core.Capabilities.*;

public class SimilarFilter extends SimpleBatchFilter {
	public String globalInfo() {
		return "similarize angles to quaternion";
	}

	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.enableAllAttributes();
		result.enableAllClasses();
		result.enable(Capability.NO_CLASS);
		return result;
	}

	protected Instances determineOutputFormat(Instances inputFormat) {
		Instances result = new Instances(inputFormat, 0);
		return result;
	}

	protected Instances process(Instances inst) {
		Instances result = new Instances(inst);
		int[] flip = { 0, 24, 26, 51, 53, 54, 56 };
		int[][] lr = { { 7, 16 }, { 10, 19 }, { 13, 22 }, { 28, 40 },
				{ 31, 43 }, { 34, 46 }, { 37, 49 } };
		for (int i = 0; i < inst.numInstances(); i++) {
			Instance row = new Instance(inst.instance(i));
			for (int j = 0; j < flip.length; ++j) {
				row.setValue(flip[j], -1 * inst.instance(i).value(flip[j]));
			}
			for (int j = 0; j < lr.length; ++j) {
				for (int k = 0; k < 3; ++k) {
					double temp = row.value(lr[j][0]);
					int sign = (k == 1 ? 1 : -1);
					row.setValue(lr[j][0] + k, sign * row.value(lr[j][1] + k));
					row.setValue(lr[j][1] + k, sign * temp);
				}
			}
			result.add(row);
		}
		if (result.classIndex() == -1)
			result.setClassIndex(result.numAttributes() - 1);
		return result;
	}
}
