import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.Capabilities.Capability;
import weka.filters.*;
import weka.core.Instance;

public class PropagateFilter extends SimpleBatchFilter {
	public String globalInfo() {
		return "quaternion propagation";
	}

	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.enableAllAttributes();
		result.enableAllClasses();
		result.enable(Capability.NO_CLASS);
		return result;
	}

	protected Instances determineOutputFormat(Instances inputFormat) {
		Instances result = new Instances(inputFormat);
		for (int i = 0; i < angle_propagation.length; ++i) {
			result = Quaternion.insertQuat(result, angle_propagation[i][angle_propagation[i].length - 1] + "Prop");
		}
		return result;
	}

	protected Instances process(Instances inst) {
		Instances result = new Instances(determineOutputFormat(inst));
		for (int i = 0; i < inst.numInstances(); ++i) {
			for (int j = 0; j < angle_propagation.length; ++j) {
				double[] quat = Quaternion.getQuat(inst.instance(i), angle_propagation[j][0]);
				for (int k = 1; k < angle_propagation[j].length; ++k) {
					quat = Quaternion.quatMul(quat, Quaternion.getQuat(inst.instance(i), angle_propagation[j][k]));
				}
				//double[] vals = inst.instance(i).toDoubleArray();
				//Instance new_inst = new Instance(1,vals);
				//result.add(new_inst);
				String name = angle_propagation[j][angle_propagation[j].length - 1] + "Prop";
				Quaternion.insertQuat(inst, name);
				Quaternion.setQuat(quat, result.instance(i), name);
			}
		}
		if (result.classIndex() == -1) {
			result.setClassIndex(result.numAttributes() - 1);
		}
		return result;
	}

	public static void main(String[] args) {
		runFilter(new PropagateFilter(), args);
		// for (int i=0;i<quat.length;++i){
		// System.out.println(quat[i]);
		// }
	}

	private static String[][] angle_propagation = {
			{ "rotationLeftHip", "rotationLeftKnee", "rotationLeftAnkle" },
			{ "rotationRightHip", "rotationRightKnee", "rotationRightAnkle" },
			{ "rotationChest", "rotationLeftCollar", "rotationLeftShoulder","rotationLeftElbow", "rotationLeftWrist" },
			{ "rotationChest", "rotationRightCollar", "rotationRightShoulder","rotationRightElbow", "rotationRightWrist" },
			{ "rotationChest", "rotationNeck", "rotationHead" },
	};
}
