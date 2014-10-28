import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class DifferenceGenerator {
	
	public Instances process(Instances inst) {
		Instances result = new Instances(inst);
		Attribute diffHip = new Attribute("_diffHipZ");
		Attribute diffShoulder = new Attribute("_diffShoulderZ");
		result.insertAttributeAt(diffHip, result.numAttributes()-1);
		result.insertAttributeAt(diffShoulder, result.numAttributes()-1);
		
		for (int i = 0; i < inst.numInstances(); i++) {
			Instance row = new Instance(result.instance(0));
			
			double zRotationLeftHip = result.instance(0).value(6);
			double zRotationRightHip = result.instance(0).value(15);
			double zRotationRightShoulder = result.instance(0).value(42);
			double zRotationLeftShoulder = result.instance(0).value(30);
			
			row.setValue(result.numAttributes()-2, zRotationLeftHip - zRotationRightHip);
			row.setValue(result.numAttributes()-3, zRotationRightShoulder - zRotationLeftShoulder);
			
			//replace the old row with the new row
			result.add(row);
			result.delete(0);
		}
		
		return result;
	}
	
	
}
