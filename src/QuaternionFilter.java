import weka.core.Instance;
import weka.core.Instances;
import weka.core.FastVector;
import weka.core.Attribute;
import weka.filters.*;
import weka.core.Capabilities.*;
import weka.core.*;

public class QuaternionFilter extends SimpleBatchFilter{
	public String globalInfo(){
		return "euler to quaternion";
	}
	
	public Capabilities getCapabilities(){
		Capabilities result = super.getCapabilities();
		result.enableAllAttributes();
		result.enableAllClasses();
		result.enable(Capability.NO_CLASS);
		return result;
	}
	
	protected Instances determineOutputFormat(Instances inputFormat) {
		FastVector list = new FastVector();
		for(int i=0; i < 3; ++i){
			list.addElement(inputFormat.attribute(i));
		}
		int done = -1;
		for(int i=1; i < (inputFormat.numAttributes()-1)/3; ++i){
			String name = inputFormat.attribute((i*3)).name();
			name = name.substring(1);
			if(done == -1){
				if(name.contains("_")){
					done = 3*i;
					break;
				} else {
					Attribute a = new Attribute("a"+name);
					Attribute b = new Attribute("b"+name);
					Attribute c = new Attribute("c"+name);
					Attribute d = new Attribute("d"+name);
					list.addElement(a);
					list.addElement(b);
					list.addElement(c);
					list.addElement(d);
				}
			}
		}
		
		if(done>0){
			for (int i=done; i<inputFormat.numAttributes(); ++i){
				list.addElement(inputFormat.attribute(i));
			}
		}
		
		list.addElement(inputFormat.attribute(inputFormat.numAttributes()-1));
		System.out.println(list.size());
		//System.out.println(list.capacity()-inputFormat.numAttributes());
		Instances result = new Instances("quat", list, 0);
		return result;
    }
 
	protected Instances process(Instances inst) {
		Instances result = new Instances(determineOutputFormat(inst), 0);
		for (int i = 0; i < inst.numInstances(); i++) {
			double[] values = new double[result.numAttributes()];
			for (int n = 0; n < 3; n++)
				values[n] = inst.instance(i).value(n);
			for (int n = 1; n < (inst.instance(i).numAttributes()-1)/3; ++n){
				double x = inst.instance(i).value(n*3);
				double y = inst.instance(i).value(n*3+1);
				double z = inst.instance(i).value(n*3+2);
				double[] arr = {z,y,x};
				double[] quat = Quaternion.toQuat(arr);
				int sign = 1;
				double max = 0;
				for(int j=0; j < 4; ++j){
					if(Math.abs(quat[j]) > max){
						//sign = (quat[j]>0 ? 1 : -1);
						max = Math.abs(quat[j]);
						break;
					}
				}
				values[3+(n-1)*4] = sign*quat[0];
				values[3+(n-1)*4+1] = sign*quat[1];
				values[3+(n-1)*4+2] = sign*quat[2];
				values[3+(n-1)*4+3] = sign*quat[3];
			}
			values[values.length-1] = inst.instance(i).value(inst.instance(i).numAttributes()-1);
			result.add(new Instance(1, values));
			//result.add(new Instance(1, Quaternion.negate(values)));
		}
		if (result.classIndex() == -1)
			result.setClassIndex(result.numAttributes() - 1);
		return result;
	}
	

	public static void main(String[] args) {
		//runFilter(new QuaternionFilter(), args);
		double[] euler = {30.0,60.0,45.0};
		double[] quat = Quaternion.toQuat(euler);	
		for (int i=0;i<quat.length;++i){
			//System.out.println(quat[i]);
		}
	}
}

