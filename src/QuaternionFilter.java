import weka.core.Instance;
import weka.core.Instances;
import weka.core.FastVector;
import weka.core.Attribute;
import weka.filters.*;
import weka.core.Capabilities.*;
import weka.core.*;
import java.lang.Math.*;

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
		for(int i=1; i < (inputFormat.numAttributes()-1)/3; ++i){
			String name = inputFormat.attribute((i*3)+3).name();
			Attribute a = new Attribute(name + "a");
			Attribute b = new Attribute(name + "b");
			Attribute c = new Attribute(name + "c");
			Attribute d = new Attribute(name + "d");
			list.addElement(a);
			list.addElement(b);
			list.addElement(c);
			list.addElement(d);
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
				double[] quat = toQuat(arr);
				values[3+(n-1)*4] = quat[0];
				values[3+(n-1)*4+1] = quat[1];
				values[3+(n-1)*4+2] = quat[2];
				values[3+(n-1)*4+3] = quat[3];
			}
			values[values.length-1] = inst.instance(i).value(inst.instance(i).numAttributes()-1);
			result.add(new Instance(1, values));
			result.add(new Instance(1, negate(values)));
		}
		if (result.classIndex() == -1)
			result.setClassIndex(result.numAttributes() - 1);
		return result;
	}
	
	public static double[] negate(double[] in){
		double[] result = new double[in.length];
		for (int i = 0; i < in.length; ++i){
			result[i] = (i>2 && i<in.length-1 ? -1 : 1) * in[i];
		}
		return result;
	}
	
	public static double[] toQuat(double[] euler){
		//zxy
		//double[] quat = {euler[0],euler[1],euler[2],euler[0]};
		//double[] Z = {0, 0, Math.sin(euler[0]/2.0), Math.cos(euler[0]/2.0)};
		//double[] X = {Math.sin(euler[1]/2), 0, 0, Math.cos(euler[1]/2)};
		//double[] Y = {0, Math.sin(euler[2]/2), 0, Math.cos(euler[2]/2)};
		double[] Z = {0, 0, Math.sin(euler[0]/2.0*Math.PI/180.0), Math.cos(euler[0]/2.0*Math.PI/180.0)};
		double[] X = {Math.sin(euler[1]/2*Math.PI/180.0), 0, 0, Math.cos(euler[1]/2*Math.PI/180.0)};
		double[] Y = {0, Math.sin(euler[2]/2*Math.PI/180.0), 0, Math.cos(euler[2]/2*Math.PI/180.0)};
		printQuat(Z);
		printQuat(X);
		printQuat(Y);
		return quatMul(quatMul(Z,X),Y);
	}
	
	public static double[] quatMul(double[] a, double[] b){
		double[] result = new double[4];
		result[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
		result[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
		result[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
		result[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
		return result;
	}
	public static void printQuat(double[] quat){
		System.out.println("quat");
		for (int i=0;i<quat.length;++i){
			System.out.println(quat[i]);
		}
	}
	public static void main(String[] args) {
		//runFilter(new QuaternionFilter(), args);
		double[] euler = {30.0,60.0,45.0};
		double[] quat = toQuat(euler);	
		for (int i=0;i<quat.length;++i){
			System.out.println(quat[i]);
		}
	}
}

