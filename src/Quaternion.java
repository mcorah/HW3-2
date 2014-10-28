package src;

import weka.core.Instances;
import weka.core.Instance;
import weka.core.Attribute;

public class Quaternion {
	public static double[] negate(double[] in) {
		double[] result = new double[in.length];
		for (int i = 0; i < in.length; ++i) {
			result[i] = (i > 2 && i < in.length - 1 ? -1 : 1) * in[i];
		}
		return result;
	}

	public static double[] toQuat(double[] euler) {
		// zxy
		// assuming extrinsic rotations
		double[] Z = { 0, 0, Math.sin(euler[0] / 2.0 * Math.PI / 180.0),
				Math.cos(euler[0] / 2.0 * Math.PI / 180.0) };
		double[] X = { Math.sin(euler[1] / 2 * Math.PI / 180.0), 0, 0,
				Math.cos(euler[1] / 2 * Math.PI / 180.0) };
		double[] Y = { 0, Math.sin(euler[2] / 2 * Math.PI / 180.0), 0,
				Math.cos(euler[2] / 2 * Math.PI / 180.0) };
		// printQuat(Z);
		// printQuat(X);
		// printQuat(Y);
		return quatMul(quatMul(Y, X), Z);
	}

	public static Instances insertQuat(Instances inst, String name) {
		Instances ret = new Instances(inst);
		for (int j = 0; j < letters.length; ++j) {
			Attribute att = new Attribute(letters[j] + name);
			int found = 0;
			String n = letters[j] + name;
			for (int k = 0; k < inst.numAttributes(); ++k) {
				if (n.equals(inst.attribute(k).name())) {
					found = 1;
					break;
				}
			}
			if (found == 0) {
				int count = ret.numAttributes();
				ret.insertAttributeAt(att, ret.numAttributes() - 1);
				assert (count - ret.numAttributes() == 1);
			}
		}
		return ret;
	}

	public static Attribute byName(Instance inst, String name) {
		for (int i = 0; i < inst.numAttributes(); ++i) {
			if (name.equals(inst.attribute(i).name())) {
				return inst.attribute(i);
			}
		}
		System.out.print(name);
		System.out.println(" not found");
		return new Attribute("");
	}

	public static double[] getQuat(Instance inst, String name) {
		double[] ret = new double[4];
		for (int i = 0; i < letters.length; ++i) {
			ret[i] = inst.value(byName(inst, letters[i] + name));
		}
		return ret;
	}

	public static void setQuat(double[] quat, Instance inst, String name) {
		for (int i = 0; i < letters.length; ++i) {
			inst.setValue(byName(inst, letters[i] + name), quat[i]);
		}
	}

	public static double[] quatMul(double[] a, double[] b) {
		double[] result = new double[4];
		result[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
		result[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
		result[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
		result[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
		return result;
	}

	public static void printQuat(double[] quat) {
		System.out.println("quat");
		for (int i = 0; i < quat.length; ++i) {
			System.out.println(quat[i]);
		}
	}

	public static String[] letters = { "a", "b", "c", "d" };
}
