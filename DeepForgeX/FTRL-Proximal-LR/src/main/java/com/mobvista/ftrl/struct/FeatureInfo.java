package com.mobvista.ftrl.struct;

public class FeatureInfo {
	volatile public double[] omiga;
	volatile public double z[];
	volatile public double[] n;      // 新增：累计梯度平方（用于回归）
	volatile public long positive = 0;
	volatile public long negtive = 0;
    volatile public double label_sum = 0.0;
	volatile public long sample_cnt = 0;
	volatile public double feature_value = 1.0;

	public FeatureInfo(int len){
		omiga = new double[len];
		z = new double[len];
		n = new double[len]; // 新增：累计梯度平方（用于回归）
	}
}
