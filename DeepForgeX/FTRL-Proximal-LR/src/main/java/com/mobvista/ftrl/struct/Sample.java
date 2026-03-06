package com.mobvista.ftrl.struct;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.mobvista.ftrl.util.Constants;

public class Sample {
	public List<String> strFeatures = new ArrayList<String>(Constants.FEATURE_LEN);
	public FeatureInfo[] featureInfos;
	// public int label;
    public double label; //修改：适配回归和分类
	public String key;
	public String name;
	public double[] sqrtN;
	public double[] n;
    public String demand_pkgname;
	//支持连续值特征
	public Map<String, Double> continuousFeatures = new HashMap<String, Double>();
}
