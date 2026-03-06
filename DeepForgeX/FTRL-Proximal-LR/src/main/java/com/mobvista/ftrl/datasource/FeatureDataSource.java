package com.mobvista.ftrl.datasource;

import java.util.Arrays;

import org.apache.commons.lang3.StringUtils;

import com.mobvista.ftrl.datasource.DataSource;
import com.mobvista.ftrl.struct.Sample;

public class FeatureDataSource implements DataSource {

	public Sample extracSample(String line) {
		String[] segs = StringUtils.split(line.trim(), '\002');
		Sample sample = new Sample();
		//原先
		sample.label = segs[0].equals("1") ? 1 : 0;


		sample.key = "all";
		for (String f:StringUtils.split(segs[2], '\003')){
			if(f.indexOf("direct=")>=0 ||f.indexOf("detect_type=")>=0 ||f.indexOf("package_size=")>=0){
				continue;
			}
			sample.strFeatures.add(f);
		}
		//sample.strFeatures.addAll(Arrays.asList(StringUtils.split(segs[2], '\003')));
		sample.name=segs[1];
		return sample;
	}

	public boolean init(String conf) {
		// TODO Auto-generated method stub
		return true;
	}

}
