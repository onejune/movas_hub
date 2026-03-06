package com.mobvista.ftrl.datasource;

import com.mobvista.ftrl.struct.Sample;

public interface DataSource {
	public Sample extracSample(String line);
	
	public boolean init(String conf);
}
