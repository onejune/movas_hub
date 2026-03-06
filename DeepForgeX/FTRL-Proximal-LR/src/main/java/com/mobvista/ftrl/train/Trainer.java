package com.mobvista.ftrl.train;

import java.io.IOException;

import com.mobvista.ftrl.datasource.DataSource;

public interface Trainer {
	public void train(String dataPath,String name) throws IOException;
	public boolean init(String confBase, DataSource dataSource);
}
