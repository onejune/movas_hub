package com.mobvista.ftrl.train;

import java.io.IOException;

import com.mobvista.ftrl.linereader.LineReader;
import com.mobvista.ftrl.linereader.S3LineReader;


public class S3ParaTrainer extends LocalParaTrainer {
	
	public LineReader getLineReader(String path) throws IOException{
		return new S3LineReader(path);
	}
	
}
