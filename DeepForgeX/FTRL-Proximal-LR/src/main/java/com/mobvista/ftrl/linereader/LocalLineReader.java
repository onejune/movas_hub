package com.mobvista.ftrl.linereader;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class LocalLineReader implements LineReader {
	BufferedReader reader;
	public LocalLineReader(String localPath) throws IOException{
		reader = new BufferedReader(new FileReader(localPath));
	}
	public String readLine() throws IOException {
		try {
            String line = reader.readLine();
            return line;
        } catch (IOException e) {
            System.err.println("Error occurred while reading line: " + e.getMessage());
            throw e;
        }
	}
	public void close() throws IOException {
		reader.close();
		
	}
}
