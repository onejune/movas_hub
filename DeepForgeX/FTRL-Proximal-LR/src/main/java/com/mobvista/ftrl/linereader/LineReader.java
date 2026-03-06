package com.mobvista.ftrl.linereader;

import java.io.IOException;

public interface LineReader {
	String readLine() throws IOException;
	void close() throws IOException;
}
