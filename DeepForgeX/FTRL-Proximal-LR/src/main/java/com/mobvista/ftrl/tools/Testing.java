package com.mobvista.ftrl.tools;

import org.apache.commons.lang3.StringUtils;

public class Testing {
	public static void main(String[] args) {
		String line="abc\t\t\002\002\002";
		System.out.println('|'+line+'|');
		System.out.println('|'+line.trim()+'|');
	}
}
