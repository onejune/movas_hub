package com.mobvista.ftrl.util;

import java.io.File;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

import com.mobvista.ftrl.struct.ModelConfig;
import com.mobvista.ftrl.struct.Parameter;

public class ToolFucs {
	public static boolean confirmDir(String path){
		File file = new File(path);
		if (file.exists()) {
			if (file.isDirectory()) {
				return true;
			} else {
				System.out.println("output is exist,but not a directory.");
				return false;
			}
		} else {
			if (file.mkdir()) {
				return true;
			} else {
				System.out.println("Can not mkdir " + path);
				return false;
			}
		}
	}
	
	public static ModelConfig singleParaConfig(){
		ModelConfig conf = new ModelConfig();
		conf.names.add("default");
		conf.paras.add(new Parameter());
		return conf;
	}

    public static void PrintCurrentTime() {
        // 获取当前日期时间
        LocalDateTime now = LocalDateTime.now();

        // 定义格式化模式：年-月-日 时:分:秒
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

        // 格式化当前时间
        String formattedDateTime = now.format(formatter);

        // 打印当前时间
        System.out.println("当前时间：" + formattedDateTime);
    }
}
