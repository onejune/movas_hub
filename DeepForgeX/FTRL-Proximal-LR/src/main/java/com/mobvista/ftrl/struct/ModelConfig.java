package com.mobvista.ftrl.struct;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.StringUtils;

import com.fasterxml.jackson.databind.ObjectMapper;

public class ModelConfig {
	public List<String> names = new ArrayList<String>();
	public List<Parameter> paras = new ArrayList<Parameter>();
    // 新增 map 用于存储每个 pkg_name 的 Parameter 参数
    private Map<String, Parameter> pkgParamsMap = new HashMap<>();

	public void init(String path, String pkg_params) throws IOException{
        System.out.println("ModelConfig init, path = " + path + ", pkg_params = " + pkg_params + "\n");
        //从 train.conf中的conf字段读取para.conf文件路径，这个 para 是全局 para
		BufferedReader br = new BufferedReader(new FileReader(path));
		String line = null;
		while((line = br.readLine())!=null){
			String[]segs = StringUtils.split(line, ',');
			if(segs.length!=5){
				System.err.println("InValid config line: "+line+"\n");
				continue;
			}
			Parameter para = new Parameter();
			para.alpha = Float.valueOf(segs[1]);
			para.beta = Float.valueOf(segs[2]);
			para.lambda1 = Float.valueOf(segs[3]);
			para.lambda2 = Float.valueOf(segs[4]);
			names.add(segs[0]);
			paras.add(para);
		}
		br.close();
        // 从 train.conf 中的 pkg_params json 字段解析每个包的 para
        if (pkg_params != null && !pkg_params.isEmpty()) {
            ObjectMapper objectMapper = new ObjectMapper();
            // 解析 JSON 字符串为 Map<String, String>
            Map<String, String> parsedMap = objectMapper.readValue(pkg_params, 
                objectMapper.getTypeFactory().constructMapType(Map.class, String.class, String.class));
            for (Map.Entry<String, String> entry : parsedMap.entrySet()) {
                String pkgName = entry.getKey().toLowerCase();
                String[] values = StringUtils.split(entry.getValue(), ',');
                if (values.length == 4) {
                    Parameter para = new Parameter();
                    para.alpha = Float.parseFloat(values[0]);
                    para.beta = Float.parseFloat(values[1]);
                    para.lambda1 = Float.parseFloat(values[2]);
                    para.lambda2 = Float.parseFloat(values[3]);
                    pkgParamsMap.put(pkgName, para);
                } else {
                    System.err.println("Invalid pkg_params value for pkg_name: " + pkgName + ", value: " + entry.getValue());
                }
            }
        }
	}
	
	final public String getName(int i){
		return names.get(i);
	}
	
	final public Parameter getPara(int i){
		return paras.get(i);
	}
	
	final public int paraSize(){
		return paras.size();
	}
	
	// 新增方法，用于获取 pkg_name 对应的 Parameter 参数
    public Parameter getPkgParameter(String pkgName) {
        pkgName = pkgName.toLowerCase();
        // 尝试从 pkgParamsMap 中获取对应参数
        Parameter parameter = pkgParamsMap.get(pkgName);
        // 如果 pkgParamsMap 中不存在该 pkgName 对应的参数，且 paras 列表不为空，则返回 paras 中的第一个参数
        if (parameter == null && !paras.isEmpty()) {
            parameter = paras.get(0);
        }
        return parameter;
    }

    public void print() {
        System.out.println("model config......");
        for (int i = 0; i < paras.size(); ++i) {
            Parameter parameter = paras.get(i);
            System.out.printf("model_name=%s:\n\talpha=%.2f, beta=%.2f, lambda1=%.2f, lambda2=%.2f\n", names.get(i),
                    parameter.alpha, parameter.beta, parameter.lambda1, parameter.lambda2);
        }
        // 打印每个 pkg_name 的参数信息
        for (Map.Entry<String, Parameter> entry : pkgParamsMap.entrySet()) {
            String pkgName = entry.getKey();
            Parameter parameter = entry.getValue();
            System.out.printf("pkg_name=%s:\n\talpha=%.2f, beta=%.2f, lambda1=%.2f, lambda2=%.2f\n", pkgName,
                    parameter.alpha, parameter.beta, parameter.lambda1, parameter.lambda2);
        }
    }
}
