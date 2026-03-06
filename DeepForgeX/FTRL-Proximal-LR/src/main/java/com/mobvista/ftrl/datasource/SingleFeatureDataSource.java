package com.mobvista.ftrl.datasource;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import org.apache.commons.lang3.StringUtils;

import com.mobvista.ftrl.linereader.S3LineReader;
import com.mobvista.ftrl.struct.Sample;
import com.mobvista.ftrl.utils.FeatureDiscretizer;

public class SingleFeatureDataSource implements DataSource {
    // 常量定义
    private static final int FEATURE_ARRAY_SIZE = 10;
    private static final char FEATURE_SEPARATOR = '\002';
    private static final char MULTI_VALUE_SEPARATOR = '\001';
    
    // 成员变量
    private final Map<String, Integer> name2idx = new HashMap<>();
    private String[] names;
    private final List<List<Integer>> combine_schema = new ArrayList<>();
    private int label_idx = -1;
    private int bidPrice_idx = -1;
    private int click_label_idx = -1;
    private int extra5_idx = -1;
    private int camp_idx = -1;
    private int demand_pkgname_idx = -1;
    private int business_type_idx = -1;
    private int bundle_cate_list_idx = -1;
    private String label_name = "label";
    private String click_label_name = "click_label";
    private String keys_name = "";
    private int duf_inner_dev_pkg_imp_bucket_7d_idx = -1;
    private List<Integer> keys_index_list = new ArrayList<>();
    private int max_col = -1;   
    private String model_type = "ivr";
    private Map<String, Integer> continuous_features = new HashMap<>();

    //添加：根据loss来进行label处理
    private String lossType = "cross_entropy"; // 默认是分类任务
    
    // 添加离散化配置映射
    private final Map<String, FeatureDiscretizer.DiscretizeMethod> discretizeConfig = new HashMap<>();
    public SingleFeatureDataSource() {
        // 初始化离散化配置
        discretizeConfig.put("cate_list", FeatureDiscretizer.DiscretizeMethod.MANUAL_BUCKET);
        discretizeConfig.put("bid_floor", FeatureDiscretizer.DiscretizeMethod.LOG);
        discretizeConfig.put("age", FeatureDiscretizer.DiscretizeMethod.EQUAL_WIDTH);
    }
    
    // 删除 loadExtraFeatureConfig 方法和相关的成员变量
    boolean loadCombineSchema(String path) {
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (line.charAt(0) == '#') {
                    continue;
                }
                String[] segs = StringUtils.split(line.trim(), '#');
                List<Integer> l = new ArrayList<>(segs.length);
                
                for (String seg : segs) {
                    Integer idx = name2idx.get(seg);
                    if (idx == null) {
                        System.out.println("null fname " + seg);
                        throw new RuntimeException("Not exist column " + seg + " line is " + line);
                    }
                    l.add(idx);
                }
                combine_schema.add(l);
            }
            System.out.println("Load " + combine_schema.size() + " feature combine schemas.");
            return true;
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }

    boolean loadColName(String path) {
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            System.out.println("loadColName:" + path);
            String line;
            
            while ((line = br.readLine()) != null) {
                String[] segs = StringUtils.split(line.trim(), ' ');
                if (segs.length != 2) {
                    System.out.println("invalid colName:" + line);
                    continue;
                }
                int idx = Integer.parseInt(segs[0]);
                max_col = Math.max(max_col, idx);
                name2idx.put(segs[1], idx);
            }

            if (!name2idx.containsKey(label_name)) {
                System.out.println("NO \"label\" column in column_name file.");
                return false;
            }
            
            // 初始化索引
            label_idx = name2idx.get(label_name);
            click_label_idx = name2idx.getOrDefault(click_label_name, -1);
            extra5_idx = name2idx.getOrDefault("requestid", -1);
            camp_idx = name2idx.getOrDefault("campaignid", -1);
            bidPrice_idx = name2idx.getOrDefault("bidPrice", -1);
            demand_pkgname_idx = name2idx.getOrDefault("demand_pkgname", -1);
            business_type_idx = name2idx.getOrDefault("business_type", -1);
            bundle_cate_list_idx = name2idx.getOrDefault("cate_list", -1);
            duf_inner_dev_pkg_imp_bucket_7d_idx = name2idx.getOrDefault("duf_inner_dev_pkg_imp_bucket_7d", -1);

            if (!keys_name.isEmpty()) {
                String[] keyNames = keys_name.split(",");
                for (String keyName : keyNames) {
                    Integer idx = name2idx.get(keyName.trim());
                    if (idx != null) {
                        keys_index_list.add(idx);	
                    }
                    else {
                        System.out.println("Not exist column " + keyName);
                    }
                }	
            }

            names = new String[max_col + 1];
            for (Map.Entry<String, Integer> entry : name2idx.entrySet()) {
                names[entry.getValue()] = entry.getKey();
            }
            
            System.out.println("Load " + name2idx.size() + " columns, label column is " + label_name + ", index is " + 
                label_idx + ", click_label index is " + click_label_idx + ", keys column is [" + keys_name + "], index is " + 
                keys_index_list + ", max_col is " + max_col);
            return true;
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }

    public int bkdrHash(String str) {
        int seed = 31; // 通常使用31或其他奇数作为种子
        int hash = 0;
        for (int i = 0; i < str.length(); i++) {
            hash = hash * seed + str.charAt(i);
        }
        return hash;
    }
    
    public String figure_device_id_bucket(String req_id){
        String segs[] = StringUtils.splitPreserveAllTokens(req_id, '\003');
        if(segs.length >= 6){
            String dev_id = segs[5];
            int hash = bkdrHash(dev_id) % 10000;
            return String.format("%d", hash);
        }
        return "unknow";
    }

    public boolean init(String conf) {
        if (!conf.endsWith("/")) {
            conf += "/";
        }
        try {
			Properties pro = new Properties();
			FileInputStream in = new FileInputStream(conf + "train.conf");
			pro.load(in);
			in.close();
            if (pro.getProperty("label") != null) {
			    label_name = pro.getProperty("label");
            }
            if (pro.getProperty("model_type") != null) {
                model_type = pro.getProperty("model_type");
            }
            if(pro.getProperty("click_label") != null){
                click_label_name = pro.getProperty("click_label");
            }
            if(pro.getProperty("keys")!= null){
			    keys_name = pro.getProperty("keys");
            }
            // 新增：获取配置中的loss type
            if (pro.getProperty("loss_func") != null) {
                lossType = pro.getProperty("loss_func").trim().toLowerCase();
            }
            //获取连续值特征集合
            if (pro.getProperty("continuous_features") != null){
                String[] continuousFeatures = pro.getProperty("continuous_features").split(",");
                for (String featureName : continuousFeatures) {
                    continuous_features.put(featureName, 1);
                }
                System.out.println("continuous_features:" + continuous_features);
            }
            System.out.println("model_type:" + model_type);
		} catch (Exception e) {
			e.printStackTrace();
			return false;
		}
        return loadColName(conf + "column_name") && loadCombineSchema(conf + "combine_schema");
    }
    
    // 修改 extracSample 方法
    public Sample extracSample(String line) {
        if (line == null || line.isEmpty()) {
            return null;
        }
        String[] segs = StringUtils.splitPreserveAllTokens(line, FEATURE_SEPARATOR);
        if(max_col + 1 > segs.length){
            String extra5 = segs[extra5_idx];
            System.err.println("max col for column_name is " + max_col + ", but the column count for sample is " + segs.length + ", requestid is " + extra5);
            return null;
        }
        // 如果是 cvr model，只保留 click_label = 1的样本, 判断 cvr model 的方法是label_idx != click_label_idx
        if(click_label_idx != -1 && "cvr".equals(model_type) && Integer.parseInt(segs[click_label_idx]) == 0){
            return null;
        }
        if(bidPrice_idx != -1 && Double.parseDouble(segs[bidPrice_idx]) <= 0){
            return null;
        }
        Sample sample = new Sample();
        if(demand_pkgname_idx != -1){
            //if(business_type_idx != -1 && segs[business_type_idx].equals("shein_cps") && segs[demand_pkgname_idx].equals("COM.ZZKKO")){
            //    segs[demand_pkgname_idx] = "COM.ZZKKO.CPS";
            //}
            sample.demand_pkgname = segs[demand_pkgname_idx];
        }

        try {
            //修改：根据loss来进行label处理
            float labelValue = Float.parseFloat(segs[label_idx]);
            switch (lossType) {
                case "cross_entropy":
                    sample.label = labelValue == 1 ? 1 : 0;
                    break;
                case "mse":
                case "wce":
                case "msle":
                case "huber_loss":
                case "mae":
                    sample.label = labelValue;
                    break;
                default:
                    System.err.println("Unknown loss_type: " + lossType);
                    break;
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            //System.err.println("Invalid line is:" + line.trim());
            return null;
        }
        if(sample.label >= 100){
            return null;
        }
        sample.key = "all";
        String[] prefixes = new String[FEATURE_ARRAY_SIZE];
        String[] values = new String[FEATURE_ARRAY_SIZE];
        
        for (List<Integer> schema : combine_schema) {
            try {
                combine_feature(prefixes, values, 0, sample, schema, segs);
            } catch (ArrayIndexOutOfBoundsException e) {
                //System.err.println("Invalid line is:|" + line.trim() + "|");
                return null;
            }
        }

        if (extra5_idx != -1) {
            sample.name = camp_idx != -1 ? 
                segs[extra5_idx] + "_" + segs[camp_idx] : segs[extra5_idx];
        }

        if (!keys_index_list.isEmpty()) {
            StringBuilder sb = new StringBuilder();
            for (int idx : keys_index_list) {
                sb.append(segs[idx]);
                sb.append('#');
            }	
            if (sb.length() > 0) {
                sb.setLength(sb.length() - 1);
                sample.key = sb.toString().replace('_', '.');
            }
        }
        else if (demand_pkgname_idx != -1) {
            sample.key = segs[demand_pkgname_idx].replace('_', '.');
        }
        return sample;
    }

    void combine_feature(String[] prefixes, String[] values, int len, Sample sample, List<Integer> schema, String[] cols) {
        if (len == schema.size()) {
            StringBuilder sb = new StringBuilder();
            String continuout_feature_value = "";
            for (int i = 0; i < len; ++i) {
                String feature_name = prefixes[i];
                String feature_value = values[i];
                //如果是连续特征，或者包含连续特征，需要特殊处理
                if(continuous_features.containsKey(feature_name)){
                    continuout_feature_value = feature_value;
                    feature_value = feature_name;
                }
                sb.append(feature_name).append('=').append(feature_value);
                if (i != len - 1) {
                    sb.append(MULTI_VALUE_SEPARATOR);
                }
            }
            sample.strFeatures.add(sb.toString());
            //把连续特征或者包含连续特征的组合特征的特征值进行 log 变换后存储
            if (continuout_feature_value.length() > 0) {
                double rawValue = Double.parseDouble(continuout_feature_value);
                // 进行 log 变换，通常使用 log(1 + x) 来处理可能的零值或负值
                double logValue = Math.log(1.0 + rawValue); 
                sample.continuousFeatures.put(sb.toString(), logValue);
            }
            return;
        }
        
        int fidx = schema.get(len);
        if (fidx >= names.length || fidx >= cols.length) {
            return;
        }
        
        String feature_name = names[fidx];
        prefixes[len] = feature_name;
        
        for (String value : StringUtils.split(cols[fidx], MULTI_VALUE_SEPARATOR)) {
            if (value.isEmpty() || "none".equals(value) || value.equals("\"\"")) {
                return;
            }
            values[len] = value;
            combine_feature(prefixes, values, len + 1, sample, schema, cols);
        }
    }
    
    public static void main(String[] args) throws IOException {
        //S3LineReader reader  = new S3LineReader("s3://mob-emr-test/wanjun/m_sys_model/m_system_train_data.ctr.2017080815");
        System.out.println("SingleFeatureDataSource test begin......");
        S3LineReader reader  = new S3LineReader("s3://mob-emr-test/wanjun/m_sys_model/feature_merge_csv_ctr/20170906/2017090600/part-00000.gz");
        SingleFeatureDataSource source = new SingleFeatureDataSource();
        source.init("C:\\02_workshop\\eclipse_workspace\\0_ftrl_maven\\conf\\");
        String line = null;
        int i  = 0;
        while((line = reader.readLine())!=null){
            Sample sample = source.extracSample(line);
            i++;
            if(sample ==null){
                System.out.println("the "+i+"th sample is null");
            }else{
                System.out.println(sample.strFeatures.toString());
            }
            if(i>1000)break;
        }
    }
}
