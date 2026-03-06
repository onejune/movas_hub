package com.mobvista.ftrl.collector;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;

import com.mobvista.ftrl.collector.food.AUCFood;
import com.mobvista.ftrl.struct.AucCollector.Food;
import com.mobvista.ftrl.struct.ModelConfig;
import com.mobvista.ftrl.struct.Parameter;
import com.mobvista.ftrl.util.Constants;
import com.mobvista.ftrl.util.ToolFucs;

public class AucCollector extends BaseCollector<AUCFood> {

	// 添加lossFunc字段
    private String lossFunc = "cross_entropy"; // 默认值

	public static class ImprClick{
		long impr = 0;
		long click = 0;
		public void add(ImprClick imprClick){
			impr+=imprClick.impr;
			click+=imprClick.click;
		}
	}

	List<Map<String,ImprClick>> aucs;
	
	final List<Map<String,ImprClick>> historyAucs = new ArrayList<Map<String,ImprClick>>();
	
	ModelConfig conf;
	
	String path;
	
	String name;
	
	public void setName(String name) {
		this.name = name;
	}

	public AucCollector(String path,ModelConfig conf){
		this.path = path;
		this.conf = conf;
	}
	public AucCollector(String path){
		this.path = path;
		this.conf = new ModelConfig();
		this.conf.names.add("default");
		this.conf.paras.add(new Parameter());
	}
	
	
	private void initAuc(){
		int i = 0;
		aucs = new ArrayList<Map<String,ImprClick>>();
		while (i < conf.paraSize()) {
			aucs.add(new HashMap<String, ImprClick>());
			++i;
		}
		if(historyAucs.isEmpty()){
			i = 0;
			while (i < conf.paraSize()) {
				historyAucs.add(new ConcurrentHashMap<String, ImprClick>());
				++i;
			}
		}
	}
    @Override
	public boolean time2dump(){
		return cnt%(Constants.PRINT_WIN*10)==0;
	}
	
	public boolean addCount(AUCFood food){
		if (food.index==0){
			cnt+=1;
			return true;
		}
		return false;
	}
	
	public void init(ModelConfig conf, String loss_func) {
		this.conf = conf;
        this.lossFunc = loss_func;
        
        initAuc();
        ToolFucs.confirmDir(path);
		super.init();
	}
	
	public void dump(boolean fullDump) throws IOException{
		System.err.println(basicInfo().append("Dumping AUC ...").toString());
		List<Map<String,ImprClick>> tempAucs =null;
		synchronized (this) {
			tempAucs =aucs;
			initAuc();
		}
		final List<Map<String,ImprClick>> dumpAucs = tempAucs;
		final String strDate = getTime();
		final String prefix = path+File.separator+"predict_"+name+"_";
		final boolean fFullDump = fullDump;
		
		// 修改dumpAuc方法中的文件写入逻辑
		Thread t = new Thread() {
			public void run() {
				for (int i = 0; i < conf.paraSize(); ++i) {
					try {
						Map<String, ImprClick> map = dumpAucs.get(i);
						Map<String, ImprClick> history = historyAucs.get(i);
						String fileName = prefix + conf.getName(i) + "_" + strDate;
						
						if (fFullDump) {
							fileName += "_full";
						}
						
						BufferedWriter bw = new BufferedWriter(new FileWriter(fileName));
						
						if ("mse".equals(lossFunc) || "wce".equals(lossFunc)  || "mae".equals(lossFunc) || "msle".equals(lossFunc) || "huber_loss".equals(lossFunc)) {
							// 回归任务写入格式: label\tpredict
							for (Entry<String, ImprClick> en : map.entrySet()) {
								bw.write(en.getKey() + "\n"); // 已经是key \t label \t predict格式
							}
						} else {
							// 分类任务保持原有写入格式
							for (Entry<String, ImprClick> en : map.entrySet()) {
								bw.write(String.format("%s\t%d\t%d\n",
										en.getKey(), en.getValue().impr,
										en.getValue().click));
								if (history.containsKey(en.getKey())) {
									history.get(en.getKey()).add(en.getValue());
								} else {
									history.put(en.getKey(), en.getValue());
								}
							}
						}
						bw.close();
						
						final String fFileName = fileName;
						Thread t2 = new Thread() {
							@Override
							public void run() {
								try {
									calculateAuc(fFileName);
								} catch (IOException e) {
									e.printStackTrace();
								}
							}
						};
						t2.start();
						
						if (fFullDump) {
							try {
								t2.join();
							} catch (InterruptedException e) {
								e.printStackTrace();
							}
						}
					} catch (IOException e) {
						e.printStackTrace();
					}
				}
			}
		};


		t.setDaemon(true);
		t.start();
		if(fullDump){
			try {
				System.err.println("Waiting full dump finish.");
				t.join();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	

	@SuppressWarnings("deprecation")
    // 修改dumpAuc方法中的calculateAuc调用
    private void calculateAuc(String predictFile) throws IOException {
		String script;
		if ("mse".equals(lossFunc) || "wce".equals(lossFunc)  || "mae".equals(lossFunc) || "msle".equals(lossFunc) || "huber_loss".equals(lossFunc)) {
			System.out.println("-调用回归py");
			//命令行传参
			script = "python figure_auc_regression.py " + predictFile + " " + predictFile + "_auc";
		} else {
			System.out.println("-调用分类py");
			script = "python score_kdd.py " + predictFile + " " + predictFile + "_auc";
		}

		
		Process process = Runtime.getRuntime().exec(new String[]{"sh", "-c", script});

		try {
			if(process.waitFor()!=0){
				BufferedReader br = new BufferedReader(new InputStreamReader(process.getErrorStream()));
				String line = null;
				while((line = br.readLine())!=null){
					System.err.println(line);
				}
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
        
        try (BufferedReader br = new BufferedReader(new FileReader(predictFile + "_auc"))) {
            String line;
            int count = 0;
            while ((line = br.readLine()) != null && count < 80) {
                System.out.println(line);
                count++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
		
	}
	
	// 修改collect方法
    final public void collect(AUCFood food) {
        synchronized (this) {
            if ("mse".equals(lossFunc) || "wce".equals(lossFunc)  || "mae".equals(lossFunc) || "msle".equals(lossFunc) || "huber_loss".equals(lossFunc)) {
                // MSE回归任务，直接存储label和predict，不需要处理key和点击曝光
                Map<String, ImprClick> map = aucs.get(food.index);

				// 修改:6.11 保存三列
				String entry = String.format("%s\t%.6f\t%.6f", food.key, food.label, food.predict);
                // String entry = String.format("%.6f\t%.6f", food.label, food.predict);

                ImprClick ic = map.get(entry);
                if (ic == null) {
                    ic = new ImprClick();
                    map.put(entry, ic);
                }
                ic.impr++; // 这里impr只是计数用
            } else {
                // 分类任务，保持原有逻辑
                Map<String, ImprClick> map = aucs.get(food.index);
                String spre = String.format("%.6f|%s", food.predict, food.key);
                ImprClick ic = map.get(spre);
                if (ic == null) {
                    ic = new ImprClick();
                    map.put(spre, ic);
                }
                ic.impr += 1;
                ic.click += food.label;
            }
        }
    }
	
	
}	
