package com.mobvista.ftrl.struct;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.Map.Entry;
import java.util.concurrent.LinkedBlockingQueue;

import com.mobvista.ftrl.util.ToolFucs;

public class AucCollector {

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
	
	public static SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyyMMddHHmmss");
	
	LinkedBlockingQueue<Food> queue = new LinkedBlockingQueue<Food>(10000);
	public static class Food{
		double predict;
		//修改为double
		double label;
		int index;
		String key;
		public Food(double pre,double lab,int ind,String key){
			predict = pre;
			label = lab;
			index = ind;
			this.key = key;
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

	public AucCollector(String path){
		this.path = path;
	}
	
	//修改：适配回归和分类，改为double
	public void feed(double predict, double label,int index,String key) {
		try {
			queue.put(new Food(predict,label,index,key));
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
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
				historyAucs.add(new HashMap<String, ImprClick>());
				++i;
			}
		}
	}
	
	// 在init方法中添加读取配置的逻辑
    public void init(ModelConfig conf) {
        this.conf = conf;
        
        // 读取loss_func配置
        try {
            Properties prop = new Properties();
            prop.load(new FileInputStream("./conf/train.conf"));
            this.lossFunc = prop.getProperty("loss_func").toLowerCase();
        } catch (IOException e) {
            System.err.println("计算指标：Failed to read train.conf, using default loss_func: cross_entropy");
        }
        
        initAuc();
        ToolFucs.confirmDir(path);
        
        new Thread(new Runnable(){
            public void run() {
                while (true) {
                    try {
                        Food food = queue.take();
                        collect(food);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        }).start();
    }

	
	public void dumpAuc(boolean fullDump) throws IOException{
		List<Map<String,ImprClick>> tempAucs =null;
		synchronized (this) {
			tempAucs =aucs;
			initAuc();
		}
		final List<Map<String,ImprClick>> dumpAucs = tempAucs;
		final String strDate = simpleDateFormat.format(new Date());
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
						
						if ("mse".equals(lossFunc)) {
							// 回归任务写入格式: label\tpredict
							for (Entry<String, ImprClick> en : map.entrySet()) {
								bw.write(en.getKey() + "\n"); // 已经是label\tpredict格式
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
        if ("mse".equals(lossFunc)) {
			System.out.println("--调用回归py");
            script = "python figure_auc_regression.py " + predictFile + " " + predictFile + "_auc";
        } else {
			System.out.println("--调用分类py");
            script = "python score_kdd.py " + predictFile + " " + predictFile + "_auc";
        }
        
        Process process = Runtime.getRuntime().exec(script);
        try {
            if (process.waitFor() != 0) {
                BufferedReader br = new BufferedReader(new InputStreamReader(process.getErrorStream()));
                String line = null;
                while ((line = br.readLine()) != null) {
                    System.err.println(line);
                }
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
	
	// 修改collect方法
    final public void collect(Food food) {
        synchronized (this) {
            if ("mse".equals(lossFunc)) {
                // MSE回归任务，直接存储label和predict，不需要处理key和点击曝光
                Map<String, ImprClick> map = aucs.get(food.index);
                String entry = String.format("%.6f\t%.6f", food.label, food.predict);
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
