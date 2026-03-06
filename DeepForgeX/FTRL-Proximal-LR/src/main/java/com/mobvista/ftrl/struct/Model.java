package com.mobvista.ftrl.struct;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import org.apache.commons.lang3.StringUtils;


public class Model {
	public Map<String, FeatureInfo> features = new ConcurrentHashMap<String, FeatureInfo>();
	ModelConfig conf;
	String path;

	int featureFreqThr;
	
	public static char FEATURE_FIELD_SEP='\002';
	
	ReentrantReadWriteLock.WriteLock writeLock = new ReentrantReadWriteLock().writeLock();

	public void load(String modelPath) throws IOException {

		System.out.println("Loading model ...");
		int paraSize = conf.paraSize();
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(modelPath));
		} catch (FileNotFoundException e) {
			System.out.printf(
					"model %s is not exist. all feature statistic set to 0.\n",
					modelPath);
			return;
		}
		String line = null;
		while ((line = br.readLine()) != null) {
			String[] segs = StringUtils.split(line, FEATURE_FIELD_SEP);
			FeatureInfo featureInfo = features.get(segs[0]);
			if (featureInfo == null) {
				featureInfo = new FeatureInfo(paraSize);
				features.put(segs[0], featureInfo);
			}
			for (int i = 0; i < paraSize; ++i) {
				featureInfo.omiga[i] = Float.valueOf(segs[1]);
				featureInfo.z[i] = Float.valueOf(segs[2]);
			}
			featureInfo.positive = Long.valueOf(segs[3]);
			featureInfo.negtive = Long.valueOf(segs[4]);
		}
		br.close();
	}

	public Model(ModelConfig conf, int featureFreqThr,String path) {
		this.conf = conf;
		this.featureFreqThr = featureFreqThr;
		this.path = path;
	}

	public void dump(boolean trimModel) throws IOException {
		if (trimModel) {
			writeLock.lock();
		} else {
			if (!writeLock.tryLock()) {
				System.out.println("Obtain write lock failed!");
				return;
			}
		}
		try {
			System.out.println("Dumping model to " + path);
			File dir = new File(path);
			if (!dir.exists()) {
				dir.mkdir();
			}
			int paraSize = conf.paraSize();
			BufferedWriter[] writers = new BufferedWriter[paraSize];
			for (int i = 0; i < paraSize; ++i) {
				String modelPath = dir + File.separator + conf.getName(i);
				System.out.printf("Dump %s to %s\n", conf.names.get(i),
						modelPath);
				writers[i] = new BufferedWriter(new FileWriter(modelPath));
			}
			Iterator<Entry<String, FeatureInfo>> entries = features.entrySet().iterator();
			while(entries.hasNext()){
				Entry<String, FeatureInfo> entry = entries.next();
				String key = entry.getKey();
				FeatureInfo featureInfo = entry.getValue();
				if (featureInfo.positive + featureInfo.negtive > featureFreqThr) {
					for (int i = 0; i < paraSize; ++i) {
						writers[i].write(key + FEATURE_FIELD_SEP + featureInfo.omiga[i]
								+ FEATURE_FIELD_SEP + featureInfo.z[i] + FEATURE_FIELD_SEP
								+ featureInfo.positive + FEATURE_FIELD_SEP
								+ featureInfo.negtive + "\n");
					}
				} else if (trimModel) {
					entries.remove();
				}
			}
			for (int i = 0; i < paraSize; ++i) {
				writers[i].close();
			}
		} finally {
			writeLock.unlock();
		}
	}

}
