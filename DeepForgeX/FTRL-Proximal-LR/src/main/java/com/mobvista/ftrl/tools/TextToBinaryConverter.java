package com.mobvista.ftrl.tools;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import org.apache.commons.lang3.StringUtils;

import com.mobvista.ftrl.struct.FeatureInfo;
import com.mobvista.ftrl.struct.ModelConfig;

public class TextToBinaryConverter {

    private Map<String, FeatureInfo> features = new ConcurrentHashMap<String, FeatureInfo>();
    private String textModelPath;
    private String binModelPath;

	public static char FEATURE_FIELD_SEP='\002';

    public TextToBinaryConverter(String textModelPath, String binModelPath) {
        this.textModelPath = textModelPath;
        this.binModelPath = binModelPath;
    }

    public void convert() throws IOException {
        loadTextModel();
        dumpBinaryModel();
    }

    private void loadTextModel() throws IOException {
        System.out.println("Loading text model from " + textModelPath);
        BufferedReader br = new BufferedReader(new FileReader(textModelPath));
        String line;
        while ((line = br.readLine()) != null) {
            String[] segs = StringUtils.split(line, FEATURE_FIELD_SEP);
            FeatureInfo featureInfo = features.get(segs[0]);
            if (featureInfo == null) {
                featureInfo = new FeatureInfo(1);
                features.put(segs[0], featureInfo);
            }
            for (int i = 0; i < 1; ++i) {
                featureInfo.omiga[i] = Float.valueOf(segs[1]);
                featureInfo.z[i] = Float.valueOf(segs[2]);
            }
            featureInfo.positive = Long.valueOf(segs[3]);
            featureInfo.negtive = Long.valueOf(segs[4]);
        }
        br.close();
    }

    private void dumpBinaryModel() throws IOException {
        System.out.println("Dumping binary model to " + binModelPath);
        DataOutputStream[] writers = new DataOutputStream[1];
        for (int i = 0; i < 1; ++i) {
            String modelPath = binModelPath;
            writers[i] = new DataOutputStream(new FileOutputStream(modelPath));
        }
        Iterator<Entry<String, FeatureInfo>> entries = features.entrySet().iterator();
        while (entries.hasNext()) {
            Entry<String, FeatureInfo> entry = entries.next();
            String key = entry.getKey();
            FeatureInfo featureInfo = entry.getValue();
            if (featureInfo.positive + featureInfo.negtive > 0) {
                for (int i = 0; i < 1; ++i) {
                    writers[i].writeUTF(key);
                    writers[i].writeDouble(featureInfo.omiga[i]);
                    writers[i].writeDouble(featureInfo.z[i]);
                    writers[i].writeLong(featureInfo.positive);
                    writers[i].writeLong(featureInfo.negtive);
                }
            }
        }
        for (int i = 0; i < 1; ++i) {
            writers[i].close();
        }
    }

    public static void main(String[] args) {
        if (args.length != 2) {
            System.out.println("Usage: java com.mobvista.ftrl.convert.TextToBinaryConverter <text_model_path> <bin_model_path>");
            return;
        }

        String textModelPath = args[0];
        String binModelPath = args[1];

        TextToBinaryConverter converter = new TextToBinaryConverter(textModelPath, binModelPath);
        try {
            converter.convert();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}