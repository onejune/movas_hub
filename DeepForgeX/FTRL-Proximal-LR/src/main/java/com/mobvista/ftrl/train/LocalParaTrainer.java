package com.mobvista.ftrl.train;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Properties;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;

import org.apache.commons.lang3.StringUtils;

import com.mobvista.ftrl.collector.AucCollector;
import com.mobvista.ftrl.collector.LossCollector;
import com.mobvista.ftrl.collector.NotRealCollector;
import com.mobvista.ftrl.collector.TimeCollector;
import com.mobvista.ftrl.collector.food.AUCFood;
import com.mobvista.ftrl.datasource.DataSource;
import com.mobvista.ftrl.linereader.LineReader;
import com.mobvista.ftrl.linereader.LocalLineReader;
import com.mobvista.ftrl.struct.FeatureInfo;
import com.mobvista.ftrl.struct.ModelConfig;
import com.mobvista.ftrl.struct.ModelText;
import com.mobvista.ftrl.struct.Parameter;
import com.mobvista.ftrl.struct.Sample;
import com.mobvista.ftrl.util.ToolFucs;

public class LocalParaTrainer implements Runnable, Trainer {
    int paraSize;
    double[] grad;
    double[] delta;
    double[] predict;
    boolean running = true;

    public LinkedBlockingQueue<RunParser> trainDataQueue;
    ExecutorService cachedThreadPool;
    String modelPath;
    double featureIncludeRate = 1;
    double pos_weight = 1.0; // 正样本权重
    int featureFreqThr = 0;
    String output;
    String aucOutput;
    public ModelText model;
    ModelConfig modelConfig;
    AucCollector aucCollector;
    TimeCollector timeCollector;
    LossCollector lossCollector;
    NotRealCollector notRealCollector;
    private double[] omigaSumBuffer;
    DataSource dataSource;
    double[][] sqrts = new double[100][10000];
    String loss_func = "cross_entropy";
    // Focal Loss 参数
    double fl_alpha = 0.25;  // 正样本权重
    double fl_gamma = 2.0;   // 聚焦参数
    double mse_scale = 1.0;  // MSE 缩放因子，用于调整预测值的范围

    //Huber loss 参数
    double huber_delta = 1.0;

    public boolean init(String confBase, DataSource dataSource) {
        try {
            Properties pro = new Properties();
            FileInputStream in = new FileInputStream(confBase + "train.conf");
            pro.load(in);
            in.close();
            modelPath = pro.getProperty("model");
            output = pro.getProperty("output");
            if (!confirmOuput()) {
                throw new RuntimeException();
            }
            if (pro.getProperty("fir") != null) {
                featureIncludeRate = Double.valueOf(pro.getProperty("fir"));
            }
            if (pro.getProperty("fft") != null) {
                featureFreqThr = Integer.parseInt(pro.getProperty("fft"));
            }
            if (pro.getProperty("pos_weight") != null) {
                pos_weight = Double.parseDouble(pro.getProperty("pos_weight"));
            }
            if (pro.getProperty("loss_func") != null) {
                loss_func = String.valueOf(pro.getProperty("loss_func"));
            }
            if (pro.getProperty("fl_alpha") != null) {
                fl_alpha = Double.parseDouble(pro.getProperty("fl_alpha"));
            }
            if (pro.getProperty("fl_gamma") != null) {
                fl_gamma = Double.parseDouble(pro.getProperty("fl_gamma"));
            }
            modelConfig = new ModelConfig();
            modelConfig.init(pro.getProperty("conf"), pro.getProperty("paras"));

            model = new ModelText(modelConfig, featureFreqThr, output, loss_func);
            model.load(modelPath);
            aucCollector = new AucCollector(aucOutput, modelConfig);
            aucCollector.init(modelConfig, loss_func);
            lossCollector = new LossCollector(modelConfig);
            lossCollector.init();
            timeCollector = new TimeCollector();
            timeCollector.init();
            notRealCollector = new NotRealCollector(this);
            notRealCollector.init();
            this.dataSource = dataSource;
            paraSize = modelConfig.paraSize();
            grad = new double[paraSize];
            delta = new double[paraSize];
            predict = new double[paraSize];

            omigaSumBuffer = new double[paraSize];
            int cores = Runtime.getRuntime().availableProcessors();
            cachedThreadPool = Executors.newFixedThreadPool(50);
            trainDataQueue = new LinkedBlockingQueue<RunParser>(3000);
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
        return true;
    }

    public boolean confirmOuput() {
        aucOutput = output + File.separatorChar + "auc";
        return ToolFucs.confirmDir(output) && ToolFucs.confirmDir(output + File.separatorChar + "auc");
    }

    public void prepareTrain(String dataPath, String name) {
        System.out.println("Begin training ...");
        System.out.printf(
                "sample_path=%s\nmodel_path=%s\noutput_path=%s\nfeature_include_rate=%.4f, \nfeature_freqence_threshold=%d, \nloss_func=%s,\nfl_alpha=%f, \nfl_gamma=%f, \npos_weight=%.4f\n",
                dataPath, modelPath, output, featureIncludeRate, featureFreqThr, loss_func, fl_alpha, fl_gamma, pos_weight);
        modelConfig.print();
        if (name == null || name.isEmpty()) {
            String[] segs = StringUtils.split(dataPath, File.separatorChar);
            SimpleDateFormat sdf1 = new SimpleDateFormat("yyyyMMdd");
            SimpleDateFormat sdf2 = new SimpleDateFormat("yyyy-MM-dd");
            int i;
            for (i = segs.length - 1; i >= 0; --i) {
                try {
                    sdf1.parse(segs[i]);
                    break;
                } catch (ParseException e) {
                }
                try {
                    sdf2.parse(segs[i]);
                    break;
                } catch (ParseException e) {
                }
            }
            if (i < 0) {
                aucCollector.setName("DEFAULT");
            } else {
                aucCollector.setName(segs[i]);

            }
        } else {
            aucCollector.setName(name);
        }
    }

    // 缓存 sigmoid 函数结果，避免重复计算
    private static final double[] SIGMOID_CACHE = new double[30001];
    static {
        for (int i = -15000; i <= 15000; i++) {
            double x = i / 1000.0;
            SIGMOID_CACHE[i + 15000] = 1 / (1 + Math.exp(-x));
        }
    }

    // 修改 cross_entropy_loss 等方法中的 sigmoid 计算
    final public void cross_entropy_loss(double [] omigaSumBuffer,
                                        double[] predict,
                                        double[] loss,
                                        double[] grad,
                                        Sample sample,
                                        FeatureInfo[] featureInfos,
                                        boolean updateLoss) {
        //loss 计算
        for (int i = 0; i < paraSize; ++i) {
            //分支1：z值较大
            if (omigaSumBuffer[i] >= 15) {
                predict[i] = 1;     
                predict[i] = Math.max(1e-15, Math.min(1 - 1e-15, predict[i]));
                if (updateLoss) {
                    if("cross_entropy".equals(loss_func)){
                        loss[i] = (sample.label == 1) ? 0 : -omigaSumBuffer[i];
                    }
                    else if ("focal_loss".equals(loss_func)){
                        if(sample.label == 0){
                            loss[i] = -fl_alpha * Math.pow(predict[i], fl_gamma) * Math.log(1 - predict[i]);
                        }
                    }
                }
            //分支1：z值较小
            } else if (omigaSumBuffer[i] <= -15) {
                predict[i] = 0;
                predict[i] = Math.max(1e-15, Math.min(1 - 1e-15, predict[i]));
                if (updateLoss) {
                    if("cross_entropy".equals(loss_func)){
                        loss[i] = (sample.label == 0) ? 0 : omigaSumBuffer[i];
                    }
                    else if ("focal_loss".equals(loss_func)){
                        if(sample.label == 1){
                            loss[i] = -(1 - fl_alpha) * Math.pow(1 - predict[i], fl_gamma) * Math.log(predict[i]);
                        }
                    }
                }
            //分支3：z值过sigmoid得到p
            } else {
                //int index = (int) (omigaSumBuffer[i] * 1000 + 15000);
                //if (index >= 0 && index < SIGMOID_CACHE.length) {
                //    predict[i] = SIGMOID_CACHE[index];
                //} else {
                predict[i] = (1 / (1 + Math.exp(-1 * omigaSumBuffer[i])));
                //}
                //predict[i] = Math.max(1e-15, Math.min(1 - 1e-15, predict[i]));
                if (updateLoss) {
                    if (sample.label == 0) {
                        if("cross_entropy".equals(loss_func)){
                            loss[i] = Math.log(1 - predict[i]);
                        }
                        else if ("focal_loss".equals(loss_func)){
                            // 负样本 Focal Loss
                            loss[i] = -(1 - fl_alpha) * Math.pow(predict[i], fl_gamma) * Math.log(1 - predict[i]);
                        }
                    } else {
                        if("cross_entropy".equals(loss_func)){
                            loss[i] = Math.log(predict[i]);
                        }
                        else if ("focal_loss".equals(loss_func)){
                            // 正样本 Focal Loss
                            loss[i] = -fl_alpha * Math.pow(1 - predict[i], fl_gamma) * Math.log(predict[i]);
                        }
                    }
                }
            }
            //梯度计算
            if("cross_entropy".equals(loss_func)){
                grad[i] = predict[i] - sample.label;
            }
            else if ("focal_loss".equals(loss_func)){
                double pt = predict[i];
                if (sample.label == 1) {
                    grad[i] = fl_alpha * Math.pow(1 - pt, fl_gamma) * (fl_gamma * pt * Math.log(pt) + pt - 1);
                } else {
                    grad[i] = (1 - fl_alpha) * Math.pow(pt, fl_gamma) * (fl_gamma * (1 - pt) * Math.log(1 - pt) + (1 - pt));
                }
            }
            if(sample.label == 1){
                grad[i] *= pos_weight;
            }
        }
    }

    //新增：wce_loss (5.30)
    final public void wce_loss(double [] omigaSumBuffer,
                                        double[] predict, 
                                        double[] loss, 
                                        double[] grad, 
                                        Sample sample, 
                                        FeatureInfo[] featureInfos, 
                                        boolean updateLoss) {
        //wce的loss计算
        for (int i = 0; i < paraSize; ++i) {
            predict[i] = (1 / (1 + Math.exp(-1 * omigaSumBuffer[i]))); // p = sigmoid(z)
            if (updateLoss) {
                // loss = t*log(p) + log(1-p)
                if (sample.label > 0){
                    loss[i] = sample.label * Math.log(predict[i]) + Math.log(1 - predict[i]);
                }
                else{
                    loss[i] = Math.log(1 - predict[i]);
                }
            }
            //梯度计算
            if("wce".equals(loss_func)){ 
                grad[i] = - ( sample.label - predict[i] * (sample.label + 1) );
            }
            else{
                System.err.println("-计算梯度时的loss错误！");
            }	
        }	
    }

    final public void mse_loss(double [] omigaSumBuffer,
                                        double[] predict, 
                                        double[] loss, 
                                        double[] grad, 
                                        Sample sample, 
                                        FeatureInfo[] featureInfos, 
                                        boolean updateLoss) {
        //loss 计算
        for (int i = 0; i < paraSize; ++i) {
            // MSE 不需要 sigmoid，直接使用线性输出
            predict[i] = omigaSumBuffer[i] * mse_scale;
            double y_true = sample.label;
            if (updateLoss) {
                // MSE 损失计算
                loss[i] = 0.5 * Math.pow(predict[i] - y_true, 2);
            }
            // MSE 梯度计算
            grad[i] = (predict[i] - y_true) * mse_scale;
        }
    }

    final public void msle_loss(double [] omigaSumBuffer,
                                        double[] predict, 
                                        double[] loss, 
                                        double[] grad, 
                                        Sample sample, 
                                        FeatureInfo[] featureInfos, 
                                        boolean updateLoss) {
        //loss 计算
        for (int i = 0; i < paraSize; ++i) {
            // MSLE = [log(1+y_hat)-log(1+y)]^2
            predict[i] = omigaSumBuffer[i];
            double y_true = sample.label;

            // 确保预测值大于-1
            double pred = Math.max(predict[i], -1 + 1e-5);
            if (updateLoss) {
                // MSLE损失计算
                loss[i] = 0.5 * Math.pow(Math.log(1 + pred) - Math.log(1 + y_true), 2);
            }
            // MSLE梯度计算
            grad[i] = (Math.log(1 + pred) - Math.log(1 + y_true)) / (1 + pred);
        }
    }


    final public void huber_loss(double [] omigaSumBuffer,
                              double[] predict, 
                              double[] loss, 
                              double[] grad, 
                              Sample sample, 
                              FeatureInfo[] featureInfos, 
                              boolean updateLoss) {
        // Huber损失计算
        for (int i = 0; i < paraSize; ++i) {
            // 直接使用线性输出
            predict[i] = omigaSumBuffer[i];
            double y_true = sample.label;
            double error = predict[i] - y_true;
            
            if (updateLoss) {
                // Huber损失计算
                if (Math.abs(error) <= huber_delta) {
                    loss[i] = 0.5 * Math.pow(error, 2);
                } else {
                    loss[i] = huber_delta * (Math.abs(error) - 0.5 * huber_delta);
                }
            }
            
            // Huber梯度计算
            if (Math.abs(error) <= huber_delta) {
                grad[i] = error;
            } else {
                grad[i] = huber_delta * Math.signum(error);
            }
        }
    }

    final public void mae_loss(double [] omigaSumBuffer,
                                double[] predict, 
                                double[] loss, 
                                double[] grad, 
                                Sample sample, 
                                FeatureInfo[] featureInfos, 
                                boolean updateLoss) {
        // MAE损失计算
        for (int i = 0; i < paraSize; ++i) {
            // 线性预测
            predict[i] = omigaSumBuffer[i];
            double y_true = sample.label;
            
            // 预测误差
            double error = predict[i] - y_true;
            
            if (updateLoss) {
                // MAE损失计算
                loss[i] = Math.abs(error);
            }
            
            // MAE梯度计算 (误差符号)
            grad[i] = (error > 0) ? 1 : (error < 0) ? -1 : 0;
        }
    }

    //修改点：预测时增加一个WCE分支（5.30）
    //predict the value using the calculated feature weights, and generate the bias with the actual value.
    final public void predict(Sample sample, double[] predict, double[] loss, double[] grad, FeatureInfo[] featureInfos,
            boolean updateLoss) {
        double [] omigaSumBuffer = new double[paraSize];
        for (FeatureInfo featureInfo : featureInfos) {
            if (featureInfo == null) continue;
            Double feature_value = featureInfo.feature_value;
            for (int i = 0; i < paraSize; ++i) {
                omigaSumBuffer[i] += featureInfo.omiga[i] * feature_value;
            }
        }
        switch (loss_func) {
            case "mse":
                mse_loss(omigaSumBuffer, predict, loss, grad, sample, featureInfos, updateLoss);
                break;
            case "cross_entropy":
                cross_entropy_loss(omigaSumBuffer, predict, loss, grad, sample, featureInfos, updateLoss);
                break;
            case "wce":
                wce_loss(omigaSumBuffer, predict, loss, grad, sample, featureInfos, updateLoss);
                break;
            case "msle":
                msle_loss(omigaSumBuffer, predict, loss, grad, sample, featureInfos, updateLoss);
                break;
            case "huber_loss":
                huber_loss(omigaSumBuffer, predict, loss, grad, sample, featureInfos, updateLoss);
                break;
            case "mae":
                mae_loss(omigaSumBuffer, predict, loss, grad, sample, featureInfos, updateLoss);
                break;
            default:
                break;
        }
    }

    public void dump(boolean trimModel) throws IOException {
        model.dump(trimModel);
        aucCollector.dump(trimModel);
        System.out.println("total features:" + model.features.size());
        System.out.println("Dump model and predict finish.");
    }

    public static class RunParser implements Runnable {

        public Sample sample;
        volatile public int status = 1;
        private DataSource dataSource;
        private LocalParaTrainer trainer;
        private String line;
        public RunParser(DataSource dataSource, LocalParaTrainer trainer, String line) {
            this.dataSource = dataSource;
            this.line = line;
            this.trainer = trainer;
        }

        public void run() {
            status = 2;
            try {   
                sample = dataSource.extracSample(line);
                if (sample != null) {
                    trainer.prepareSampleFeature(sample);
                    status = 0;
                    return;
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
            status = -1;

        }

    }

    public void prepareSampleFeature(Sample sample) {
        //featureInfos在这里初始化
        sample.featureInfos = new FeatureInfo[sample.strFeatures.size()];
        sample.n = new double[sample.featureInfos.length];
        sample.sqrtN = new double[sample.featureInfos.length];
        for (int i = 0; i < sample.featureInfos.length; ++i) {
            String feature = sample.strFeatures.get(i);
            sample.featureInfos[i] = model.features.get(feature);
            //如果是连续特征，把特征值存起来，离散特征都是 1.0
            if(sample.continuousFeatures.containsKey(feature) && sample.featureInfos[i] != null){
                sample.featureInfos[i].feature_value = sample.continuousFeatures.get(feature);
                //打印 feature 和 feature_value
                //System.out.println("continuousFeatures: " + feature + "\t" + sample.featureInfos[i].feature_value);
            }
        }
        double[] loss = new double[modelConfig.paraSize()];
        double[] local_predict = new double[paraSize];
        double[] local_grad = new double[paraSize];
        predict(sample, local_predict, loss, local_grad, sample.featureInfos, true);
        lossCollector.feed(loss);
        for (int j = 0; j < modelConfig.paraSize(); ++j) {
            aucCollector.feed(new AUCFood(local_predict[j], sample.label, j, sample.key));
        }
        notRealCollector.feed(this);
    }

    @Override
    public void run() {
        RunParser runParser = null;
        long curr = System.currentTimeMillis();
        while (running || !trainDataQueue.isEmpty()) {
            while ((runParser = trainDataQueue.peek()) != null) {
                while (true) {
                    if (runParser.status == 1 || runParser.status == 2) {
                        // timeCollector.feed(new
                        // TimeFood(1,System.currentTimeMillis()-curr));
                        // curr = System.currentTimeMillis();
                        continue;
                    } else if (runParser.status == 0) {
                        switch (loss_func) {
                            case "mse":
                            case "wce":
                            case "msle":
                            case "huber_loss":
                            case "mae":
                                optimizeFtrlNative(runParser.sample);
                                break;
                            case "cross_entropy":
                                optimize(runParser.sample);
                                break;
                            default:
                                System.err.println("Unknown loss_type: " + loss_func);
                                break;
                        }
                        break;
                    } else {
                        break;
                    }
                }
                trainDataQueue.poll();
            }
            try {
                Thread.sleep(1);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

    }

    strictfp static float floatSqrt(float x) {
        float xhalf = 0.5f * x;
        int i = Float.floatToRawIntBits(x); // convert integer to keep the
                                            // representation IEEE 754
        i = 0x5f3759df - (i >> 1);
        x = Float.intBitsToFloat(i);
        x = x * (1.5f - xhalf * x * x);
        return 1 / x;
    }

    strictfp public static double doubleSqrt(double x) {
        double xhalf = 0.5d * x;
        long i = Double.doubleToLongBits(x);
        i = 0x5fe6ec85e7de30daL - (i >> 1);
        x = Double.longBitsToDouble(i);
        x *= (1.5d - xhalf * x * x);
        return 1 / x;
    }

    public void printMsg() {
        System.err.printf("Size feature %d.\n", model.features.size());
    }
    // 用于分类的FTRL优化，采用正负样本
    private void optimize(Sample sample) {
        for (int i = 0; i < sample.strFeatures.size(); ++i) {
            FeatureInfo featureInfo = sample.featureInfos[i];
            if (featureInfo == null) {
                String fName = sample.strFeatures.get(i);
                FeatureInfo fi = model.features.get(fName);
                if (fi == null) {
                    if (featureIncludeRate == 0 || Math.random() <= featureIncludeRate) {
                        featureInfo = new FeatureInfo(paraSize);
                        //如果是连续特征，把特征值存起来，离散特征都是 1.0
                        if(sample.continuousFeatures.containsKey(fName)){
                            featureInfo.feature_value = sample.continuousFeatures.get(fName);
                        }
                        model.features.put(fName, featureInfo);
                        sample.featureInfos[i] = featureInfo;
                        // featureCollector.feed(fName, hash);
                    }
                    continue;
                } else {
                    if(sample.continuousFeatures.containsKey(fName)){
                        fi.feature_value = sample.continuousFeatures.get(fName);
                    }
                    sample.featureInfos[i] = fi;
                    featureInfo = fi;
                }
            }
            if (featureInfo.positive > 0 && featureInfo.negtive > 0) {
                long featureCnt = featureInfo.positive + featureInfo.negtive;
                double n = featureInfo.positive * 1.0f * featureInfo.negtive / featureCnt;
                sample.n[i] = n;
                sample.sqrtN[i] = doubleSqrt(n);
            } else {
                sample.n[i] = sample.sqrtN[i] = 0;
            }

            for (int m = 0; m < modelConfig.paraSize(); ++m) {
                //Parameter parameter = modelConfig.getPara(m);
                Parameter parameter = modelConfig.getPkgParameter(sample.demand_pkgname);
                if (featureInfo.z[m] <= parameter.lambda1 && featureInfo.z[m] >= -parameter.lambda1) {
                    featureInfo.omiga[m] = 0;
                } else {
                    double rst = -1 / ((parameter.beta + sample.sqrtN[i]) / parameter.alpha + parameter.lambda2)
                            * (featureInfo.z[m] - Math.signum(featureInfo.z[m]) * parameter.lambda1);
                    //omiga is the feature weight
                    featureInfo.omiga[m] = rst;
                }
            }
        }
        //calcuate gradient of loss(grad)
        predict(sample, predict, null, grad, sample.featureInfos, false);

        for (int i = 0; i < sample.featureInfos.length; ++i) {
            FeatureInfo featureInfo = sample.featureInfos[i];
            if (featureInfo == null) {
                continue;
            }
            Double feature_value = featureInfo.feature_value;
            for (int j = 0; j < modelConfig.paraSize(); ++j) {
                //连续特征的梯度需要乘上特征值，离散特征不变（乘以1.0)
                grad[j] = grad[j] * feature_value;
                delta[j] = (doubleSqrt(sample.n[i] + grad[j] * grad[j]) - sample.sqrtN[i]) / modelConfig.getPara(j).alpha;
                featureInfo.z[j] += grad[j] - delta[j] * featureInfo.omiga[j];
            }
            if (sample.label == 1) {
                featureInfo.positive += 1;
            } else {
                featureInfo.negtive += 1;
            }
        }
    }

    // 新增：用于回归的FTRL（新）
    private void optimizeFtrlNative(Sample sample) {
        // 1. 确保所有特征都有初始化（featureInfo）
        for (int i = 0; i < sample.strFeatures.size(); ++i) {
            String fName = sample.strFeatures.get(i);
            FeatureInfo featureInfo = sample.featureInfos[i];

            if (featureInfo == null) {
                FeatureInfo fi = model.features.get(fName);
                if (fi == null) {
                    if (featureIncludeRate == 0 || Math.random() <= featureIncludeRate) {
                        featureInfo = new FeatureInfo(paraSize);
                        model.features.put(fName, featureInfo);
                        sample.featureInfos[i] = featureInfo;
                    } else {
                        continue;
                    }
                } else {
                    sample.featureInfos[i] = fi;
                    featureInfo = fi;
                }
            }

            // 使用 featureInfo.n 计算权重 omiga
            for (int m = 0; m < modelConfig.paraSize(); ++m) {
                Parameter parameter = modelConfig.getPkgParameter(sample.demand_pkgname);
                double n_sqrt = Math.sqrt(featureInfo.n[m]);

                if (Math.abs(featureInfo.z[m]) <= parameter.lambda1) {
                    featureInfo.omiga[m] = 0;
                } else {
                    double denom = (parameter.beta + n_sqrt) / parameter.alpha + parameter.lambda2;
                    double sign = Math.signum(featureInfo.z[m]);
                    featureInfo.omiga[m] = -1.0 / denom * (featureInfo.z[m] - sign * parameter.lambda1);
                }
            }
        }

        // 2. 计算当前样本的预测梯度
        predict(sample, predict, null, grad, sample.featureInfos, false);

        // 3. 使用梯度更新 n 和 z
        for (int i = 0; i < sample.featureInfos.length; ++i) {
            FeatureInfo featureInfo = sample.featureInfos[i];
            if (featureInfo == null) continue;

            for (int j = 0; j < modelConfig.paraSize(); ++j) {
                double g = grad[j]; // 当前特征的梯度
                double n_old = featureInfo.n[j];
                double n_new = n_old + g * g;
                featureInfo.n[j] = n_new;

                double sigma = (Math.sqrt(n_new) - Math.sqrt(n_old)) / modelConfig.getPara(j).alpha;
                featureInfo.z[j] += g - sigma * featureInfo.omiga[j];
                featureInfo.label_sum += sample.label;
                featureInfo.sample_cnt += 1;
            }
        }
    }



    public void train(String dataPath, String name) throws IOException {
        prepareTrain(dataPath, name);
        running = true;
        Thread trainThread = new Thread(this);
        trainThread.setPriority(Thread.MAX_PRIORITY);
        trainThread.start();
        LineReader reader = getLineReader(dataPath);
        String line = null;
        while ((line = reader.readLine()) != null) {
            RunParser runParser = new RunParser(dataSource, this, line);
            try {
                trainDataQueue.put(runParser);
            } catch (InterruptedException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
            cachedThreadPool.execute(runParser);
        }
        reader.close();
        running = false;
        System.out.println("Waiting trainDataQueue tobe empty...");
        long i = 0;
        while (!trainDataQueue.isEmpty()) {
            try {
                Thread.sleep(1000);
                if (++i % 100 == 0) {
                    System.out.println("trainDataQueue size is " + trainDataQueue.size());
                }
            } catch (InterruptedException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        printMsg();
        dump(true);
    }

    public LineReader getLineReader(String path) throws IOException {
        return new LocalLineReader(path);
    }

}
