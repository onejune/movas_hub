package com.mobvista.ftrl.tools;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import com.mobvista.ftrl.collector.AucCollector;
import com.mobvista.ftrl.collector.food.AUCFood;
import com.mobvista.ftrl.datasource.SingleFeatureDataSource;
import com.mobvista.ftrl.linereader.LocalLineReader;
import com.mobvista.ftrl.struct.FeatureInfo;
import com.mobvista.ftrl.struct.ModelText;
import com.mobvista.ftrl.struct.Sample;
import com.mobvista.ftrl.util.ToolFucs;

public class OnlineReplayCCTM {

    // 声明为静态变量
    private static String loss_func = "cross_entropy";

    public static double calculateCTR(double sum) {
        switch (loss_func) {
            case "cross_entropy":
                return 1 / (1 + Math.exp(-1 * sum)); // p = sigmoid(z)
            case "mse":
                return sum; // p = z
            case "wce":
                return Math.exp(sum); // p = e ^ z
            default:
                System.err.println("[Error] - 推理loss设置异常");
                return -1;
        }
    }

    public static void main(String[] args) throws ParseException, IOException {
        Options options = new Options();
        int[] errCnt = {0};

        try {
            options.addOption("model_ctr", true, "ctr_model_path");
            options.addOption("model_cvr", true, "cvr_model_path");
            options.addOption("conf", true, "data config path");
            options.addOption("data", true, "data_path");
            options.addOption("out", true, "output dir");
            options.addOption("name", true, "validation tag");

            @SuppressWarnings("deprecation")
            CommandLineParser parser = new BasicParser();
            CommandLine commandLine = parser.parse(options, args);

            // 读取loss_func配置（使用Properties类）
            try {
                Properties prop = new Properties();
                prop.load(new FileInputStream(commandLine.getOptionValue("conf")));
                if (prop.getProperty("loss_func") != null) {
                    loss_func = String.valueOf(prop.getProperty("loss_func"));
                }
            } catch (IOException e) {
                System.err.println("-推理时：Failed to read conf, using default loss_func: cross_entropy");
            }

            // 验证loss_func值
            if (!"cross_entropy".equals(loss_func) && !"mse".equals(loss_func) && !"wce".equals(loss_func)) {
                throw new RuntimeException("-inference Unsupported loss function: " + loss_func +
                                          ". Allowed values: cross_entropy, mse");
            }

            SingleFeatureDataSource source = new SingleFeatureDataSource();
            source.init(commandLine.getOptionValue("conf"));

            if (!ToolFucs.confirmDir(commandLine.getOptionValue("out"))) {
                throw new RuntimeException("invalid output dir " + commandLine.getOptionValue("out"));
            }
            AucCollector collector = new AucCollector(commandLine.getOptionValue("out"), ToolFucs.singleParaConfig());
            collector.init(ToolFucs.singleParaConfig(), loss_func);
            collector.setName(commandLine.getOptionValue("name"));

            ModelText model_ctr = new ModelText(ToolFucs.singleParaConfig(), 0, commandLine.getOptionValue("model_ctr"), loss_func);
            model_ctr.load(commandLine.getOptionValue("model_ctr"));
            System.out.println("CTR model load completely!");
            ToolFucs.PrintCurrentTime();

            ModelText model_cvr = new ModelText(ToolFucs.singleParaConfig(), 0, commandLine.getOptionValue("model_cvr"), loss_func);
            model_cvr.load(commandLine.getOptionValue("model_cvr"));
            System.out.println("CVR model load completely!");
            ToolFucs.PrintCurrentTime();

            LocalLineReader reader = new LocalLineReader(commandLine.getOptionValue("data"));
            // 使用有界阻塞队列控制内存占用
            BlockingQueue<Runnable> workQueue = new ArrayBlockingQueue<>(3000); // 控制队列大小
            ExecutorService executor = new ThreadPoolExecutor(
                32, // 核心线程数
                32, // 最大线程数
                60, TimeUnit.SECONDS,
                workQueue
            );
            String line;
            while ((line = reader.readLine()) != null) {
                final String currentLine = line;
                try {
                    // 提交任务时如果队列满，则阻塞等待
                    executor.submit(() -> {
                        Sample sample = source.extracSample(currentLine);
                        if (sample == null) {
                            handleError(errCnt);
                            return null;
                        }

                        float sum_ctr = 0;
                        float sum_cvr = 0;
                        try {
                            for (String feature : sample.strFeatures) {
                                FeatureInfo fi_ctr = model_ctr.features.get(feature);
                                if (fi_ctr != null) {
                                    sum_ctr += fi_ctr.omiga[0];
                                }

                                FeatureInfo fi_cvr = model_cvr.features.get(feature);
                                if (fi_cvr != null) {
                                    sum_cvr += fi_cvr.omiga[0];
                                }
                            }
                        } catch (Exception e) {
                            handleError(errCnt);
                            return null;
                        }

                        double ctr = calculateCTR(sum_ctr);
                        double cvr = calculateCTR(sum_cvr);
                        double pivr = ctr * cvr;
                        collector.collect(new AUCFood(pivr, sample.label, 0, sample.key));
                        return null;
                    });
                } catch (RejectedExecutionException e) {
                    // 队列满时处理策略（如记录日志、等待后重试）
                    Thread.sleep(1000);
                }
            }
            executor.shutdown();
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);

            System.out.println("PIVR predict completely!");
            ToolFucs.PrintCurrentTime();

            collector.dump(true);
            System.out.println("Exception data count: " + errCnt[0]);
            ToolFucs.PrintCurrentTime();
            System.exit(0);
        } catch (Exception e) {
            e.printStackTrace();
            HelpFormatter hf = new HelpFormatter();
            hf.printHelp("Online replay usage:", options, true);
        }
    }

    private static void handleError(int[] errCnt) {
        synchronized (OnlineReplay.class) {
            errCnt[0]++;
        }
    }
}



