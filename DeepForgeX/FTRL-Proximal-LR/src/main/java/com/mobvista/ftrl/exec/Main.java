package com.mobvista.ftrl.exec;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.Options;

import com.mobvista.ftrl.datasource.DataSource;
import com.mobvista.ftrl.datasource.FeatureDataSource;
import com.mobvista.ftrl.datasource.SingleFeatureDataSource;
import com.mobvista.ftrl.train.LocalParaTrainer;
import com.mobvista.ftrl.train.S3ParaTrainer;
import com.mobvista.ftrl.train.Trainer;

@SuppressWarnings("deprecation")
public class Main {
	
	String input;
	String confBase;
	DataSource dataSource;
	String name="";
	
	@SuppressWarnings("rawtypes")
	static Map<String,Class> formats = new HashMap<String,Class>();
	{
		formats.put("f", SingleFeatureDataSource.class);
		//formats.put("f", FeatureDataSource.class);
	}
    
    private  Options getOption(){
		Options options = new Options();
		// add by online learning pipeline
		options.addOption("i", true, "Input train data");
		options.addOption("c", true, "Config file dir");
		options.addOption("n", true, "The name of this train process");
		options.addOption("f", true, "The format of input data source.  \"f\" the feature format only.");
		return options;
	}
    
    public boolean processArgs(String[] args) throws Exception{
		Options options = getOption();
		CommandLineParser parser = new BasicParser();
		CommandLine commandLine = parser.parse(options, args);
		
		input = commandLine.getOptionValue("i");
		name = commandLine.getOptionValue('n');
		confBase = commandLine.getOptionValue("c");
		if(!confBase.endsWith(File.separatorChar+"")){
			confBase+=File.separatorChar;
		}

        String currentDir = System.getProperty("user.dir");
        System.out.println("Current Directory: " + currentDir); 
		String format = commandLine.getOptionValue("f");
        System.out.println("args: input = " + input + ", confBase = " + confBase + ", name = " + name + ", format = " + format);

		if(!formats.containsKey(format)){
			return false;
		}
		dataSource = (DataSource)formats.get(format).newInstance();
		return dataSource.init(confBase);
		
    }
    
    public void run() throws Exception{
		Trainer trainer;
		//trainer = new S3ParaTrainer();
        trainer = new LocalParaTrainer();
		System.out.println("Parallel train.");
		trainer.init(confBase, dataSource);
		trainer.train(input,name);
    }

	public static void main(String[] args) throws Exception {
		try {
			Main runner = new Main();
			if (!runner.processArgs(args)) {
				System.out.println("Init Main failed!");
				return;
			}
            System.out.println("Init Main succeed!");
			runner.run();
			System.exit(0);
		} catch (Exception e) {
			System.exit(-1);
		}
	}
}
