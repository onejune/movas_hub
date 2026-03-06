package com.mobvista.ftrl.collector;

import java.io.IOException;

import com.mobvista.ftrl.struct.ModelConfig;
import com.mobvista.ftrl.util.Constants;

public class LossCollector extends BaseCollector<double[]> {
	double[] allLoss;
	double[] windowLoss;
	ModelConfig conf;
	@Override
	public void collect(double[] food) {
		for(int i =0;i<food.length;++i){
			windowLoss[i]+=food[i];
		}
	}
	public LossCollector(ModelConfig conf){
		this.conf = conf;
	}
	
	public void init(){
		allLoss = new double[conf.paraSize()];
		windowLoss = new double[conf.paraSize()];
		super.init();
	}

	@Override
	public void dump(boolean fullDump) throws IOException {
		for (int j = 0; j < conf.paraSize(); ++j) {
			double averageLoss = windowLoss[j] / Constants.PRINT_WIN;
			allLoss[j] += windowLoss[j];
			windowLoss[j] = 0;
			System.err.println(
					basicInfo().append("\t").append(conf.getName(j))
					.append(": WinAverageLoss=")
					.append(averageLoss)
					.append(", AllAverageLoss=")
					.append(allLoss[j] / cnt)
					.append("\n\tsamples:")
					.append(cnt / 10000.0).append('w'));
		}
		
	}

}
