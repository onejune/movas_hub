package com.mobvista.ftrl.collector;

import java.io.IOException;

import com.mobvista.ftrl.collector.food.TimeFood;

public class TimeCollector extends BaseCollector<TimeFood> {
	long[] waits = new long[3];

	@Override
	public void collect(TimeFood food) {
		waits[food.idx]+=food.time;
		
	}
	@Override
	public void init() {
		// TODO Auto-generated method stub
		super.init();
	}
	
	public boolean addCount(TimeFood food){
		if(food.idx==0 || food.idx==2){
			cnt+=1;
			return true;
		}
		return false;
	}


	@Override
	public void dump(boolean fullDump) throws IOException {
		System.err.println(basicInfo().append("\tTrainning ")
				.append(waits[0]).append(" ms, Waiting ")
				.append(waits[1]).append(" ms,Wrong parsing ")
				.append(" ms.").toString());
		waits = new long[3];
	}

}
