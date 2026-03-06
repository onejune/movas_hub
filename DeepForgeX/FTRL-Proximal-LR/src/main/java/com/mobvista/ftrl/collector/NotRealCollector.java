package com.mobvista.ftrl.collector;

import java.io.IOException;

import com.mobvista.ftrl.train.LocalParaTrainer;

public class NotRealCollector extends BaseCollector<Object> {
	LocalParaTrainer trainer;
	public NotRealCollector(LocalParaTrainer trainer){
		this.trainer = trainer;
	}
	@Override
	public void collect(Object food) {
		return;
	}
	@Override
	public void init() {
		// TODO Auto-generated method stub
		super.init();
	}

	@Override
	public void dump(boolean fullDump) throws IOException {
		System.err.println(basicInfo().append("\tSize of feature ")
				.append(trainer.model.features.size())
				.append(".\n\tLength of trainning queue ")
				.append(trainer.trainDataQueue.size())
				.append('.').toString());
	}

}
