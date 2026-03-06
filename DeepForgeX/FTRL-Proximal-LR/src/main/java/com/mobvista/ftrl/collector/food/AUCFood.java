package com.mobvista.ftrl.collector.food;

public class AUCFood {
		public double predict;
		public double label;
		public int index;
		public String key;
		public AUCFood(double pre,double lab,int ind,String key){
			predict = pre;
			label = lab;
			index = ind;
			this.key = key;
		}
}
