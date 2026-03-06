package com.mobvista.ftrl.collector;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.concurrent.LinkedBlockingQueue;

import com.mobvista.ftrl.util.Constants;

public abstract class BaseCollector<T> {
	
	public static SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyyMMddHHmmss");
	
	public static String getTime(){
		return simpleDateFormat.format(new Date());
	}
	
	public int cnt=0;
	LinkedBlockingQueue<T> queue = new LinkedBlockingQueue<T>(10000);
	
	final public void feed(T food) {
		try {
			queue.add(food);
		} catch (IllegalStateException e) {
			//e.printStackTrace();
		}
	}
	
	public void init() {
		new Thread(new Runnable(){

			boolean refresh=false;
			public void run() {
				while (true) {
					try {
						T food = queue.take();
						collect(food);
						if(addCount(food)){
							refresh=true;
						}
						if(time2dump() && refresh){
							try {
								dump(false);
								refresh=false;
							} catch (IOException e) {
								e.printStackTrace();
							}
						}
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				}
			}
			
		}).start();
	}
	
	StringBuilder basicInfo(){
		StringBuilder sb = new StringBuilder();
		sb.append('\n');
		sb.append(this.getClass().getSimpleName());
		sb.append(" at ");
		sb.append(getTime());
		sb.append(" (Queue size is ");
		sb.append(queue.size());
		sb.append(")\n");
		return sb;
	}
	
	protected abstract void collect(T food);
	
	public boolean time2dump(){
		return (cnt%Constants.PRINT_WIN)==0;
	}
	
	public boolean addCount(T food){
		cnt+=1;
		return true;
	}
	
	public abstract void dump(boolean fullDump)throws IOException;
 
}
