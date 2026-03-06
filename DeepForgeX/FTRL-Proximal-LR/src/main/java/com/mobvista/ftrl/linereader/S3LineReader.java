package com.mobvista.ftrl.linereader;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.SocketException;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;

import org.apache.commons.lang3.StringUtils;

import com.amazonaws.auth.AWSCredentials;
import com.amazonaws.auth.BasicAWSCredentials;
import com.amazonaws.services.s3.AmazonS3Client;
import com.amazonaws.services.s3.model.ObjectListing;
import com.amazonaws.services.s3.model.S3Object;
import com.amazonaws.services.s3.model.S3ObjectSummary;

public class S3LineReader implements LineReader {
	
	public static class BucAndKey{
		String bucketName;
		String key;
		public BucAndKey(String buc,String key){
			this.bucketName = buc;
			this.key = key;
		}
	}
	public String AWS_ACCESS_KEY_ID="xxxx";
	public String AWS_SECRET_ACCESS_KEY="xxxx";
	AmazonS3Client s3Client;
	BufferedReader reader;
	S3Object s3Obj;
	List<BucAndKey> s3keys = new ArrayList<BucAndKey>();
	String s3Key;
	String bucketName;
	public S3LineReader(String fullS3Pathes) throws IOException{
		String[] pathes = StringUtils.split(fullS3Pathes, ';');
		for (String fullS3Path : pathes) {
			int idx = fullS3Path.indexOf("//");
			if (idx < 0) {
				throw new RuntimeException("invalid s3 path " + fullS3Path);
			}
			String fullPath = fullS3Path.substring(idx + 2);
			idx = fullPath.indexOf('/');
			if (idx < 0) {
				throw new RuntimeException("invalid s3 path " + fullS3Path);
			}
			String buc = fullPath.substring(0, idx);
			String path = fullPath.substring(idx + 1);
			AWSCredentials credentials = null;
			//credentials = new BasicAWSCredentials(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY);
			//s3Client = new AmazonS3Client(credentials);
			s3Client = new AmazonS3Client();
			if (s3Client.doesObjectExist(buc, path)) {
				s3keys.add(new BucAndKey(buc,path));
			} else {
				ObjectListing listObj = s3Client.listObjects(buc, path);
				for (S3ObjectSummary smy : listObj.getObjectSummaries()) {
					String key = smy.getKey();
					String subfolder = key.substring(path.length());
					if(subfolder.charAt(0)== '/'){
						subfolder = subfolder.substring(1);
					}
					if(subfolder.indexOf('/')>0){
						continue;
					}
					if (s3Client.doesObjectExist(buc, key) && !key.endsWith("$folder$")) {
						System.out.println("S3LineReader add s3 key:" + key);
						s3keys.add(new BucAndKey(buc,smy.getKey()));
					}
				}
			}
		}
		if(s3keys.size()>0){
			refreshReader(false);
		}else{
			throw new IOException("No available data in "+fullS3Pathes);
		}
	}
	
	public boolean refreshReader(boolean reRead) throws IOException{
		while (!s3keys.isEmpty()) {
			close();
			if(!reRead){
				BucAndKey bk = s3keys.remove(0);
				s3Key = bk.key;
				bucketName = bk.bucketName;
			}
			if(s3Key.endsWith("/_SUCCESS")){
				continue;
			}
			s3Obj = s3Client.getObject(bucketName, s3Key);
			if(s3Obj==null){
				continue;
			}
			if(s3Key.endsWith(".gz")){
				reader = new BufferedReader(new InputStreamReader(new GZIPInputStream(s3Obj.getObjectContent())));
			}else{
				reader = new BufferedReader(new InputStreamReader(s3Obj.getObjectContent()));
			}
			System.out.println("S3LineReader switch to " + s3Obj.getKey());
			return true;
		}
		close();
		return false;
	}
	
	public String readLine() throws IOException {
		String line=null;
		do{
			try {
				line = reader.readLine();
				if (line != null) {
					return line;
				} else {
					close();
					if (!refreshReader(false)) {
						return null;
					}
				}
			} catch (IOException e) {
				close();
				if (!refreshReader(true)) {
					System.out.println("Reconnect fail.");
					return null;
				}
			}

		} while (true);
	}
	public void close() throws IOException {
		if(s3Obj!=null){
			s3Obj.close();	
			s3Obj=null;
		}
		if(reader!=null){
			reader.close();
			reader=null;
		}
	}
	
	public static void main(String[] args) throws IOException {
		S3LineReader reader  = new S3LineReader("s3://mob-emr-test/wanjun/m_sys_model/testapi;s3://mob-emr-test/wanjun/m_sys_model/testapi/dir/");
		//S3LineReader reader  = new S3LineReader(args[0]);
		//S3LineReader reader  = new S3LineReader("s3://mob-emr-test/wanjun/m_sys_model/gztest.gz");
		String line  = null;
		int cnt=0;
		while ((line = reader.readLine())!=null){
			System.out.println(line);
			if(cnt++>20){
				break;
			}
		}		reader.close();
	}
	

}
