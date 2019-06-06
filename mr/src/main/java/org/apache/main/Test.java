package org.apache.main;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

public class Test {
	
	Process mProcess;

	public void runScript(){
	    Process process;
	    try{
	          process = Runtime.getRuntime().exec(new String[]{System.getProperty("user.dir")+"\\py_scripts\\dist\\graph_learning",System.getProperty("user.dir")+"\\Datasets\\TestSet0.data"});
	          mProcess = process;
	    }catch(Exception e) {
	       System.out.println("Exception Raised" + e.toString());
	    }
	    InputStream stdout = mProcess.getInputStream();
	    BufferedReader reader = new BufferedReader(new InputStreamReader(stdout,StandardCharsets.UTF_8));
	    String line;
	    try{
	       while((line = reader.readLine()) != null){
	            System.out.println("stdout: "+ line );
	       }
	    }catch(IOException e){
	          System.out.println("Exception in reading output"+ e.toString());
	    }
	}


	public static void main(String[] args) {
		Test scriptPython = new Test();
        scriptPython.runScript();

	}

}
