package org.apache.mahout.cf.taste.impl.recommender;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Scanner;
import java.util.stream.Stream;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

public class ParseFile {
	public JSONObject jsonObject=null ;
	//private String file_knn= "C:\\Users\\Moham\\PycharmProjects\\knng_pfe-master\\scripts_py\\KNNG_FULL_GRADIENT_DECENT\\1557358837\\KNNG_FULL_DECENT.txt";


	public ParseFile(String file_path) throws FileNotFoundException, IOException{
		JSONParser parser = new JSONParser();
		 try {
		     Object obj = parser.parse(new FileReader(file_path));

		     this.jsonObject = (JSONObject)obj ;

		 }catch (Exception e) {
		}
	}
	
	
	public ParseFile() {
	}
	
	public void convertToCSV(String nameTextFile) {
		final Path path = Paths.get("path", "to", "folder");
	    final Path txt = path.resolve(System.getProperty("user.dir")+"\\Datasets\\"+nameTextFile+".data");
	    final Path csv = path.resolve(System.getProperty("user.dir")+"\\Datasets\\"+nameTextFile+".csv");
	    System.out.println(System.getProperty("user.dir")+"\\Datasets\\"+nameTextFile+".csv");
	    final Charset utf8 = Charset.forName("UTF-8");

	    try (
	    		final Scanner scanner = new Scanner(Files.newBufferedReader(txt, utf8));
	            final PrintWriter pw = new PrintWriter(Files.newBufferedWriter(csv, utf8, StandardOpenOption.CREATE))
	        ) {
		        	while (scanner.hasNextLine()) {
		        		pw.println(scanner.nextLine().replace(' ', ','));
		        	}
	          } catch (IOException e) {
				e.printStackTrace();
			}
	}
	
	public JSONObject getJsonObject() {
		return jsonObject;
	}

	public void setJsonObject(JSONObject jsonObject) {
		this.jsonObject = jsonObject;
	}

}
