package org.apache.main;


import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.eval.Fold;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.ParseFile;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhoodImpl;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.cf.taste.similarity.UserSimilarityIml;

public class MyRecommenderBuilder implements RecommenderBuilder{

	ParseFile p;
	Process mProcess;
	String rootPath= System.getProperty("user.dir") ;
	public MyRecommenderBuilder(ParseFile p) throws IOException {
		this.p = p;
	}
	
	public MyRecommenderBuilder() {
	}

	@Override
	public Recommender buildRecommender(DataModel dataModel) throws TasteException {
		return null ;
		
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public Recommender buildRecommender(DataModel dataModel, Fold fold) throws TasteException {
		
		String generatedKnngFilePath="\\Datasets\\KNNG_LBNN.txt" ;
		String generatedTrainFilePath="\\Datasets\\fold_train.data" ;
		String generatedTestFilePath="\\Datasets\\fold_test.data" ;

		Object[] arrays= buildTrain(fold.getTraining()) ;
		Object[] arraysTests= buildTest(fold.getTesting()) ;
		

		//for training building
		ArrayList<Long> usersIDs= (ArrayList<Long>) arrays[0] ;
		ArrayList<ArrayList<Long>> usersItems= (ArrayList<ArrayList<Long>>) arrays[1] ;
		ArrayList<ArrayList<Float>> itemsRatings= (ArrayList<ArrayList<Float>>) arrays[2] ;
		
		//for test buildings
		ArrayList<Long> t_usersIDs= (ArrayList<Long>) arraysTests[0] ;
		ArrayList<ArrayList<Long>> t_usersItems= (ArrayList<ArrayList<Long>>) arraysTests[1] ;
		ArrayList<ArrayList<Float>> t_itemsRatings= (ArrayList<ArrayList<Float>>) arraysTests[2] ;
		
		printDataToTabDelimitedFile(t_usersIDs, t_usersItems, t_itemsRatings, generatedTestFilePath );

		printDataToTabDelimitedFile(usersIDs, usersItems, itemsRatings, generatedTrainFilePath );
		
		System.out.println("-------------------- fold passage ----------------------");

		System.out.println("Before lunching pyhton file");
		runScript(rootPath+generatedTrainFilePath);
		System.out.println("After lunching pyhton file");
		
		try {
			p = new ParseFile(rootPath+generatedKnngFilePath);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		  
		UserNeighborhood neighborhood= new UserNeighborhoodImpl(p.getJsonObject()) ;
		UserSimilarity similarity = new UserSimilarityIml(p.getJsonObject()) ;
		UserBasedRecommender recommender = new GenericUserBasedRecommender(dataModel, neighborhood, similarity);
		return recommender;
	}
	
	
	Object[] buildTrain(DataModel fold) throws TasteException {
		
		@SuppressWarnings("deprecation")
		DataModel data_training=new GenericDataModel(fold);
		LongPrimitiveIterator it_userIds= data_training.getUserIDs() ;
		LongPrimitiveIterator it_itemIds = null ;
		ArrayList<Long> usersIDs =new ArrayList<Long>() ;
		ArrayList<ArrayList<Long>> usersItems =new ArrayList<ArrayList<Long>>() ;
		ArrayList<ArrayList<Float>> itemsRatings =new ArrayList<ArrayList<Float>>() ;
		long myUser, myItem ;
		float rating_item_user;
		ArrayList<Long> items=null ;
		ArrayList<Float> ratings=null ;
		FastIDSet item_Ids=null ;
		while(it_userIds.hasNext()) {
			myUser= it_userIds.nextLong() ;
			usersIDs.add(myUser) ;
			//System.out.println("user: "+myUser+"\t");
			item_Ids=data_training.getItemIDsFromUser(myUser) ;
			it_itemIds = item_Ids.iterator() ;
			items =new ArrayList<Long>() ;
			ratings =new ArrayList<Float>() ;

			while(it_itemIds.hasNext()) {
				myItem= it_itemIds.nextLong() ;
				items.add(myItem);
				rating_item_user=data_training.getPreferenceValue(myUser, myItem) ;
				ratings.add(rating_item_user) ;
				//System.out.println(myUser+"\t"+myItem+"\t"+rating_item_user);
			}
			usersItems.add(items) ;
			itemsRatings.add(ratings);
			//System.out.println("items= "+items.size()+" ratings= "+ratings.size());
		}
		
		Object[] myArrays=new Object[3] ;
		myArrays[0]=usersIDs ;
		myArrays[1]=usersItems ;
		myArrays[2]=itemsRatings ;
		return myArrays;
	}
	
	
Object[] buildTest(FastByIDMap<PreferenceArray> fold) throws TasteException {
		
		DataModel data_training=new GenericDataModel(fold);
		LongPrimitiveIterator it_userIds= data_training.getUserIDs() ;
		LongPrimitiveIterator it_itemIds = null ;
		ArrayList<Long> usersIDs =new ArrayList<Long>() ;
		ArrayList<ArrayList<Long>> usersItems =new ArrayList<ArrayList<Long>>() ;
		ArrayList<ArrayList<Float>> itemsRatings =new ArrayList<ArrayList<Float>>() ;
		long myUser, myItem ;
		float rating_item_user;
		ArrayList<Long> items=null ;
		ArrayList<Float> ratings=null ;
		FastIDSet item_Ids=null ;
		while(it_userIds.hasNext()) {
			myUser= it_userIds.nextLong() ;
			usersIDs.add(myUser) ;
			//System.out.println("user: "+myUser+"\t");
			item_Ids=data_training.getItemIDsFromUser(myUser) ;
			it_itemIds = item_Ids.iterator() ;
			items =new ArrayList<Long>() ;
			ratings =new ArrayList<Float>() ;

			while(it_itemIds.hasNext()) {
				myItem= it_itemIds.nextLong() ;
				items.add(myItem);
				rating_item_user=data_training.getPreferenceValue(myUser, myItem) ;
				ratings.add(rating_item_user) ;
				//System.out.println(myUser+"\t"+myItem+"\t"+rating_item_user);
			}
			usersItems.add(items) ;
			itemsRatings.add(ratings);
			//System.out.println("items= "+items.size()+" ratings= "+ratings.size());
		}
		
		Object[] myArrays=new Object[3] ;
		myArrays[0]=usersIDs ;
		myArrays[1]=usersItems ;
		myArrays[2]=itemsRatings ;
		return myArrays;
	}
	
	public void printDataToTabDelimitedFile(ArrayList<Long> users, ArrayList<ArrayList<Long>> items, ArrayList<ArrayList<Float>> ratings, String path)
	{
		try {
		// Tab delimited file will be written to data with the name tab-file.csv
			FileWriter fos = new FileWriter(rootPath+path);
			PrintWriter dos = new PrintWriter(fos);
			// loop through all your data and print it to the file
			for (int i=0;i<users.size();i++)
			{
				for(int j=0;j<items.get(i).size();j++) {
					dos.print(users.get(i)+" ");
					dos.print(items.get(i).get(j)+" ");
					dos.print(ratings.get(i).get(j)+" ");
					if(i!=users.size()-1) {
						dos.println();
					}
					else { 
						if(j != items.get(i).size()-1) dos.println() ;
					}
				}
			}
			dos.close();
			fos.close();
			} catch (IOException e) {
				System.out.println("Error Printing Tab Delimited File");
			}
	}
	
	public void runScript(String arg){
	    Process process;
		try{
	          process = Runtime.getRuntime().exec(new String[]{rootPath+"\\py_scripts\\dist\\graph_learning",arg});
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

}
