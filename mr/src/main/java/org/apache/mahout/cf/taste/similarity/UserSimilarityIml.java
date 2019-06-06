package org.apache.mahout.cf.taste.similarity;

import java.util.Collection;
import java.util.Iterator;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhoodImpl;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;


public class UserSimilarityIml implements UserSimilarity{

	private JSONObject jsonObject=null ;
	
	
	public UserSimilarityIml(JSONObject jsonObject) {
		super();
		this.jsonObject = jsonObject;
	}

	@Override
	public void refresh(Collection<Refreshable> alreadyRefreshed) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public double userSimilarity(long userID1, long userID2) throws TasteException {
   	 	JSONArray valObj = (JSONArray) this.jsonObject.get(Long.toString(userID1));
   	 	UserNeighborhoodImpl user =new UserNeighborhoodImpl(jsonObject) ;
   	 	long[] neighbors_ids= user.getUserNeighborhood(userID1) ;

		for (int i = 0; i < valObj.size(); i++) {
		    JSONObject json = (JSONObject) valObj.get(i) ;
		    Iterator t = json.values().iterator() ;
		    while(t.hasNext()) {
		    	Double myKey=(Double) t.next();
		    	if(neighbors_ids[i]==userID2) {
		    		return myKey ;
		    	}
		    }
		}
		return 0 ;
	}

	@Override
	public void setPreferenceInferrer(PreferenceInferrer inferrer) {
		// TODO Auto-generated method stub
		
	}

}
