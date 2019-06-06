package org.apache.mahout.cf.taste.neighborhood;


import java.util.Collection;
import java.util.Iterator;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

public class UserNeighborhoodImpl  implements UserNeighborhood{
	private JSONObject jsonObject=null ;

	
	
	
	public UserNeighborhoodImpl(JSONObject jsonObject) {
		super();
		this.jsonObject = jsonObject;
	}



	@Override
	public void refresh(Collection<Refreshable> alreadyRefreshed) {
		// TODO Auto-generated method stub
		
	}
	
	

	@Override
	public long[] getUserNeighborhood(long userID) throws TasteException {
		 JSONArray valObj = (JSONArray) this.jsonObject.get(Long.toString(userID));
		 //System.out.println("UserId= "+userID+" size= "+valObj.size());
	   	 long[] neighbors=new long[valObj.size()] ;
	   	 //System.out.println(valObj.toString());
	   	 for (int i = 0; i < valObj.size(); i++) {
	   		    JSONObject json = (JSONObject) valObj.get(i) ;
	   		    Iterator t = json.keySet().iterator() ;
	   		    while(t.hasNext()) {
	   		    	String myKey=(String) t.next();
	   		    	neighbors[i]=Long.valueOf(myKey);
	   		    }
	   	 }
   	 return neighbors ;
	}

}
