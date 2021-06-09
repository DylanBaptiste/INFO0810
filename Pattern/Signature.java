package Pattern;

import java.util.*;

public class Signature extends ArrayList<Object>{

    @Override
	public boolean add(Object element){
        if(!super.contains(element)){
            return super.add(element);
        }else{
            return false;
        }
	}
    
}