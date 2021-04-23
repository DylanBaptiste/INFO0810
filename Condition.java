import java.util.*;

public class Condition {
	static int cond_count = 0;
	private int id;
	List<Triplet> triplets; 

	public Condition(){
		this.id = ++cond_count;
		this.triplets = new ArrayList<Triplet>();
	}
	
	public Condition(ArrayList<Triplet> ts){
		this.id = ++cond_count;
		this.triplets = ts;
	}

	public int getId(){ return this.id; }

	public List<Triplet> getTriplet() {
        return Collections.unmodifiableList(this.triplets);
    }

	public boolean add(Triplet t){
		return this.triplets.add(t);
	}

	/**
	 * Cherche les triplets directement precendent à un temps donné.
	 * @param dt La valeur d'une contrainte temporel presente dans la condition
	 * @return La liste des triplets directement precendent à un temps donné.
	 */
	public List<Triplet> getPreviousTriplets(int dt){
		if(this.triplets == null){ return null; }
		
		int previousDt = 0;
		Triplet tmpTriplet = null;
		List<Triplet> tmp = new ArrayList<Triplet>();
		
		ListIterator<Triplet> listIterator = this.triplets.listIterator(this.triplets.size());
		
		if(!listIterator.hasPrevious()){ return tmp; }
		
		tmpTriplet = listIterator.previous();
		
		
		while (listIterator.hasPrevious() && tmpTriplet.ct.range != null && tmpTriplet.ct.getMin() != dt) {
			tmpTriplet = listIterator.previous();
		}
		
		if(!listIterator.hasPrevious()){ return tmp; }
		
		if(tmpTriplet.ct.range == null){
			while (listIterator.hasPrevious()){
				tmp.add(listIterator.previous());
			}
			return tmp;
		}

		//si ici : tmpTriplet.ct.getMin() == dt
		
		while(listIterator.hasPrevious() && tmpTriplet.ct.getMin() == dt){
			tmpTriplet = listIterator.previous();
		}
		

		//si ici : tout les dt sont consomé
		previousDt = tmpTriplet.ct.getMax();
		tmp.add(tmpTriplet);
		if(!listIterator.hasPrevious()){return tmp;}
		else{
			tmpTriplet = listIterator.previous();
		}
		while(tmpTriplet.ct.getMax() == previousDt){
			tmp.add(tmpTriplet);
			
			if(!listIterator.hasPrevious()){
				return tmp;
			}else{
				tmpTriplet = listIterator.previous();
			}
			
		}

		//si ici tout les tmp ont été rentré
		return tmp;
			

	}
	public boolean isValid(){
		return true;
	}

	public String toString(){
		return this.triplets.toString();
	}
	
}
