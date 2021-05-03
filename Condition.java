import java.util.*;
import org.graphstream.graph.implementations.*;
import org.graphstream.graph.*;

public class Condition {

	
	Graph graph = new SingleGraph("Tutorial 1");

	static int cond_count = 0;
	private int id;
	ArrayList<Triplet> triplets;
	List<ArrayList<Triplet>> archive = new ArrayList<ArrayList<Triplet>>();
	private int current = 0; 
	//List<String> whitheList = new ArrayList<String>(Arrays.asList());
	//List<String> whitheList = new ArrayList<String>(Arrays.asList("RE_A"));
	List<String> whitheList = new ArrayList<String>(Arrays.asList("RE_CPU_PROD_BOUCHON", "RE_EJ", "RE_VTAS", "RE_VRM", "RE_VRC", "RE_VBB", "RE_CONV", "RE_BMC", "RE_BME", "RE_DVL", "RE_PINCES", "RE_VTEX", "RE_VBR", "RE_VBN"));

	public Condition(){
		this.id = ++cond_count;
		this.triplets = new ArrayList<Triplet>();
		this.archive.add(new ArrayList<Triplet>());

		graph.addNode("In_2");

	}

	public int getId(){ return this.id; }

	public ArrayList<Triplet> get(){ return this.archive.get(this.current); }

	public List<ArrayList<Triplet>> getAll(){ return this.archive; }

	public List<Triplet> getTriplet() {
        return Collections.unmodifiableList(this.triplets);
    }

	

	private String getSignature(int i){
		String s = "";
		for(Triplet triplet: this.archive.get(i)){
			s += "(" + triplet.referent.toString() + "," + triplet.contraint.toString() + ")";
		}
		return s;
	}

	// private String getSignature(int i){

	// 	String s = "";
	// 		for(Triplet triplet: this.archive.get(i)){
	// 			if(triplet.referent.isIn() && !whitheList.contains(triplet.contraint.toString())){
	// 				continue;
	// 			}
	// 			if(!triplet.referent.isIn() && whitheList.contains(triplet.contraint.toString())){	
	// 				s += "(" + Evenement.In().toString() + "," + triplet.contraint.toString() + ")";
	// 			}else{
	// 				s += "(" + triplet.referent.toString() + "," + triplet.contraint.toString() + ")";

	// 			}
	// 		}
	// 	return s;
	// }

	
	public List<ArrayList<Triplet>> computeSTC(){
		List<String> signatures = new ArrayList<String>(this.archive.size());
		for(int i = 0; i < this.archive.size(); i++){
			signatures.add("");
			signatures.set(i, this.getSignature(i));
		}

		List<String> signaturesFactorised = new ArrayList<>(new HashSet<>(signatures));

		System.out.println(this.archive.size() + " lignes " + signaturesFactorised.size() + " règles detectées");
		// for(String s: signaturesFactorised){
		// 	System.out.println(s);
		// }

		List<ArrayList<Triplet>> regles = new ArrayList<ArrayList<Triplet>>();
		for(int i = 0; i <  signaturesFactorised.size(); i++){
			regles.add(new ArrayList<Triplet>());
		}

		for(int i = 0; i < this.archive.size(); i++){
			int indice = signaturesFactorised.indexOf(this.getSignature(i));
			List<Triplet> ligne = regles.get(indice);

			if(ligne.isEmpty()){
				for(Triplet triplet: this.archive.get(i)){
					ligne.add(new Triplet(triplet.referent, triplet.contraint, new ContrainteTemporel(triplet.ct.getMin(), triplet.ct.getMax())));
				}
			}else{
				for(int j = 0; j < this.archive.get(i).size(); j++){
					regles.get(indice).get(j).ct.updateRange(this.archive.get(i).get(j).ct);
				}
			}
			
		}
		return regles;
	}

	public boolean add(Triplet t){
		return this.add1(t);
	   //return this.add2(t);
   }



	

	private boolean add1(Triplet t){



		//List<String> evenements_referents = new ArrayList<String>(Arrays.asList("RE_CPU_PROD_BOUCHON", "RE_EJ", "RE_VTAS", "RE_VRM", "RE_VRC", "RE_VBB", "RE_CONV", "RE_BMC", "RE_BME", "RE_DVL", "RE_PINCES"));
		//List<String> evenements_referents = new ArrayList<String>(Arrays.asList("RE_A1B2", "RE_A1B4"));
		//List<String> evenements_referents = new ArrayList<String>(Arrays.asList("RE_A"));
		List<String> evenements_referents = new ArrayList<String>(Arrays.asList("RE_A"));

		if(evenements_referents.contains(t.contraint.toString())){
			if(this.archive.get(current).isEmpty()){
				return this.archive.get(current).add(t);
			}else{
				current++;
				this.archive.add(new ArrayList<Triplet>());
				return this.archive.get(current).add(new Triplet(new Evenement(null, null), t.contraint, new ContrainteTemporel(-1, -1)));
			}
			
		}else{
			return this.archive.get(current).add(t);
		}
	}

	public boolean add2(List<Evenement> referents, Evenement contraint, ContrainteTemporel ct){

		graph.addNode(contraint.toString() +"_"+ contraint.getId());

		boolean newLigne = false;
		
		newLigne = whitheList.contains(contraint.toString());
		
		
		for(Triplet triplet: this.archive.get(this.current)){
			if(newLigne == true){ break; }
			if(!triplet.referent.isIn()){
				Evenement currentContrain = triplet.contraint;
				//si le meme toString mais id different alors n,ouvelle ligne
				newLigne = currentContrain.equals(contraint) && !currentContrain.equalsId(contraint);
			
			}
		}

		if(newLigne){
			if(!this.archive.get(current).isEmpty()){
				this.current++;
				this.archive.add(new ArrayList<Triplet>());
			}
			
			//mettre en nct les contraint d'avant
			for(Evenement referent: referents){
				if(!referent.isIn()){
					this.archive.get(this.current).add(new Triplet(Evenement.In(), referent, ContrainteTemporel.NCT()));
				}
			}

			//
			// if(!whitheList.contains(contraint.toString())){

			// 	for(Evenement referent: referents){
			// 		if(!referent.isIn()){
			// 			this.archive.get(this.current).add(new Triplet(Evenement.In(), referent, ContrainteTemporel.NCT()));
			// 		}
			// 	}
				
			// }


		}
		// if(!whitheList.contains(contraint.toString())){
		// 	for(Evenement referent: referents){
		// 		this.archive.get(this.current).add(new Triplet(referent, contraint, ct));
		// 	}
		// }else{
		// 	this.archive.get(this.current).add(new Triplet(Evenement.In(), contraint, ContrainteTemporel.NCT()));
		// }
		for(Evenement referent: referents){
			this.archive.get(this.current).add(new Triplet(referent, contraint, ct));
			graph.addEdge(contraint.toString() +"_"+ contraint.getId()+""+referent.toString() +"_"+ referent.getId(), contraint.toString() +"_"+ contraint.getId(), referent.toString() +"_"+ referent.getId());
		}
		
		
		return true;
	}
	
	/*private boolean add2(Triplet t){
		//List<String> evenements_referents = new ArrayList<String>(Arrays.asList("RE_A1B2", "RE_A1B4"));
		//List<String> evenements_referents = new ArrayList<String>(Arrays.asList("RE_CPU_PROD_BOUCHON", "RE_EJ", "RE_VTAS", "RE_VRM", "RE_VRC", "RE_VBB", "RE_CONV", "RE_BMC", "RE_BME", "RE_DVL", "RE_PINCES"));
		List<String> evenements_referents = new ArrayList<String>(Arrays.asList("RE_A"));
		boolean newLigne = false;
		boolean isAddNct = t.isNct();
		boolean isAddInWhiteList = evenements_referents.contains(t.contraint.toString());

		for(Triplet triplet: this.archive.get(current)){
			
			boolean isTestNct = triplet.isNct();
			boolean sameEvenement = triplet.contraint.type.equals(t.contraint.type) && triplet.contraint.contrainte.equals(t.contraint.contrainte) && triplet.contraint.getId() != t.contraint.getId();
		
			newLigne = isAddInWhiteList || (sameEvenement && !isTestNct);

			if( newLigne ){
			//if(triplet.contraint.toString().equals(t.contraint.toString())){
				current++;
				ArrayList<Triplet> precedents = this.archive.get(current - 1);
				Triplet precedent = this.archive.get(current - 1).get(this.archive.get(current - 1).size() - 1);

				this.archive.add(new ArrayList<Triplet>());
				ContrainteTemporel nct = new ContrainteTemporel(-1, -1);
				
				//verifier tout ceux qui ont la meme contrainte que this.archive.get(current - 1).get(this.archive.get(current - 1).size() - 1).contraint
				for(int i = precedents.size() - 1; i >= 0; i--){
					if( precedent.ct == precedents.get(i).ct ){
						this.archive.get(current).add(new Triplet(new Evenement(null, null), precedents.get(i).contraint, nct));
					}
					else{
						break;
					}
				}

				//supprime les doublons nct (meme evenement contraint) de la nouvelle ligne
				for(int i = 0; i < this.archive.get(current).size() - 1; i++){

					if(this.archive.get(current).get(i).contraint == this.archive.get(current).get(i+1).contraint){
						this.archive.get(current).remove(i);
					}
				}

				this.archive.get(current).add(t);
				return true;
			}

		}
		return this.archive.get(current).add(t);
	}*/

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
		return Condition.toString(this.archive);
	}

	public static String toString(List<ArrayList<Triplet>> regle){
		String s = "";
		String inter = " * ";
		String r = "\n";
		
		for(ArrayList<Triplet> triplets: regle){
			for(Triplet triplet: triplets){
				s += triplet + inter;
			}
			s = s.substring(0, s.length() - inter.length());
			s += r;
		}
		s.substring(0, s.length() - (inter.length() + r.length()));

		return s;
	}
	
}
