import java.util.*;
import java.util.Map.Entry;

import org.graphstream.graph.implementations.*;
import org.graphstream.graph.*;

public class Condition {

	
	static int cond_count = 0;
	private int id;

	List<ArrayList<Triplet>> archive = new ArrayList<ArrayList<Triplet>>();
	private int current = 0;

	List<ArrayList<Triplet>> regles = new ArrayList<ArrayList<Triplet>>();

	//List<String> whitheList = new ArrayList<String>(Arrays.asList());
	//List<String> blackList = new ArrayList<String>(Arrays.asList());

	//List<String> whitheList = new ArrayList<String>(Arrays.asList("RE_CPU_PROD_BOUCHON", "RE_EJ", "RE_VTAS", "RE_VRM", "RE_VRC", "RE_VBB", "RE_CONV", "RE_BMC", "RE_BME", "RE_DVL", "RE_PINCES", "RE_VTEX", "RE_VBR", "RE_VBN"));	
	//List<String> blackList = new ArrayList<String>(Arrays.asList("FE_CPU_PROD_BOUCHON", "FE_EJ", "FE_VTAS", "FE_VRM", "FE_VRC", "FE_VBB", "FE_CONV", "FE_BMC", "FE_BME", "FE_DVL", "FE_PINCES", "FE_VTEX", "FE_VBR", "FE_VBN"));

	// List<String> whitheList = new ArrayList<String>(Arrays.asList("RE_A"));
	// List<String> blackList = new ArrayList<String>(Arrays.asList("FE_A"));

	List<String> whitheList = new ArrayList<String>(Arrays.asList("RE_A1B2", "RE_A1B4"));
	List<String> blackList = new ArrayList<String>(Arrays.asList("FE_A1B2","FE_A1B4", "RE_P2E", "RE_P4E" ,"FE_P1S"));
	
	// List<String> whitheList = new ArrayList<String>(Arrays.asList("RE_Box_conveyor", "RE_Part_conveyor", "RE_Grab", "RE_C_plus"));
	// List<String> blackList = new ArrayList<String>(Arrays.asList("FE_Auto", "RE_Auto", "RE_Manual", "FE_Manual", "RE_Start", "FE_Start", "RE_Stop", "FE_Stop","RE_Reset_button", "FE_Reset_button"));
	
	/*
	0,0,0,0,0,1,1,0,0,1,1,1,0,1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,[datetime.datetime(2020; 8; 24; 12; 58; 8; 943587)]
	Box_conveyor, Box_at_place
	Part_conveyor, Part_at_place
	Grab, C_plus, Detected, C_limit
	*/
	Graph graph = new SingleGraph("graph");
	boolean displayWithBlackList = false;

	public Condition(){
		this.id = ++cond_count;
		this.archive.add(new ArrayList<Triplet>());
	}

	public int getId(){ return this.id; }

	public ArrayList<Triplet> get(){ return this.archive.get(this.current); }

	public List<ArrayList<Triplet>> getAll(){ return this.archive; }
	
	private List<String> getCurrentContraints() {
        List<String> l = new ArrayList<String>();

		for(Triplet triplet: this.getCurrentRegle()){
			l.add(triplet.contraint.toString());
		}
		return l;
    }

	private void creerNouvelleRegle(){
		this.current++;
		this.archive.add(new ArrayList<Triplet>());
	}

	private List<Triplet> getCurrentRegle(){
		return this.archive.get(this.current);
	}

	public boolean add(List<Evenement> referents, List<Evenement> contraints, ContrainteTemporel ct){

		boolean canAdd = true;
		boolean first = (this.getCurrentRegle().size() == 0 && this.current == 0);
		//on parcours la regle current pour verifier si les evenments contraints que l'on veux ajouté existe deja
		List<String> listeCurrentContraints = this.getCurrentContraints();
		for(Evenement contraint: contraints){
			if(listeCurrentContraints.contains(contraint.toString()) || whitheList.contains(contraint.toString())){
				canAdd = false;
				break;
			}
		}

		if(first){ this.graph.addNode("start").setAttribute("ui.class", "start"); }

		if(canAdd || first){
			//On ajoute le(s) contraint(s) en les referents à leur(s) referent(s)
			for(Evenement contraint: contraints){
				String strC = contraint.toString() +"_"+ contraint.getId();
				Node node = this.graph.addNode(strC);
				node.setAttribute("ui.class", whitheList.contains(contraint.toString()) ? "whiteList" : "");
				node.setAttribute("ui.label", "("+this.current+")"+strC);

				for(Evenement referent: referents){
					this.getCurrentRegle().add(new Triplet(referent, contraint, ct));
					
					String strR = first ? "start" : referent.toString() +"_"+ referent.getId();
					Edge edge = this.graph.addEdge(strR + strC, strR, strC, true);
					edge.setAttribute("layout.weight", 1);
				}
			}
		}else{
			//Sinon On ajoute une nouvelle regles en mettant les contraints en NCT
			this.creerNouvelleRegle();
			for(Evenement contraint: contraints){
				this.getCurrentRegle().add(new Triplet(Evenement.In(), contraint, ContrainteTemporel.NCT()));

				String strC = contraint.toString() +"_"+ contraint.getId();
				Node node = this.graph.addNode(strC);
				node.setAttribute("ui.label", "("+this.current+")"+strC);
				node.setAttribute("ui.class", whitheList.contains(contraint.toString()) ? "whiteList" : "");

				for(Evenement referent: referents){					
					String strR = referent.toString() +"_"+ referent.getId();
					Edge edge = this.graph.addEdge(strR + strC, strR, strC, true);
					edge.setAttribute("layout.weight", 2);
					edge.setAttribute("ui.class", "NEW");
					edge.setAttribute("ui.label", "NEW");
				}
			}
		}

		// for(Evenement r1: referents){
		// 	for(Evenement r2: referents){
		// 		if(r1 == r2){ continue;}
		// 		String s1 = r1.toString() + "_" + r1.getId();
		// 		String s2 = r2.toString() + "_" + r2.getId();
		// 		Edge edge = this.graph.addEdge(s1 + "" + s2, s1, s2, false);
		// 		edge.setAttribute("layout.weight", 1.42);
		// 		edge.setAttribute("ui.class", "inter");
		// 	}
		// }

		return true;
	}

	private String getSignature(int i){
		String s = "";
		for(Triplet triplet: this.archive.get(i)){
			s += "(" + triplet.referent.toString() + "," + triplet.contraint.toString() + ")";
		}
		return s;
	}
	
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

		//List<ArrayList<Triplet>> regles = new ArrayList<ArrayList<Triplet>>();
		
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
		return this.regles;
	}

	public List<ArrayList<Integer>> encodeAll(List<String> captors){
		List<ArrayList<Integer>> codes = new ArrayList<ArrayList<Integer>>();
		int c = 0;
		for(List<Triplet> triplets: this.archive){
			codes.add(new ArrayList<Integer>());
			for(Triplet triplet: triplets){
				codes.get(c).add(triplet.referent.encodeT());
				codes.get(c).add(triplet.referent.encodeC(captors));
				codes.get(c).add(triplet.contraint.encodeT());
				codes.get(c).add(triplet.contraint.encodeC(captors));
				codes.get(c).add(triplet.ct.getMin());
			}
			c++;
		}
		return codes;
	}

	public static List<Integer> encode(List<Triplet> triplets, List<String> captors){
		List<Integer> codes = new ArrayList<Integer>();
		
		for(Triplet triplet: triplets){
			codes.add(1);
			codes.add(triplet.referent.encodeT());
			codes.add(triplet.referent.encodeC(captors));
			codes.add(triplet.contraint.encodeT());
			codes.add(triplet.contraint.encodeC(captors));
			codes.add(triplet.ct.getMin());
		}
		return codes;
	}

	//(In, RE_A1B2, nct) * (RE_A1B2, RE_P1S, 2) * (RE_P1S, FE_P1VP2,  9) * (FE_P1VP2, RE_P1VP2, 1) * (RE_P1VP2, FE_P2E, 4)

	public Map<String, ArrayList<Triplet>> createBadRules(int indice){
		
		Map<String, ArrayList<Triplet>> badRules = new HashMap<String, ArrayList<Triplet>>();
		ArrayList<Triplet> rule = new ArrayList<Triplet>();
		ArrayList<Triplet> tmprule = new ArrayList<Triplet>();
		Triplet deleted = null;
		for(Triplet triplet: this.archive.get(indice)){
			rule.add(triplet.clone());
			tmprule.add(triplet.clone());
		}

		for(int i = 0; i < rule.size() - 1; i++){
			if(!rule.get(i).isNct()){
				deleted = tmprule.remove(i);
				List<Triplet> prev = previous(indice, i);
				List<Triplet> next = next(indice, i);
				if(prev.size() == 1 && next.size() == 1){
					tmprule.get(i).referent = prev.get(0).contraint.clone(); //i car le delete a reduit d'un cran tout
					tmprule.get(i).ct.setMin(deleted.ct.getMin() + next.get(0).ct.getMin());
					tmprule.get(i).ct.setMax(deleted.ct.getMax() + next.get(0).ct.getMax());

					String cause = "collage à " + (deleted.contraint.type=="FE" ? "1" : "0") + " de " + deleted.contraint.contrainte;
					badRules.put(cause, new ArrayList<Triplet>());

					for(Triplet t: tmprule){
						badRules.get(cause).add(t.clone());
					}
				}
			}
			tmprule.clear();
			for(Triplet triplet: this.archive.get(indice)){
				tmprule.add(triplet.clone());
			}
		}
		return badRules;
	}

	public List<Triplet> previous(int regleIndice, int tripletIndice){
		List<Triplet> previous = new ArrayList<Triplet>();
		Triplet current = this.archive.get(regleIndice).get(tripletIndice);
		Triplet last = this.archive.get(regleIndice).get(tripletIndice);

		while (last.ct == current.ct) {
			tripletIndice--;
			last = this.archive.get(regleIndice).get(tripletIndice);
		}
		ContrainteTemporel lastCT = this.archive.get(regleIndice).get(tripletIndice).ct;
		while (last.ct == lastCT) {
			previous.add(last.clone());
			tripletIndice--;
			if(tripletIndice <= 0){ break; }
			last = this.archive.get(regleIndice).get(tripletIndice);
			
		}
		return previous;
	}

	public List<Triplet> next(int regleIndice, int tripletIndice){
		List<Triplet> next = new ArrayList<Triplet>();
		Triplet current = this.archive.get(regleIndice).get(tripletIndice);
		Triplet last = this.archive.get(regleIndice).get(tripletIndice);

		while (last.ct == current.ct) {
			tripletIndice++;
			last = this.archive.get(regleIndice).get(tripletIndice);
		}
		ContrainteTemporel lastCT = this.archive.get(regleIndice).get(tripletIndice).ct;
		while (last.ct == lastCT) {
			next.add(last.clone());
			tripletIndice++;
			if(tripletIndice > this.archive.get(regleIndice).size() - 1){ break; }
			last = this.archive.get(regleIndice).get(tripletIndice);
		}
		return next;
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
			if(s.length() > 3){
				s = s.substring(0, s.length() - inter.length());
				s += r;
			}

		}
		s.substring(0, s.length() - (inter.length() + r.length()));

		return s;
	}
	
}
