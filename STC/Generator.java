package STC;
import java.util.*;

import org.graphstream.graph.implementations.*;
import org.graphstream.graph.*;

public class Generator {

	public List<ArrayList<Triplet>> archive = new ArrayList<ArrayList<Triplet>>(); // contient les STC non factorisé
	private int current = 0; // entier pour savoir quelle STC de this.archive est en cours de contruction

	List<ArrayList<Triplet>> regles = new ArrayList<ArrayList<Triplet>>(); // contient les STC factorisée

	// List<String> whitheList = new ArrayList<String>(Arrays.asList());
	// List<String> blackList = new ArrayList<String>(Arrays.asList());

	public List<String> whitheList = new ArrayList<String>(Arrays.asList("RE_YCONV", "RE_ZDESC", "RE_ZMONT", "RE_FERMER", "RE_OUVRIR", "RE_XGLISS", "XCONV"));
	public List<String> blackList = new ArrayList<String>(Arrays.asList("RE_conv_debut"));

	public Graph graph = new SingleGraph("graph"); // graph d'graphstream, utiliser que pour un affichage graphique de this.archive

	// Constructor
	public Generator(){
		this.archive.add(new ArrayList<Triplet>());
	}

	public List<ArrayList<Triplet>> getAll(){ return this.archive; }
	
	private List<String> getCurrentContraints() {
        List<String> l = new ArrayList<String>();

		for(Triplet triplet: this.getCurrentSTC()){
			l.add(triplet.contraint.toString());
		}
		return l;
    }

	/**
	 * Creer une nouvelle entré dans this.archive et met à jour this.current
	 */
	private void creerNouvelleSTC(){
		this.current++;
		this.archive.add(new ArrayList<Triplet>());
	}

	/**
	 * @return la liste de triplet qui correspond à la STC en cours de contruction
	 */
	private List<Triplet> getCurrentSTC(){
		return this.archive.get(this.current);
	}

	/**
	 * Ajoute à la liste des STC dans this.archive les triplets
	 * Elle fait le calcul pour savoir si les evenement contraint peuvent etre mis dans la STC en cours de contruction ou s'il faut en creer une nouvelle
	 * @param referents une liste d'evenement referents
	 * @param contraints un liste d'evenement contraint par les evenement present dans contraints
	 * @param ct la ContrainteTemporel qui serapre referents de contraints
	 */
	public void add(List<Evenement> referents, List<Evenement> contraints, ContrainteTemporelle ct){

		boolean canAdd = true;
		boolean first = (this.getCurrentSTC().size() == 0 && this.current == 0); // le cas first arrive quand this.archive est vide
		
		//on parcours la STC current pour verifier si les evenments contraints que l'on veux ajouté existe deja
		List<String> listeCurrentContraints = this.getCurrentContraints();
		for(Evenement contraint: contraints){
			if(listeCurrentContraints.contains(contraint.toString()) || whitheList.contains(contraint.toString())){
				canAdd = false;
				break;
			}
		}

		if(first){
			Node node = this.graph.addNode("start");
			node.setAttribute("ui.class", "start");
			node.setAttribute("layout.frozen", true);
			node.setAttribute("x", 0.0);
			node.setAttribute("y", 0.0);
		}

		if(canAdd || first){
			//On ajoute le(s) contraint(s) en les referents à leur(s) referent(s)
			for(Evenement contraint: contraints){
				
				// pour le graph on ajouyte un noeud pour cet evenemlent contrazint
				String strC = contraint.toString() +"_"+ contraint.getId();
				Node node = this.graph.addNode(strC);
				node.setAttribute("ui.class", whitheList.contains(contraint.toString()) ? "whiteList" : "");
				node.setAttribute("ui.label", "("+this.current+")"+strC);
				node.setAttribute("x", ct.getId() + 1 * 10);
				
				// pour chaqsue referents
				for(Evenement referent: referents){
					//on ajoute un nouveau triplet  
					this.getCurrentSTC().add(new Triplet(referent, contraint, ct));
					
					// pour le graph
					String strR = first ? "start" : referent.toString() +"_"+ referent.getId();
					Edge edge = this.graph.addEdge(strR + strC, strR, strC, true);
					edge.setAttribute("layout.weight", 1); /* weight peut ectre egale à la valeur de CT et du coup la longueur sur le graph retranscrira la durer de la CT */
					edge.setAttribute("ui.label", ct.getMin()); // on labelise la liaison juste pour l'affichage
				}
			}
		}else{
			// Sinon on termine la STC courante
			// On ajoute une nouvelle STC en mettant les contraints en NCT
			this.creerNouvelleSTC();
			for(Evenement contraint: contraints){
				this.getCurrentSTC().add(new Triplet(Evenement.In(), contraint, ContrainteTemporelle.NCT()));

				String strC = contraint.toString() +"_"+ contraint.getId();
				Node node = this.graph.addNode(strC);
				node.setAttribute("ui.label", "("+this.current+")"+strC);
				node.setAttribute("ui.class", whitheList.contains(contraint.toString()) ? "whiteList" : "");
				node.setAttribute("x", ct.getId() + 1 * 10);

				for(Evenement referent: referents){					
					String strR = referent.toString() +"_"+ referent.getId();
					Edge edge = this.graph.addEdge(strR + strC, strR, strC, true);
					edge.setAttribute("layout.weight", 2);
					edge.setAttribute("ui.class", "NEW");
					edge.setAttribute("ui.label", "NEW");
				}
			}
		}

		// uniquement pour le graph
		// on lie les evement contraints à leur referents
		for(int i = 0; i < contraints.size() - 1; i++){
			Evenement r1 = contraints.get(i);
			Evenement r2 = contraints.get(i+1);
			if(r1 == r2){ continue; }
			String s1 = r1.toString() + "_" + r1.getId();
			String s2 = r2.toString() + "_" + r2.getId();
			Edge edge = this.graph.addEdge(s1 + "" + s2 , s1, s2, true);
			Edge edge2 = this.graph.addEdge(s2 + "" + s1 , s2, s1, true);
			float w = 0.5f;
			edge.setAttribute("ui.class", "inter");
			edge.setAttribute("ui.label", w);
			edge.setAttribute("layout.weight", w);
			edge2.setAttribute("ui.class", "inter");
			edge2.setAttribute("ui.label", w);
			edge2.setAttribute("layout.weight", w);
		}

	}

	/**
	 * Calcule la signature d'une STC
	 * c'est a dire si deux STC ont le meme ordre d'evenement alors ils auront la meme signature
	 * cette fonction est utilisé pour savoir si deux STC peuvent se factoriser
	 * @param i l'id dand this.archive
	 * @return la signature de this.archive.get(i)
	 */
	private String getSignature(int i){
		String s = "";
		for(Triplet triplet: this.archive.get(i)){
			s += "(" + triplet.referent.toString() + "," + triplet.contraint.toString() + ")";
		}
		return s;
	}
	
	/**
	 * Creer un list de list de triplet qui represente la factorisation des STC dans this.archive
	 * @return
	 */
	public List<ArrayList<Triplet>> computeFactorized(){
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
					ligne.add(new Triplet(triplet.referent, triplet.contraint, new ContrainteTemporelle(triplet.ct.getMin(), triplet.ct.getMax())));
				}
			}else{
				for(int j = 0; j < this.archive.get(i).size(); j++){
					regles.get(indice).get(j).ct.updateRange(this.archive.get(i).get(j).ct);
				}
			}
			
		}
		return this.regles;
	}

	/**
	 * Encode les stc presentes dans this.archive
	 * @param composantsNames  la liste des nom des composants (utilise leur id dans le tableau pour l'encodage)
	 * @return une liste de liste (comme this.archive) d'entier qui representes les STC de this.archive encodé
	 */
	public List<ArrayList<Integer>> encodeNormalSTC(List<String> composantsNames){
		List<ArrayList<Integer>> codes = new ArrayList<ArrayList<Integer>>();
		int c = 0;
		for(List<Triplet> triplets: this.archive){
			codes.add(new ArrayList<Integer>());
			for(Triplet triplet: triplets){
				codes.get(c).add(triplet.referent.encodeT());
				codes.get(c).add(triplet.referent.encodeC(composantsNames));
				codes.get(c).add(triplet.contraint.encodeT());
				codes.get(c).add(triplet.contraint.encodeC(composantsNames));
				codes.get(c).add(triplet.ct.getMin());
			}
			c++;
		}
		return codes;
	}

	/**
	 * Encode une list de triplets
	 * @param triplets la list à encoder
	 * @param composantsNames la liste des nom des composants (utilise leur id dans le tableau pour l'encodage)
	 * @return une list d'entier qui reresentes la liste de triplets encodé 
	 */
	public static List<Integer> encodeTriplet(List<Triplet> triplets, List<String> composantsNames){
		List<Integer> codes = new ArrayList<Integer>();
		
		for(Triplet triplet: triplets){
			//codes.add(1);
			codes.add(triplet.referent.encodeT());
			codes.add(triplet.referent.encodeC(composantsNames));
			codes.add(triplet.contraint.encodeT());
			codes.add(triplet.contraint.encodeC(composantsNames));
			codes.add(triplet.ct.getMin());
		}
		return codes;
	}

	/**
	 * créé tout les symptomes possible pour un indice de stc dans this.archive donné
	 * @param indice indice de la ligne dans this.archive
	 * @return une hasmap { symptome => list triplet }
	 * 
	 * 
		page 210
		(In, RE_A1B4, nct) *                              (RE_A1B4, FE_P1VP4, 13) * (FE_P1VP4, RE_P1VP4, 1) * (RE_P1VP4, FE_P4E, 8)


		(In, RE_A1B4, nct) * (RE_A1B4, S_P1S, 3])      * (RE_P1S, FE_P1VP4, 11) * (FE_P1VP4, RE_P1VP4, 1) * (RE_P1VP4, FE_P4E, [6, 8])
		(In, RE_A1B4, nct) * (RE_A1B4, RE_P1S, [1, 3]) * (RE_P1S, FE_P1VP4, 11) * (FE_P1VP4, RE_P1VP4, 1) * (RE_P1VP4, FE_P4E, [6, 8])
	 * 
	 */
	public Map<String, ArrayList<Triplet>> createBadRulesV2(int indice){

		ArrayList<Triplet> currentListContraint = new ArrayList<Triplet>();
		Map<String, ArrayList<Triplet>> badRules = new HashMap<String, ArrayList<Triplet>>();
		ArrayList<Triplet> rule = new ArrayList<Triplet>();
		ArrayList<Triplet> tmprule = new ArrayList<Triplet>();
		Triplet deleted = null;

		for(Triplet triplet: this.archive.get(indice)){
			rule.add(triplet.clone());
			tmprule.add(triplet.clone());
		}

		for(int i = 0; i < rule.size(); i++){

			//recuperer les triplets qui ont la meme contrainte que celui que je veux supprimer
			for(Triplet triplet: this.archive.get(indice)){
				if(triplet.ct == this.archive.get(indice).get(i).ct){
					currentListContraint.add(triplet.clone());
				}
			}
			//cas plusieurs noeuds freres si parmi ceux qui ont la meme contrainte on trouve differents contraint
			Set<String> freres = new HashSet<String>();
			for(Triplet t: currentListContraint){ freres.add(t.contraint.toString()); }

			//deleted = tmprule.remove(i); //supprimer le triplet
			deleted = tmprule.get(i);
			Triplet replace = deleted.clone();
			replace.contraint = Evenement.S(replace.contraint.composant);
			tmprule.set(i, replace);

			if(!rule.get(i).isNct()){

				
				List<Triplet> prev = previous(indice, i); //les triplets qui ont la contrainte precedente
				//List<Triplet> next = next(indice, i); //pas utilisé
				
				//cas plusieurs noeuds freres
				if(freres.size() > 1){
					//ont supprime le triplet 
					// et ont supprime dans les suivants les triplets ayant en referent le contraitn du triplet supprimé
					for(int j = 0; j < tmprule.size(); j++){
						if(tmprule.get(j).referent.equals(deleted.contraint) || tmprule.get(j).contraint.equals(deleted.contraint)){
							tmprule.remove(j);
							j--;
						}
					}
					String cause = "collage à " + (deleted.contraint.type=="FE" ? "1" : "0") + " de " + deleted.contraint.composant;
					badRules.put(cause, new ArrayList<Triplet>());

					for(Triplet t: tmprule){
						badRules.get(cause).add(t.clone());
					}

				}

				//cas seul
				if(freres.size() == 1){

					String cause = "collage à " + (deleted.contraint.type=="FE" ? "1" : "0") + " de " + deleted.contraint.composant;
					badRules.put(cause, new ArrayList<Triplet>());

					for(int j = 0; j < tmprule.size(); j++){
						if(tmprule.get(j).contraint.equals(deleted.contraint)){
							//on supprime les triplet dont le contraint est le meme celui du triplet supprimé
							tmprule.remove(j);
							j--;
						}
						else{
							//sinon on ajoute
							if(tmprule.get(j).referent.equals(deleted.contraint)){
								//sauf si le triplet à en referent le contraint du triplet supprimé
								for(Triplet t: prev){
									//dans ce cas pour chaque contraint des triplets de la contrainte temporelle precedente on ajoute un nouveau triplet avec en referent ces contraints là
									ContrainteTemporelle ct = tmprule.get(j).ct.clone();
									ct.setMin(deleted.ct.getMin() + ct.getMin());
									ct.setMax(deleted.ct.getMax() + ct.getMax());
									badRules.get(cause).add(new Triplet(t.contraint.clone(), tmprule.get(j).contraint.clone(), ct));
								}
							}else{
								badRules.get(cause).add(tmprule.get(j));
							}
						}
					}
				}
			}else{
				//cas plusieurs noeuds freres
				if(freres.size() > 1){
					for(int j = 0; j < tmprule.size(); j++){
						if(tmprule.get(j).referent.equals(deleted.contraint) || tmprule.get(j).contraint.equals(deleted.contraint)){
							tmprule.remove(j);
							j--;
						}
					}
					String cause = "cas collage à " + (deleted.contraint.type=="FE" ? "1" : "0") + " de " + deleted.contraint.composant;
					badRules.put(cause, new ArrayList<Triplet>());

					for(Triplet t: tmprule){
						badRules.get(cause).add(t.clone());
					}
				}

				//cas seul nct
				if(freres.size() == 1){
					String cause = "cas collage à " + (deleted.contraint.type=="FE" ? "1" : "0") + " de " + deleted.contraint.composant;
					badRules.put(cause, new ArrayList<Triplet>());

					ContrainteTemporelle nct = ContrainteTemporelle.NCT();
					
					for(int j = 0; j < tmprule.size(); j++){
						if(tmprule.get(j).contraint.equals(deleted.contraint)){
							//on supprime les triplet dont le contraint est le meme celui du triplet supprimé
							tmprule.remove(j);
							j--;
						}else{
							if(tmprule.get(j).referent.equals(deleted.contraint)){
								badRules.get(cause).add(new Triplet(Evenement.In(), tmprule.get(j).contraint.clone(), nct));
							}else{
								badRules.get(cause).add(tmprule.get(j));
							}
						}
					}
				}
			}
			//reinitialisation des variables temporaires
			tmprule.clear();
			currentListContraint.clear();
			for(Triplet triplet: this.archive.get(indice)){ tmprule.add(triplet.clone()); }
		}

		// remarque : les regles avec un symptome similaire
		// (dans la meme ligne: i.e this.archive.get(indice) ce cas arrive pour les triplets qui ont le meme evenement contraint)
		// génèrent la meme règle
		// comme badRules est une hasmap les doublons sont calculé mais automatiquement ignoré (par ecrasement)

		// remarque : becoup d'optimisation possible ici
		// plus facile si on avait une structure de graphe où l'on suppimerait puis regenerait le resultat sous forme de liste de triplets
		return badRules;
	}

	/**
	 * créé tout les symptome possible pour un indice de ligne donné
	 * @param indice indice de la ligne dans this.archive
	 * @return une hasmap { symptome => list triplet }
	 */
	public Map<String, ArrayList<Triplet>> createBadRulesV1(int indice){

		ArrayList<Triplet> currentListContraint = new ArrayList<Triplet>();
		Map<String, ArrayList<Triplet>> badRules = new HashMap<String, ArrayList<Triplet>>();
		ArrayList<Triplet> rule = new ArrayList<Triplet>();
		ArrayList<Triplet> tmprule = new ArrayList<Triplet>();
		Triplet deleted = null;

		for(Triplet triplet: this.archive.get(indice)){
			rule.add(triplet.clone());
			tmprule.add(triplet.clone());
		}

		for(int i = 0; i < rule.size(); i++){

			//recuperer les triplets qui ont la meme contrainte que celui que je veux supprimer
			for(Triplet triplet: this.archive.get(indice)){
				if(triplet.ct == this.archive.get(indice).get(i).ct){
					currentListContraint.add(triplet.clone());
				}
			}
			//cas plusieurs noeuds freres si parmi ceux qui ont la meme contrainte on trouve differents contraint
			Set<String> freres = new HashSet<String>();
			for(Triplet t: currentListContraint){ freres.add(t.contraint.toString()); }

			deleted = tmprule.remove(i); //supprimer le triplet

			if(!rule.get(i).isNct()){

				
				List<Triplet> prev = previous(indice, i); //les triplets qui ont la contrainte precedente
				//List<Triplet> next = next(indice, i); //pas utilisé
				
				//cas plusieurs noeuds freres
				if(freres.size() > 1){
					//ont supprime le triplet 
					// et ont supprime dans les suivants les triplets ayant en referent le contraitn du triplet supprimé
					for(int j = 0; j < tmprule.size(); j++){
						if(tmprule.get(j).referent.equals(deleted.contraint) || tmprule.get(j).contraint.equals(deleted.contraint)){
							tmprule.remove(j);
							j--;
						}
					}
					String cause = "cas plusieurs collage à " + (deleted.contraint.type=="FE" ? "1" : "0") + " de " + deleted.contraint.composant;
					badRules.put(cause, new ArrayList<Triplet>());

					for(Triplet t: tmprule){
						badRules.get(cause).add(t.clone());
					}

				}

				//cas seul
				if(freres.size() == 1){

					String cause = "cas seul collage à " + (deleted.contraint.type=="FE" ? "1" : "0") + " de " + deleted.contraint.composant;
					badRules.put(cause, new ArrayList<Triplet>());

					for(int j = 0; j < tmprule.size(); j++){
						if(tmprule.get(j).contraint.equals(deleted.contraint)){
							//on supprime les triplet dont le contraint est le meme celui du triplet supprimé
							tmprule.remove(j);
							j--;
						}
						else{
							//sinon on ajoute
							if(tmprule.get(j).referent.equals(deleted.contraint)){
								//sauf si le triplet à en referent le contraint du triplet supprimé
								for(Triplet t: prev){
									//dans ce cas pour chaque contraint des triplets de la contrainte temporelle precedente on ajoute un nouveau triplet avec en referent ces contraints là
									ContrainteTemporelle ct = tmprule.get(j).ct.clone();
									ct.setMin(deleted.ct.getMin() + ct.getMin());
									ct.setMax(deleted.ct.getMax() + ct.getMax());
									badRules.get(cause).add(new Triplet(t.contraint.clone(), tmprule.get(j).contraint.clone(), ct));
								}
							}else{
								badRules.get(cause).add(tmprule.get(j));
							}
						}
					}
				}
			}else{
				//cas plusieurs noeuds freres
				if(freres.size() > 1){
					for(int j = 0; j < tmprule.size(); j++){
						if(tmprule.get(j).referent.equals(deleted.contraint) || tmprule.get(j).contraint.equals(deleted.contraint)){
							tmprule.remove(j);
							j--;
						}
					}
					String cause = "cas plusieurs et NCT collage à " + (deleted.contraint.type=="FE" ? "1" : "0") + " de " + deleted.contraint.composant;
					badRules.put(cause, new ArrayList<Triplet>());

					for(Triplet t: tmprule){
						badRules.get(cause).add(t.clone());
					}
				}

				//cas seul nct
				if(freres.size() == 1){
					String cause = "cas seul et NCT collage à " + (deleted.contraint.type=="FE" ? "1" : "0") + " de " + deleted.contraint.composant;
					badRules.put(cause, new ArrayList<Triplet>());

					ContrainteTemporelle nct = ContrainteTemporelle.NCT();
					
					for(int j = 0; j < tmprule.size(); j++){
						if(tmprule.get(j).contraint.equals(deleted.contraint)){
							//on supprime les triplet dont le contraint est le meme celui du triplet supprimé
							tmprule.remove(j);
							j--;
						}else{
							if(tmprule.get(j).referent.equals(deleted.contraint)){
								badRules.get(cause).add(new Triplet(Evenement.In(), tmprule.get(j).contraint.clone(), nct));
							}else{
								badRules.get(cause).add(tmprule.get(j));
							}
						}
					}
				}
			}
			//reinitialisation des variables temporaires
			tmprule.clear();
			currentListContraint.clear();
			for(Triplet triplet: this.archive.get(indice)){ tmprule.add(triplet.clone()); }
		}

		// remarque : les regles avec un symptome similaire
		// (dans la meme ligne: i.e this.archive.get(indice) ce cas arrive pour les triplets qui ont le meme evenement contraint)
		// génèrent la meme règle
		// comme badRules est une hasmap les doublons sont calculé mais automatiquement ignoré (par ecrasement)

		// remarque : becoup d'optimisation possible ici
		// plus facile si on avait une structure de graphe où l'on suppimerait puis regenerait le resultat sous forme de liste de triplets
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
		ContrainteTemporelle lastCT = this.archive.get(regleIndice).get(tripletIndice).ct;
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
			if(tripletIndice > this.archive.get(regleIndice).size() - 1){ return next; }
			last = this.archive.get(regleIndice).get(tripletIndice);
		}
		ContrainteTemporelle lastCT = this.archive.get(regleIndice).get(tripletIndice).ct;
		while (last.ct == lastCT) {
			next.add(last.clone());
			tripletIndice++;
			if(tripletIndice > this.archive.get(regleIndice).size() - 1){ break; }
			last = this.archive.get(regleIndice).get(tripletIndice);
		}
		return next;
	}

	public String toString(){
		return Generator.toString(this.archive);
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
