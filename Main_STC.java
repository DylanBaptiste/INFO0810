import STC.*;

import java.util.*;
import javax.imageio.ImageIO;
import java.io.*;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.sql.Timestamp;

import java.awt.image.BufferedImage;
// F1-score recall

public class Main_STC {

    public static void main(String[] args) throws IOException {

		/*========== Paramètres ==========*/
		
		/* fichier */
		String delimiter = ","; // le demimiter dans le fichier csv
		String input = "normal_reel"; // Nom du fichier
		String system = "import_export"; //Nom du system
		// le chemin vers le fichier est : ./data/[system]/[input].csv
		// les fichier genéré suront dans : ./resultat/[system]/ (fichier images, txt avec les STC et csv avec les stc encodées/labélisées)
		String timeColumnName = "Temps"; // Nom de la colone qui contieznt le temps dans le fichier csv
		String TrueValue = "1"; // valeur fausse dans le fichier csv
		String FalseValue = "0"; // valeur vrai dans le fichier csv
		
		/* graph */
		boolean displayGraph = false; //affichage du graph ou non à la fin
		int graphQuality = 0; /*0 -> 4*/ //visuel : qualité du graph
		boolean graphAntialias = false;  //visuel : anti aliasing
		
		/*=================================*/

		Generator generator = new Generator(); // generateur des STC

		// String output = input; 
		String pathInput = System.getProperty("user.dir")+"\\data\\"+ system+"\\" + input + ".csv"; // fichier csv du systeme
		String pathOutput = System.getProperty("user.dir")+"\\resultat\\" +system+"\\" + input + ".txt"; // fichier resultat apres generation
		String pathOutputCSV = System.getProperty("user.dir")+"\\resultat\\" +system+"\\"+ input + ".csv"; // fichier resultat encodé et labélisé apres generation
		
		String[] headers, readValues, previousReadValues;
		int timeColumnIndex = -1;
		String line;
		BufferedReader reader;
		
		List<String> lastValues = new ArrayList<String>();
		List<Integer> composantsIndexes = new ArrayList<Integer>(); // contient l'id des colonnes contenant les valeurs logique (permet d'ignorer les valeur continue/string etc...)

		List<Evenement> evReferents = new ArrayList<Evenement>(); // list des evenements referents
		List<Evenement> evContraints = new ArrayList<Evenement>(); // liste des evenement contraint par les evenement referents
		// à chaque lecture d'un nouveau etat du systeme on fait le roulement :
		// les evenement(s) contraint(s) à T-1 deviennent referent pour les evenement contraint au temps T
		ContrainteTemporelle CT = ContrainteTemporelle.NCT(); // CT contiendra la valueur de temps qui contraint les evenement contraint par rapport au evenements referents
		
		Timestamp lastTime, currentTime = null; // pour savoir la difference de temps entre deux etats
		Boolean updateReferents = false; // variable de controle

		// creation des dossiers et fichiers si il n'existe pas sinon ecrasement des fichiers
		try{
			Files.createDirectories(Paths.get(pathOutput).getParent());
			Files.createFile(Paths.get(pathOutput));
			Files.createFile(Paths.get(pathOutputCSV));
			System.out.println("Creation des fichiers");
		}catch(FileAlreadyExistsException e){
			System.out.println("Ecrasement des fichiers");
		}

		// pour ecrire dans les fichiers
		PrintWriter out = new PrintWriter(pathOutput);
		PrintWriter outCSV = new PrintWriter(pathOutputCSV);
		
		try{ // try pour capturer les erreurs de lecture dans les fichiers

			int nbligne = (int)Files.lines(Paths.get(pathInput)).count(); // utile pour les images

			reader = new BufferedReader(new FileReader(pathInput));
			headers = reader.readLine().split(delimiter);
			timeColumnIndex = Arrays.asList(headers).indexOf(timeColumnName);
			line = reader.readLine();
			previousReadValues = line.split(delimiter); // contient une ligne du csv
			for(int i = 0; i < previousReadValues.length; i++){
				String value = previousReadValues[i];
				lastValues.add(value);
				if(value.equals(FalseValue) || value.equals(TrueValue)){ // si le fichiers contient des autres valeur que nos  FalseValue ou TrueValue les colonnes sont ignorées
					composantsIndexes.add(i);
				}
			}

			evReferents.add(Evenement.In()); // au debut l'evenement referents des premiers evenement est forcement IN
			lastTime = stringToTimestamp(previousReadValues[timeColumnIndex]); // on recupere le temps auquel à eu lieu le premier etat
			boolean first = true; // pour gerer le cas d'initialisation 


			BufferedImage b = new BufferedImage(composantsIndexes.size(), nbligne - 1, BufferedImage.TYPE_INT_RGB);
			BufferedImage bCompressed = new BufferedImage(composantsIndexes.size(), nbligne - 1, BufferedImage.TYPE_INT_RGB);

			// les variable l, ll, j, jj, l, ll ne servent que pour les images
			// cette partie peut etre ignorer si on ne veut pas faire les images car ca n'influe pas sur la generation des stc
			int l = 0;
			int ll = 0;
			int j = 0;
			int jj = 0;
			for(int i: composantsIndexes){
				int v = previousReadValues[i].equals(FalseValue) ? 0 : 255;
				b.setRGB(j, l, (v << 16) | (v << 8) | v); // set la couleur du pixel en fonction de la valeurs dans le fichier csv
				bCompressed.setRGB(jj, ll, (v << 16) | (v << 8) | v);
				j++;
				jj++;
			}
			jj=0;
			l++;
			ll++;

			while ( (line = reader.readLine()) != null && !line.equals("")) {
				readValues = line.split(delimiter);
				
				// pour l'image
				j = 0;
				for(int i: composantsIndexes){
					int v = readValues[i].equals(FalseValue) ? 0 : 255;
					b.setRGB(j, l, (v << 16) | (v << 8) | v);
					j++;
				}
				l++;

				// on compare l'etat du systeme à T-1 et T pour voir si ils sont different
				if(!Arrays.equals(getSubArray(readValues, composantsIndexes), getSubArray(previousReadValues, composantsIndexes))){
					//Si de nouveau evenements arrivent
					currentTime = stringToTimestamp(readValues[timeColumnIndex]); // on recupere le temps T
					int t = (int)diffInNano(lastTime, currentTime); // on regarde la difference avec la precision souhaité
					
					CT = first ? ContrainteTemporelle.NCT() : new ContrainteTemporelle(t, t); // si c'est l'etat initale on est en NCT sinon on creer le Contrainte temporel
					updateReferents = true; // à la fin de la boucle les referents actuel seront ecrasé par les evenement contraints actuellement
					first = false; // on n'est plus dans l'etat T-0
				}else{
					//Si il sont identique on ignore la ligne
					continue;
				}

				jj = 0;
				//On boucle sur tout les composants pour chercher celui/ceux qui a/ont changé
				for(int i: composantsIndexes){
					
					int v = readValues[i].equals(FalseValue) ? 0 : 255; // pour l'image
					bCompressed.setRGB(jj, ll, (v << 16) | (v << 8) | v); // pour l'image
					jj++; // pour l'image

					if(!readValues[i].equals(previousReadValues[i])){ // si ce composant à changé

						// on regarde le type de l'evement Front-montant ou Front-descendant
						String type = "ERROR";
						if(previousReadValues[i].equals(FalseValue)){ type = "RE"; }
						if(previousReadValues[i].equals(TrueValue)){ type = "FE"; }
						// si il y a un probleme dans le fichier (par exemple une valeur manquante ou une valeur ni false ni true le type sera ERROR)

						// on affiche dans le fichers l'evenement detecté
						out.print(type +"_"+ headers[i] +"   \t"+ stringToTimestamp(readValues[timeColumnIndex]));
						
						if(generator.blackList.contains(type +"_"+ headers[i])){
							//si on appartient à la black list on ne l'ajoute juste pas dans les contraints
							out.println("\tBLACK LIST");
							continue; // on coupe la boucle car on ne fera rien de plus avec cette evenement blacklisté
						}
						else{
							out.println("");
						}
						
						//On ajoute dans la des evenement contraint le nouveau evenement detecté
						evContraints.add(new Evenement(type, headers[i]));
					}
				}
				ll++;// pour l'image

				//quand tout les nouveaux contraints sont enregistrés
				//envoyer à add les contraints et leur referents
				// avec la contrainte de temps qui les separent
				generator.add(evReferents, evContraints, CT);

				// Puis ont fait le roulement :
				// evContraints = evReferents
				// L'etat precedent = l'etat courant
				// T-1 = T
				previousReadValues = readValues.clone();		
				if(updateReferents == true && evContraints.size() > 0){
					updateReferents = false;
					evReferents = new ArrayList<Evenement>();
					for(Evenement e: evContraints){
						evReferents.add(e);
					}
					evContraints = new ArrayList<Evenement>();
					lastTime = stringToTimestamp(previousReadValues[timeColumnIndex]);
				}

			}

			
			// on enregistre les images

			// avec les etats doublons successifs
			File img = new File(System.getProperty("user.dir")+"\\resultat\\"+system+"\\"+input+".png");
			ImageIO.write(b, "png", img);

			// sans les etats soublons successifs
			File img2 = new File(System.getProperty("user.dir")+"\\resultat\\"+system+"\\"+input+"_compressed.png");
			BufferedImage resized = new BufferedImage(composantsIndexes.size(), ll, BufferedImage.TYPE_INT_RGB);
			for(int h = 0; h < resized.getHeight(); h++){
				for(int w = 0; w < resized.getWidth(); w++){
					resized.setRGB(w, h, bCompressed.getRGB(w, h));
				} 
			} 
			ImageIO.write(resized, "png", img2);

			// on recupere le nom des coposants dans une liste pour l'encodage
			List<String> composantsNames = new ArrayList<String>();
			for(int columnIndex: composantsIndexes){
				composantsNames.add(headers[columnIndex]);
			}

			// Generation des regles (les STC pas factorisé)
			out.println("\n\n\n===========================================================================================");
			out.println("==================================== Règles ===============================================");
			out.println("===========================================================================================\n\n\n");
			out.println(generator);

			// On gnere et affiche les STC factorisé
			out.println("\n\n\n============================================================================================");
			out.println("================================= Factorisation ============================================");
			out.println("============================================================================================\n\n\n");
			out.println(Generator.toString(generator.computeSTC()));
			
			
			// on genere les symptomes et on les affiches classiquement dans le fichier avec les STC en les labélisants 
			// et en format encodé/labélisé dans un ficher csv à part
			out.println("\n\n\n============================================================================================");
			out.println("=================================== Symptomes ==============================================");
			out.println("============================================================================================\n\n\n");


			// Les cas normaux sont encodé par encodeAll
			for(ArrayList<Integer> codes: generator.encodeAll(composantsNames)){
				outCSV.print("Normal"); // le label
				codes.forEach(value -> outCSV.print("," + value)); // la relge encodé
				outCSV.println();
			}
			
			// Les cas de défaillance
			for(int i = 0; i < generator.archive.size(); i++){ // pour CHAQUE STC dand archive
				
				// on genere tout les probleme possible avec la fonction createBadRulesV1
				Map<String, ArrayList<Triplet>> badrules = generator.createBadRulesV2(i);
				
				// puis ont encode les symptome en avec leur label
				
				// dans le fichier txt
				out.println();
				for(Map.Entry<String, ArrayList<Triplet>> rule : badrules.entrySet()){
					out.print(rule.getKey() + " ==> " );
					rule.getValue().forEach(t -> out.print(" * " + t.toString()));
					out.println();
				}

				// et dans le csv en encodant les list de triplet avec les symptomes				
				for(Map.Entry<String, ArrayList<Triplet>> rule : badrules.entrySet()){
					outCSV.print(rule.getKey());
					Generator.encode(rule.getValue(), composantsNames).forEach(value -> outCSV.print("," + value));
					outCSV.println();
				}
			}
			
			// on ferme les fichiers
			out.close();
			outCSV.close();

			// on affiche le graph qui represente generator.archive
			if(displayGraph){
				System.setProperty("org.graphstream.ui", "swing"); 
				System.setProperty("gs.ui.renderer", "org.graphstream.ui.j2dviewer.J2DGraphRenderer");
				String styleSheet = "";
				try { styleSheet = Files.readString(Paths.get(System.getProperty("user.dir") + "\\STC\\style.txt")); }
				catch (Exception e) { e.printStackTrace(); }
				generator.graph.setAttribute("ui.stylesheet", styleSheet);
				generator.graph.setAttribute("ui.quality", graphQuality);
				generator.graph.setAttribute("ui.antialias", graphAntialias);
				// c.graph.setAttribute("layout.stabilization-limit", 0);
				// c.graph.setAttribute("layout.force", 1);
				generator.graph.display();
			}
			
			// on ferme le leucteur du fichier
			reader.close();
		}
		catch(FileNotFoundException e){
			System.err.println(e.getMessage());
			System.exit(0);
		}
		catch(IOException e){
			System.err.println(e.getMessage());
			System.exit(0);
		}
    }

	// fonction utiles pour les difference de temps
	private static long getTimeNoMillis(Timestamp t) {
		return t.getTime() - (t.getNanos()/1000000);
	}
	public static long diffInNano(Timestamp t1, Timestamp t2) {
		long firstTime = (getTimeNoMillis(t1) * 1000000) + t1.getNanos();
		long secondTime = (getTimeNoMillis(t2) * 1000000) + t2.getNanos();
		long diff = Math.abs(firstTime - secondTime); // diff is in nanoseconds
		Timestamp ret = new Timestamp(diff / 1000000);
		ret.setNanos((int) (diff % 1000000000));
		return (getTimeNoMillis(ret) * 1000000) + ret.getNanos();
	}

	public static long diffInMicro(Timestamp t1, Timestamp t2) {
		return diffInNano(t1, t2) / 1000;
	}
	public static long diffInMilli(Timestamp t1, Timestamp t2) {
		return diffInNano(t1, t2) / 100000;
	}

	public static long diffInCenti(Timestamp t1, Timestamp t2) {
		return diffInNano(t1, t2) / 1000000;
	}


	// fonction pour transformer le temps en string dans le fichier en object Timestamp java
	// suivant comment est le format dans le fichier il faut changer la fonction stringToTimestamp

	public static Timestamp stringToTimestamp(String d){
		return Timestamp.valueOf(d);
	}

	// public static Timestamp stringToTimestamp(String d){
	// 	String[] v;
	// 	String date_time = d.substring(19);
	// 	date_time = date_time.substring(0, date_time.length() - 2);
	// 	v = date_time.split("; ");
	// 	return Timestamp.valueOf(v[0]+"-"+v[1]+"-"+v[2]+" "+v[3]+":"+v[4]+":"+v[5]+"."+v[6]);
	// }

	// fonction pour regarder si deux etat du systeme sont identique
	// elle retourne un tableau qui correspond uniquement aux valeurs des coposants indiqsué dans le parametre indexes
	public static String[] getSubArray(String[] a, List<Integer> indexes){
		List<String> ret = new ArrayList<String>();

		for(int i: indexes){
			ret.add(a[i]);
		}

		return ret.toArray(String[]::new);
	}
}