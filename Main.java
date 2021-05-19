import java.util.*;

import javax.imageio.ImageIO;

import java.io.*;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.sql.Timestamp;

import java.awt.image.BufferedImage;

//FE_A1B2 RE_P1VP2 RE_P2E FE_P1S FE_A1B4

public class Main {

    public static void main(String[] args) throws IOException {

		/*========== Paramètres ==========*/
		
		/* fichier */
		String delimiter = ",";
		String input = "import_export"; //test //convoyeur //input_file //
		String timeColumnName = "Temps";
		String TrueValue = "1";
		String FalseValue = "0";
		
		/* graph */
		boolean displayGraph = false;
		int graphQuality = 0; /*0 -> 4*/
		boolean graphAntialias = false;
		
		/*=================================*/

		Condition c = new Condition();

		// String output = input; 
		String pathInput = System.getProperty("user.dir")+"\\data\\"+ input+"\\" + input + ".csv";
		String pathOutput = System.getProperty("user.dir")+"\\resultat\\" +input+"\\" + input + ".txt";
		String pathOutputCSV = System.getProperty("user.dir")+"\\resultat\\" +input+"\\"+ input + ".csv";
		String[] headers, readValues, previousReadValues;
		int timeColumnIndex = -1;
		String line;
		BufferedReader reader;
		
		List<String> lastValues = new ArrayList<String>();
		List<Integer> captorIndexes = new ArrayList<Integer>();

		List<Evenement> evReferents = new ArrayList<Evenement>();
		List<Evenement> evContraints = new ArrayList<Evenement>();
		
		Boolean updateReferents = false;
		
		ContrainteTemporelle CT = ContrainteTemporelle.NCT();
		
		Timestamp lastTime, currentTime = null;

		try{
			Files.createDirectories(Paths.get(pathOutput).getParent());
			Files.createFile(Paths.get(pathOutput));
			Files.createFile(Paths.get(pathOutputCSV));
			System.out.println("Creation des fichiers");
		}catch(FileAlreadyExistsException e){
			System.out.println("Ecrasement des fichiers");
		}

		PrintWriter out = new PrintWriter(pathOutput);
		PrintWriter outCSV = new PrintWriter(pathOutputCSV);

		
		try{

			int nbligne = (int)Files.lines(Paths.get(pathInput)).count();

			reader = new BufferedReader(new FileReader(pathInput));
			headers = reader.readLine().split(delimiter);
			timeColumnIndex = Arrays.asList(headers).indexOf(timeColumnName);
			line = reader.readLine();
			previousReadValues = line.split(delimiter);
			for(int i = 0; i < previousReadValues.length; i++){
				String value = previousReadValues[i];
				lastValues.add(value);
				if(value.equals(FalseValue) || value.equals(TrueValue)){
					captorIndexes.add(i);
				}
			}

			evReferents.add(Evenement.In());
			captorIndexes.remove(Integer.valueOf(timeColumnIndex));
			lastTime = stringToTimestamp(previousReadValues[timeColumnIndex]);
			boolean first = true;


			BufferedImage b = new BufferedImage(captorIndexes.size(), nbligne - 1, BufferedImage.TYPE_INT_RGB);
			BufferedImage bCompressed = new BufferedImage(captorIndexes.size(), nbligne - 1, BufferedImage.TYPE_INT_RGB);

			int l = 0;
			int ll = 0;

			int j = 0;
			int jj = 0;
			for(int i: captorIndexes){
				int v = previousReadValues[i].equals(FalseValue) ? 0 : 255;
				b.setRGB(j, l, (v << 16) | (v << 8) | v);
				bCompressed.setRGB(jj, ll, (v << 16) | (v << 8) | v);
				j++;
				jj++;
			}
			jj=0;
			l++;
			ll++;

			while ( (line = reader.readLine()) != null && !line.equals("")) {
				readValues = line.split(delimiter);
				
				j = 0;
				for(int i: captorIndexes){
					int v = readValues[i].equals(FalseValue) ? 0 : 255;
					b.setRGB(j, l, (v << 16) | (v << 8) | v);
					j++;
				}
				l++;

				if(!Arrays.equals(getSubArray(readValues, captorIndexes), getSubArray(previousReadValues, captorIndexes))){
					//Si de nouveau evenements arrivent
					currentTime = stringToTimestamp(readValues[timeColumnIndex]);
					int t = (int)diffInCenti(lastTime, currentTime);
					CT = first ? ContrainteTemporelle.NCT() : new ContrainteTemporelle(t, t);
					updateReferents = true;
					first = false;
				}else{
					//Sinon aller à la prochaine ligne
					continue;
				}

				//chercher celui/ceux qui a/ont changé
				jj = 0;
				for(int i: captorIndexes){
					
					int v = readValues[i].equals(FalseValue) ? 0 : 255;
					bCompressed.setRGB(jj, ll, (v << 16) | (v << 8) | v);
					jj++;

					if(!readValues[i].equals(previousReadValues[i])){

						String type = "ERROR";
						if(previousReadValues[i].equals(FalseValue)){ type = "RE"; }
						if(previousReadValues[i].equals(TrueValue)){ type = "FE"; }

						out.print(type +"_"+ headers[i] +"   \t"+ stringToTimestamp(readValues[timeColumnIndex]));
						
						//si on appartient à la black list on ne l'ajoute juste pas dans les contrai,ts
						if(c.blackList.contains(type +"_"+ headers[i])){
							out.println("\tBLACK LIST");
							continue;
						}
						else{
							out.println("");
						}
						
						//ajouter dans la liste le nouveau evenements à ajouter
						evContraints.add(new Evenement(type, headers[i]));
					}
				}
				ll++;

				//quand tout les niuveaux contraints sont enregistré envoyer à l'ajout les contraint et leur referents
				c.add(evReferents, evContraints, CT);

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

			// Path p = Paths.get(System.getProperty("user.dir")+"\\resultat\\"+input+".jpg");
			// if ( Files.exists(p) ){
			// 	Files.delete(p);
			// }
			
			File img = new File(System.getProperty("user.dir")+"\\resultat\\"+input+"\\"+input+".png");
			ImageIO.write(b, "png", img);

			File img2 = new File(System.getProperty("user.dir")+"\\resultat\\"+input+"\\"+input+"_compressed.png");
			BufferedImage resized = new BufferedImage(captorIndexes.size(), ll, BufferedImage.TYPE_INT_RGB);
			for(int h = 0; h < resized.getHeight(); h++){
				for(int w = 0; w < resized.getWidth(); w++){
					resized.setRGB(w, h, bCompressed.getRGB(w, h));
				} 
			} 
			ImageIO.write(resized, "png", img2);

			List<String> captors = new ArrayList<String>();
			for(int columnIndex: captorIndexes){
				captors.add(headers[columnIndex]);
			}
			out.println("\n\n\n===========================================================================================");
			out.println("==================================== Règles ===============================================");
			out.println("===========================================================================================\n\n\n");
			out.println(c);
			out.println("\n\n\n============================================================================================");
			out.println("================================= Factorisation ============================================");
			out.println("============================================================================================\n\n\n");
			out.println(Condition.toString(c.computeSTC()));
			out.println("\n\n\n============================================================================================");
			out.println("=================================== Symptomes ==============================================");
			out.println("============================================================================================\n\n\n");

			for(ArrayList<Integer> codes: c.encodeAll(captors)){
				outCSV.print("Normal");
				codes.forEach(value -> outCSV.print("," + value));
				outCSV.println();
			}
			
			for(int i = 0; i < c.archive.size(); i++){
				Map<String, ArrayList<Triplet>> badrules = c.createBadRulesV1(i);
				
				out.println();
				for(Map.Entry<String, ArrayList<Triplet>> rule : badrules.entrySet()){
					out.print(rule.getKey() + " ==> " );
					rule.getValue().forEach(t -> out.print(" * " + t.toString()));
					out.println();
				}
				
				for(Map.Entry<String, ArrayList<Triplet>> rule : badrules.entrySet()){
					outCSV.print(rule.getKey());
					Condition.encode(rule.getValue(), captors).forEach(value -> outCSV.print("," + value));
					outCSV.println();
				}
			}
			

			//System.out.println(c.previous(0, 1));
			
			out.close();
			outCSV.close();

			if(displayGraph){
				System.setProperty("org.graphstream.ui", "swing"); 
				System.setProperty("gs.ui.renderer", "org.graphstream.ui.j2dviewer.J2DGraphRenderer");
				String styleSheet = "";
				try { styleSheet = Files.readString(Paths.get(System.getProperty("user.dir") + "\\style.txt")); }
				catch (Exception e) { e.printStackTrace(); }
				c.graph.setAttribute("ui.stylesheet", styleSheet);
				c.graph.setAttribute("ui.quality", graphQuality);
				c.graph.setAttribute("ui.antialias", graphAntialias);
				// c.graph.setAttribute("layout.stabilization-limit", 0);
				// c.graph.setAttribute("layout.force", 1);
				c.graph.display();
			}

		
			

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

	public static String[] getSubArray(String[] a, List<Integer> indexes){
		List<String> ret = new ArrayList<String>();

		for(int i: indexes){
			ret.add(a[i]);
		}

		return ret.toArray(String[]::new);
	}
}