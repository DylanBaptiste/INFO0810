import java.util.*;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.sql.Timestamp;

//FE_A1B2 RE_P1VP2 RE_P2E FE_P1S FE_A1B4

public class Main {

    public static void main(String[] args) throws FileNotFoundException {

		/*========== Paramètres ==========*/
		
		/* fichier */
		String delimiter = ",";
		String input = "input_file";
		String timeColumnName = "Temps";
		String TrueValue = "1";
		String FalseValue = "0";
		
		/* graph */
		boolean displayGraph = true;
		int graphQuality = 0; /*0 -> 4*/
		boolean graphAntialias = false;
		
		/*=================================*/

		Condition c = new Condition();

		String output = "resultat_" + input; 
		String pathInput = System.getProperty("user.dir")+"\\data\\" + input + ".csv";
		String pathOutput = System.getProperty("user.dir")+"\\resultat\\" + output + ".txt";
		String[] headers, readValues, previousReadValues;
		int timeColumnIndex = -1;
		String line;
		BufferedReader reader;
		
		List<String> lastValues = new ArrayList<String>();
		List<Integer> captorIndexes = new ArrayList<Integer>();

		List<Evenement> evReferents = new ArrayList<Evenement>();
		List<Evenement> evContraints = new ArrayList<Evenement>();
		
		Boolean updateReferents = false;
		
		ContrainteTemporel CT = ContrainteTemporel.NCT();
		
		Timestamp lastTime, currentTime = null;

		PrintWriter out = new PrintWriter(pathOutput);
		
		try{
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
			while ( (line = reader.readLine()) != null && !line.equals("")) {
				readValues = line.split(delimiter);
				
				if(!Arrays.equals(getSubArray(readValues, captorIndexes), getSubArray(previousReadValues, captorIndexes))){
					//Si de nouveau evenements arrivent
					currentTime = stringToTimestamp(readValues[timeColumnIndex]);
					int t = (int)diffInMicro(lastTime, currentTime);
					CT = first ? ContrainteTemporel.NCT() : new ContrainteTemporel(t, t);
					updateReferents = true;
					first = false;
				}else{
					//Sinon aller à la prochaine ligne
					continue;
				}

				//chercher celui/ceux qui a/ont changé
				for(int i: captorIndexes){
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

			out.println("\n\n\n=============================================================================================\n\n\n");
			out.println(c);
			out.println("\n\n\n=============================================================================================\n\n\n");
			out.println(Condition.toString(c.computeSTC()));
			out.close();

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
			c.graph.display(displayGraph);
			

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

	public static Timestamp stringToTimestamp(String d){
		String[] v;
		String date_time = d.substring(19);
		date_time = date_time.substring(0, date_time.length() - 2);
		v = date_time.split("; ");
		return Timestamp.valueOf(v[0]+"-"+v[1]+"-"+v[2]+" "+v[3]+":"+v[4]+":"+v[5]+"."+v[6]);
	}

	public static String[] getSubArray(String[] a, List<Integer> indexes){
		List<String> ret = new ArrayList<String>();

		for(int i: indexes){
			ret.add(a[i]);
		}

		return ret.toArray(String[]::new);
	}
}