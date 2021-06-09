import java.util.*;

import Pattern.*;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.sql.Timestamp;

//FE_A1B2 RE_P1VP2 RE_P2E FE_P1S FE_A1B4


public class Main2 {

	public static String signature(String[] arr){
		return Arrays.toString(arr).replaceAll("[^0-9]", "");
	} 

    public static void main(String[] args) throws FileNotFoundException {

		/*========== Paramètres ==========*/
		
		/* fichier */
		String delimiter = ",";
		String input = "test"; //convoyeur //input_file //test
		String timeColumnName = "Temps";
		String TrueValue = "1";
		String FalseValue = "0";
		
		/* graph */
		boolean displayGraph = true;
		int graphQuality = 0; /*0 -> 4*/
		boolean graphAntialias = false;
		
		/*=================================*/


		String pathInput = System.getProperty("user.dir")+"\\data\\" + input+"\\"+input + ".csv";
		String[] headers, readValues = null, previousReadValues = null;
		int timeColumnIndex = -1;
		String line;
		BufferedReader reader;
		
		List<String> lastValues = new ArrayList<String>();
		List<Integer> captorIndexes = new ArrayList<Integer>();

		List<Evenement> evReferents = new ArrayList<Evenement>();
		List<Evenement> evContraints = new ArrayList<Evenement>();
		
		Boolean updateReferents = false;
		
		
		Timestamp lastTime, currentTime = null;

		Signature signatures = new Signature();
		List<Integer> sequence = new ArrayList<Integer>();
		
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
			
			String signa = signature(getSubArray(previousReadValues, captorIndexes));
			signatures.add(signa);
			sequence.add(signatures.indexOf(signa));

			while ( (line = reader.readLine()) != null && !line.equals("")) {
				readValues = line.split(delimiter);
				if(!signa.equals( signature(getSubArray(readValues, captorIndexes)))){
					signa = signature(getSubArray(readValues, captorIndexes));
					signatures.add(signa);
					sequence.add(signatures.indexOf(signa));
				}
			}

			//List<String> signatures = new ArrayList<String>(signaturesSet);

			Pattern p = new Pattern(signatures, sequence);


			List<String> captors = new ArrayList<String>();
			for(int columnIndex: captorIndexes){
				captors.add(headers[columnIndex]);
			}
			
			

			if(displayGraph){
				System.setProperty("org.graphstream.ui", "swing"); 
				System.setProperty("gs.ui.renderer", "org.graphstream.ui.j2dviewer.J2DGraphRenderer");
				String styleSheet = "";
				try { styleSheet = Files.readString(Paths.get(System.getProperty("user.dir") + "\\style.txt")); }
				catch (Exception e) { e.printStackTrace(); }

				Node.graph.setAttribute("ui.stylesheet", styleSheet);
				Node.graph.setAttribute("ui.quality", graphQuality);
				Node.graph.setAttribute("ui.antialias", graphAntialias);
				// c.graph.setAttribute("layout.stabilization-limit", 0);
				// c.graph.setAttribute("layout.force", 1);
				Node.graph.display();
				
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