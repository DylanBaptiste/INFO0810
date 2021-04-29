import java.util.*;
import java.io.*;
import java.sql.Timestamp;
// import java.time.format.*;
// import java.time.*;
public class Main {

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

	public static Timestamp stringToTimestamp(String d){
		String[] v;
		String date_time = d.substring(19);
		date_time = date_time.substring(0, date_time.length() - 2);
		v = date_time.split(", ");
		return Timestamp.valueOf(v[0]+"-"+v[1]+"-"+v[2]+" "+v[3]+":"+v[4]+":"+v[5]+"."+v[6]);
	}

	//todo si utile faire return <hasmap<String, List<String>> ou avec une structure perso
	/*public static List<String> getLigne(HashMap<String, List<String>> record, int i){
		Set<String> keys = record.keySet();
		List<String> values = new ArrayList<String>();
		for(String key: keys){
			values.add(record.get(key).get(i));
		}
		return values;
	}*/

	public static boolean isLignesEqual(HashMap<String, List<String>> record, int l1, int l2, String columnToIgnor){
		Set<String> keys = record.keySet();
		//keys.remove(columnToIgnor);
		for(String key: keys){
			if(key == columnToIgnor){ continue; }
			if(record.get(key).get(l1) != record.get(key).get(l2)){
				return false;
			}
		}
		return true;
	}

	public static void removeLigne(HashMap<String, List<String>> record, int i){
		Set<String> keys = record.keySet();
		for(String key: keys){
			record.get(key).remove(i);
		}
		return;
	}

	public static String[] getSubArray(String[] a, List<Integer> indexes){
		List<String> ret = new ArrayList<String>();

		for(int i: indexes){
			ret.add(a[i]);
		}

		return ret.toArray(String[]::new);
	}
	

    public static void main(String[] args) {
		Condition c = new Condition();
		String delimiter = ";";
		
		//String path = "C:\\Cours\\Semestre 8\\INFO0810\\doc_dylan.csv";
		String path = "C:\\Cours\\Semestre 8\\INFO0810\\doc_dylan_simulatane.csv";

		String[] headers;
		String line;
		BufferedReader reader;
		String[] readValues;
		String[] previousReadValues;
		//List<String> currentValues = new ArrayList<String>();
		List<String> lastValues = new ArrayList<String>();
		String timeColumnName = "Temps";
		int timeColumnIndex = -1;
		List<Integer> captorIndexes = new ArrayList<Integer>();

		List<Evenement> evReferents = new ArrayList<Evenement>();
		List<Evenement> evContraints = new ArrayList<Evenement>();
		Boolean updateReferents = false;
		Evenement contraint = null;
		ContrainteTemporel nct = new ContrainteTemporel(-1, -1);
		ContrainteTemporel CT = null;
		Timestamp lastTime = null;
		Timestamp currentTime = null;

		try{
			reader = new BufferedReader(new FileReader(path));
			//csv header -> hashmap keys
			headers = reader.readLine().split(delimiter);
			timeColumnIndex = Arrays.asList(headers).indexOf(timeColumnName);

			line = reader.readLine();
			previousReadValues = line.split(delimiter);
			for(int i = 0; i < previousReadValues.length; i++){
				String value = previousReadValues[i];
				lastValues.add(value);
				try {
					Float.parseFloat(value);
					continue; //si ont peut le parser on l'ignore
				}
				catch (NumberFormatException e) {
					captorIndexes.add(i);
				}
			}

			captorIndexes.remove(Integer.valueOf(timeColumnIndex));
			lastTime = stringToTimestamp(previousReadValues[timeColumnIndex]);

			
			while ( (line = reader.readLine()) != null && !line.equals("")) {

				readValues = line.split(delimiter);
				
				if(!Arrays.equals(getSubArray(readValues, captorIndexes), getSubArray(previousReadValues, captorIndexes))){
					currentTime = stringToTimestamp(readValues[timeColumnIndex]);
					int t = (int)diffInMicro(lastTime, currentTime);
					CT = new ContrainteTemporel(t, t);
					updateReferents = true;
				}
				for(int i: captorIndexes){
					if(!readValues[i].equals(previousReadValues[i])){
						String type = "ERROR";
						if(previousReadValues[i].equals("false")){
							type = "RE";
						}
						if(previousReadValues[i].equals("true")){
							type = "FE";
						}

						System.out.println(type +"_"+ headers[i] +"  \t"+ stringToTimestamp(readValues[timeColumnIndex]));

						contraint = new Evenement(type, headers[i]);
						evContraints.add(contraint);

						if(evReferents.size() == 0){
							c.add(new Triplet(new Evenement(null, headers[i]), contraint, nct));
						}else{
							for(Evenement referent: evReferents){
								c.add(new Triplet(referent, contraint, CT));
							}
						}
					}

					
					
				}
				previousReadValues = readValues.clone();
				
				
				if(updateReferents == true){
					evReferents = new ArrayList<Evenement>();
					for(Evenement e: evContraints){
						evReferents.add(e);
					}
					evContraints = new ArrayList<Evenement>();
					updateReferents = false;
					lastTime = stringToTimestamp(previousReadValues[timeColumnIndex]);

				}
				

				
			}

			
			System.out.println("\n" + c + "\n");


			System.out.println("\n" + Condition.toString(c.computeSTC()));
			


			/*
			List<Triplet> stc = new ArrayList<Triplet>();
			int modulo = 0;
			List<String> evsR = new ArrayList<String>(); 
			boolean first = true;
			boolean retry = true;
			for(Triplet t1: c.triplets){
				if(first){
					evsR.add(t1.contraint.toString());
					first = false;
					modulo++;
					continue;
				}
				if(retry && !t1.contraint.toString().equals(evsR.get(0))){
					evsR.add(t1.contraint.toString());
					modulo++;
				}else{
					retry = false;

					for(int i = modulo; i < c.triplets.size(); i++){
						int indice = (i+1) % (modulo + 1);
						String tester = evsR.get(indice);
						String test = c.triplets.get(i).contraint.toString();
						if(!evsR.get(0).equals(test) && !tester.equals(test)){
							modulo = i + 1;
							evsR.add(test);
							System.out.println(" ");
							break;
						}

						evsR.add(test);
						
					}
					
				}
				for(Triplet t2: c.triplets){
					if(t1 != t2 && t1.referent.type == t2.referent.type && t1.referent.contrainte == t2.referent.contrainte){
						stc.add(new Triplet(t1.referent, t2.contraint, new ContrainteTemporel(Integer.min(t1.ct.range[0], t2.ct.range[0]), Integer.max(t1.ct.range[1], t2.ct.range[1]))));
					}
				}

			}
			*/
			/*String s = "";
			String inter = " * ";
			for(Triplet t: stc){
				s += t + inter;
			}
			System.out.println(s.substring(0, s.length() - inter.length()));
			*/

			reader.close();
			//System.out.println("\nmodulo => " + modulo);
		}
		catch(FileNotFoundException e){
			System.err.println(e.getMessage());
			System.exit(0);
		}
		catch(IOException e){
			System.err.println(e.getMessage());
			System.exit(0);
		}

		// ContrainteTemporel ct1 = new ContrainteTemporel();
		// ContrainteTemporel ct2 = new ContrainteTemporel();
		// ContrainteTemporel ct3 = new ContrainteTemporel(2);
		// ContrainteTemporel ct4 = new ContrainteTemporel(3);
		// Evenement ev1 = new Evenement(null, "A");
		// Evenement ev2 = new Evenement("FE", "A");
		// Evenement ev3 = new Evenement("RE", "B");
		// Evenement ev4 = new Evenement("RE", "A");
		// Evenement ev5 = new Evenement("RE", "B");
		// Evenement ev6 = new Evenement("RE", "A");
		// Evenement ev7 = new Evenement("RE", "B");
		// Evenement ev8 = new Evenement("RE", "A");

		// Triplet t1 = new Triplet(ev1, ev2, ct1);
		// Triplet t2 = new Triplet(ev3, ev4, ct2);
		// Triplet t3 = new Triplet(ev5, ev6, ct3);
		// Triplet t4 = new Triplet(ev7, ev8, ct4);

		// Condition c = new Condition();
		// c.add(t1);
		// c.add(t2);
		// c.add(t3);
		// c.add(t4);
		// System.out.println(c);
		// System.out.println(c.getPreviousTriplets(1));
		// System.out.println(c.getPreviousTriplets(2));
		// System.out.println(c.getPreviousTriplets(3));

		/*
		t1.referent.type = "test";
		System.out.println(c);
		*/
    }
}