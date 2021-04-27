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
	

    public static void main(String[] args) {
		Condition c = new Condition();
		String delimiter = "\t";
		String path = "C:\\Cours\\Semestre 8\\INFO0810\\test2.csv";
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
		Timestamp start = null;
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
			start = stringToTimestamp(previousReadValues[timeColumnIndex]);

			while ( (line = reader.readLine()) != null ) {
				readValues = line.split(delimiter);
				Timestamp currentTime = stringToTimestamp(readValues[timeColumnIndex]);
				int t = (int)diffInMicro(start, currentTime);
				CT = new ContrainteTemporel(t, t);
				//System.out.println("\n" + Arrays.toString(previousReadValues) +"\n"+ Arrays.toString(readValues));
				for(int i: captorIndexes){
					// System.out.println("\n" + (previousReadValues[i]) +"\n"+ (readValues[i]));
					if(!readValues[i].equals(previousReadValues[i])){
						String type = "ERROR";
						updateReferents = true;
						if(previousReadValues[i].equals("false")){
							type = "RE";
						}
						if(previousReadValues[i].equals("true")){
							type = "FE";
						}

						System.out.println(type +"_"+ headers[i] + readValues[timeColumnIndex]);

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
				}
				

				
			}
			System.out.println("\n" + c);

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