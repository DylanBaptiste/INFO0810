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
	
		/**
		 * supposiitons:
		 * - le fichier csv est complet (aucunne valeur man,quante)
		 * - le fichier à un header en prmeiere ligne avec le nom des capteurs
		 * - aucune colonne n'a un nom identique
		 * - les seul colonne de type string sont les capteur (en true/false) et le temps
		 * option: true value = "true", false value = "false" //todo permettre de changer ca
		 */
		String path = "C:\\Cours\\Semestre 8\\Stage\\Code\\test.csv";
		String line;
		String[] headers;
		String timeColumnName = "Time";
		HashMap<String, List<String>> records = new HashMap<String, List<String>>(){};
		System.out.println("Fichier:\n");
		try{
			BufferedReader reader = new BufferedReader(new FileReader(path));
			//csv header -> hashmap keys
			headers = reader.readLine().split("\t");
			System.out.println(Arrays.toString(headers));
			for(String header: headers){
				records.put(header, new ArrayList<String>());
			}

			while ( (line = reader.readLine()) != null ) {
				String[] values = line.split("\t");
				System.out.println(Arrays.toString(values));
				String value;
				for(int i = 0; i < values.length; i++){
					value = values[i];
					try {
						Float.parseFloat(value);
						continue; //si ont peut le parser on l'ignore
					}
					catch (NumberFormatException e) {
						switch (value) {
							case "true":
								records.get(headers[i]).add("1");
								break;
							case "false":
								records.get(headers[i]).add("0");
								break;
							default:
								//[datetime.datetime(2020, 8, 24, 12, 59, 39, 215267)]
								//notrmalement que le temps ici -> possibilité de fixé le nom de la clefs
								timeColumnName = headers[i];
								records.get(headers[i]).add(value);
								break;
						}
					}
				}
			}

			reader.close();

			//clean hasmap
			System.out.println("Clean hasmap:\n");
			Iterator<Map.Entry<String, List<String>>> itr = records.entrySet().iterator();
			while (itr.hasNext()){
				Map.Entry<String, List<String>> curr = itr.next();
				System.out.println(curr.getKey() + "\t => \t" + curr.getValue().isEmpty());
				if (curr.getValue().isEmpty()) {
					itr.remove();
				}
			}

			//convertir la colonne temps
			System.out.println("\nConvertir la colonne temps:\n");
			List<String> relativeTime = new ArrayList<String>();
			String[] currTime = records.get(timeColumnName).stream().toArray(String[]::new);

			
			//Temps relatif par rapport au precedent:
			/*
			relativeTime.add("0µs");
			for(int i = 0; i < currTime.length - 1; i++){
				relativeTime.add(diffInMicro(stringToTimestamp(currTime[i]), stringToTimestamp(currTime[i + 1])) + "µs");
			}*/
			Timestamp t1 = stringToTimestamp(currTime[0]);
			for(int i = 0; i < currTime.length; i++){
				Timestamp t2 = stringToTimestamp(currTime[i]);
				relativeTime.add(diffInMicro(t1, t2)+"");
			}
			records.remove(timeColumnName);
			records.put(timeColumnName, relativeTime);
			
			
			System.out.println("\nResult:\n");
			itr = records.entrySet().iterator();
			while (itr.hasNext()){
				Map.Entry<String, List<String>> curr = itr.next();
				System.out.println(curr.getKey() + "\t => \t" + curr.getValue());
			}
			System.out.println("\nSupprimer les doublons...\n");

			//todo remove deuplicate
			//System.out.println(isLignesEqual(records, 2, 3, timeColumnName));
			//removeLigne(records, 1);

			//todo make unit test !
			int numberElement = records.get(timeColumnName).size();
			int elmementRestant = numberElement - 1;
			for(int i = 0; i < elmementRestant;){
				if(isLignesEqual(records, i, i+1, timeColumnName)){
					removeLigne(records, i+1);
					elmementRestant--;
				}else{
					i++;
				}
			}

			System.out.println("\nResult:\n");
			itr = records.entrySet().iterator();
			while (itr.hasNext()){
				Map.Entry<String, List<String>> curr = itr.next();
				System.out.println(curr.getKey() + "\t => \t" + curr.getValue());
			}
		}
		catch(FileNotFoundException e){
			System.err.println(e.getMessage());
			System.exit(0);
		}
		catch(IOException e){
			System.err.println(e.getMessage());
			System.exit(0);
		}
		


		ContrainteTemporel ct1 = new ContrainteTemporel();
		ContrainteTemporel ct2 = new ContrainteTemporel();
		ContrainteTemporel ct3 = new ContrainteTemporel(2);
		ContrainteTemporel ct4 = new ContrainteTemporel(3);
		Evenement ev1 = new Evenement(null, "A");
		Evenement ev2 = new Evenement("FE", "A");
		Evenement ev3 = new Evenement("RE", "B");
		Evenement ev4 = new Evenement("RE", "A");
		Evenement ev5 = new Evenement("RE", "B");
		Evenement ev6 = new Evenement("RE", "A");
		Evenement ev7 = new Evenement("RE", "B");
		Evenement ev8 = new Evenement("RE", "A");

		Triplet t1 = new Triplet(ev1, ev2, ct1);
		Triplet t2 = new Triplet(ev3, ev4, ct2);
		Triplet t3 = new Triplet(ev5, ev6, ct3);
		Triplet t4 = new Triplet(ev7, ev8, ct4);

		Condition c = new Condition();
		c.add(t1);
		c.add(t2);
		c.add(t3);
		c.add(t4);
		System.out.println(c);
		System.out.println(c.getPreviousTriplets(1));
		System.out.println(c.getPreviousTriplets(2));
		System.out.println(c.getPreviousTriplets(3));

		/*
		t1.referent.type = "test";
		System.out.println(c);
		*/
    }
}