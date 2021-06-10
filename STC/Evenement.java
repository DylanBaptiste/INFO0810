package STC;

import java.util.List;

public class Evenement {
	static int ev_count = 0;
	private int id;
	String type;
	String contrainte;

    public Evenement(String type, String contrainte){
		this.id = ++ev_count;
		this.type = type;
		this.contrainte = contrainte;
	}

	public static Evenement In(){
		return new Evenement(null, null);
	}

	public static Evenement S(String composant){
		return new Evenement(null, composant);
	}

	public int getId(){ return this.id; }

	public boolean isS(){
		return this.type == null && this.contrainte != null;
	}

	public boolean isIn(){
		return this.type == null && this.contrainte == null;
	}

	public boolean equals(Evenement other){
		return !this.isIn() && !other.isIn() && !this.isS() && !other.isS() && type.equals(other.type) && this.contrainte.equals(other.contrainte);
	}
	public boolean equalsId(Evenement other){
		return this.id == other.id;
	}
	private void setId(int id){
		this.id = id;
	}

	public int encodeT(){
		if(this.isIn()){
			return -1;
		}
		if(this.isS()){
			return -1;
		}
		return this.type.equals("FE") ? 0 : 1;
	}
	public int encodeC(List<String> captors){
		if(this.isIn()){
			return -1;
		}
		return captors.indexOf(this.contrainte);
	}

	public Evenement clone(){
		
		if(this.isIn()){
			Evenement clone = Evenement.In();
			clone.setId(this.id);
			Evenement.ev_count--;
			return clone;
		}
		Evenement clone = new Evenement(this.type, this.contrainte);
		clone.setId(this.id);
		Evenement.ev_count--;
		return clone;
	}

	public String toString(){
		if(this.isIn()){
			return "In"/* + '_' + this.id*/;
		}
		if(this.isS()){
			return "S_" + this.contrainte/* + '_' + this.id*/;
		}
		return this.type + "_" + this.contrainte/* + '_' + this.id*/;
	}
}