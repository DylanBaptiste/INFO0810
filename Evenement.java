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

	public int getId(){ return this.id; }

	public static Evenement In(){
		return new Evenement(null, null);
	}

	public boolean isIn(){
		return this.type == null || this.contrainte == null;
	}

	public boolean equals(Evenement other){
		return !this.isIn() && !other.isIn() && type.equals(other.type) && this.contrainte.equals(other.contrainte);
	}
	public boolean equalsId(Evenement other){
		return this.id == other.id;
	}

	public String toString(){
		if(this.type == null){
			return "In"/* + '_' + this.id*/;
		}
		return this.type + "_" + this.contrainte/* + '_' + this.id*/;
	}
}