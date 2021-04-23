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

	public String toString(){
		if(this.type == null){
			return "In";
		}
		return this.type + "_" + this.contrainte;
	}
}