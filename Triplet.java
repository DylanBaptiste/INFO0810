
public class Triplet {
	static int tri_count = 0;
	private int id;
	Evenement referent = null;
	Evenement contraint = null;
	ContrainteTemporel ct = null;

    public Triplet(Evenement referent, Evenement contraint, ContrainteTemporel ct){
		this.id = ++tri_count;
		this.referent = referent;
		this.contraint = contraint;
		this.ct = ct;
	}

	public int getId(){ return this.id; }

	public Boolean isNct(){
		return this.referent.contrainte == null;
	}

	public String toString(){
		return "(" + this.referent.toString() + ", " + this.contraint.toString() + ", " + this.ct + ")"/* + '_' + this.id*/; 
	}

}