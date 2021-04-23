import java.util.*;

public class ContrainteTemporel {
	static int ct_count = 0;
	private int id;
	int range[];

    public ContrainteTemporel(int min, int max){
		assert min <= max && min >= 0 && max >= 0;

		this.id = ++ct_count;
		this.range = new int[]{min, max};
	}

	public ContrainteTemporel(int dt){
		assert dt >= 0;

		this.id = ++ct_count;
		this.range = new int[]{dt, dt};
	}

	public ContrainteTemporel(){ this.range = null; }

	public int getId(){ return this.id; }
	
	public int getMin(){ return this.range == null ? -1 : this.range[0]; }
	public int getMax(){ return this.range == null ? -1 : this.range[1]; }
	public int[] getRange(){
		return this.range == null ? null : this.range.clone();
	}

	public String toString(){
		if(this.range == null){
			return "nct";
		}
		if(this.range[0] == this.range[1]){
			return Integer.toString(this.range[0]);
		}
		return Arrays.toString(this.range);
	}
}