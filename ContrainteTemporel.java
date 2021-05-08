import java.util.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ContrainteTemporel {

	static int ct_count = 0;
	private int id;
	int range[];

    public ContrainteTemporel(int min, int max){
		//assert min <= max && min >= 0 && max >= 0;
		this.id = ++ct_count;
		this.range = min == -1 || max == -1 ? null : new int[]{min, max};
	}

	public int getId(){ return this.id; }
	
	public int getMin(){ return this.range == null ? -1 : this.range[0]; }
	public int getMax(){ return this.range == null ? -1 : this.range[1]; }
	public void setMin(int value){ if(this.range != null){this.range[0] = value; } }
	public void setMax(int value){ if(this.range != null){this.range[1] = value; } }
	public int[] getRange(){
		return this.range == null ? null : this.range.clone();
	}

	public static ContrainteTemporel NCT(){
		return new ContrainteTemporel(-1, -1);
	}

	public boolean updateRange(ContrainteTemporel other){
		if(other.getRange() == null || this.getRange() == null){ return false; }
		this.range[0] = Math.min(this.range[0], other.getMin());
		this.range[1] = Math.max(this.range[1], other.getMax());
		return true;
	}

	public String toString(){
		if(this.range == null){
			return "nct"/* + this.id*/;
		}
		if(this.range[0] == this.range[1]){
			return Integer.toString(this.range[0])/* + '_' + this.id*/;
		}
		return Arrays.toString(this.range)/* + '_' + this.id*/;
	}

	public boolean isNct(){
		return this.range == null;
	}

	private void setId(int id){
		this.id = id;
	}

	@Override
	public ContrainteTemporel clone(){
		if(this.isNct()){
			ContrainteTemporel clone = ContrainteTemporel.NCT();
			clone.setId(this.id);
			ContrainteTemporel.ct_count--;
			return clone;
		}
		ContrainteTemporel clone = new ContrainteTemporel(this.getMin(), this.getMax());
		clone.setId(this.id);
		ContrainteTemporel.ct_count--;
		return clone;
	}

	/* * * * * * *
	 * * TESTS * *
	 * * * * * * */
	
	@Test
	public void testToString() {

		String s = this.toString();
		assertEquals(1,1);
		
		assertEquals(1, s);
	}

}