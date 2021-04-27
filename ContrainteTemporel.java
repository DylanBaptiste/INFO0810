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