package tree;

import java.util.*;

public class Pattern{
	
	Signature signatures;
	List<Integer> sequence;

	Tree tree;

	List<List<Integer>> patterns;

	
	public Pattern(Signature signatures, List<Integer> sequence){
		this.signatures = signatures; 
		this.sequence = sequence; 

		this.tree = new Tree();

		for(int i = 0; i < signatures.size(); i++){
			this.tree.root.addChild(i).setOccurence(0);
		}

		for(Node n: this.tree.root.childrens){
			n.setOccurence(Collections.frequency(this.sequence, n.getValue()));
		}


		
		for(Node child: this.tree.root.childrens){
			computeTree(child);
		}

		evaluateTree(tree.root);

	}

	public void computeTree(Node node){

		computeNode(node);

		if(node.computeChildrenOccurence() != node.getOccurence()){
			node.unFertilised();
		}else{
			for(Node child: node.childrens){
				computeTree(child);
			}
		}
	}

	/**
	 * calcule iteration de la sequence à partir de n et les lie au parent de n
	 * @param n
	 */
	public void computeNode(Node n){

		List<Integer> seq = new ArrayList<Integer>();
		branchToSequence(n, seq);
		boolean seqComplete = true;

		for(int i = 0; i < sequence.size() - 1; i++){
			if(sequence.get(i) == seq.get(0)){
				for(int j = 1; j < seq.size(); j++){
					i++;
					if(i >= sequence.size() - 1 || sequence.get(i) != seq.get(j)){
						seqComplete = false;
						break;
					}
				}
				if(seqComplete){
					i++;
					n.addChild(sequence.get(i));
					// // deux version : sans i++;
					// n.addChild(sequence.get(i + 1));

				}
			}
			seqComplete = true;
		}
	}

	public void branchToSequence(Node n,  List<Integer> s){
		if(!n.isRoot()){
			branchToSequence(n.parent, s);
			s.add(n.getValue());
		}
	}

	public void evaluateTree(Node n){
		if(n.isRoot()){
			for(Node child: n.childrens){
				evaluateTree(child);
			}
			return;
		}

		if(!n.parent.isFertile()){ return; }
		
		if(n.parent.isRoot() || n.countChildrens() > 1){
			List<Integer> s = new ArrayList<Integer>();
			branchToSequence(n, s);
			System.out.print(n.getDepth() * n.getOccurence() + " => ");
			s.forEach(i -> System.out.print(i+" "));
			System.out.println();
			for(Node child: n.childrens){
				evaluateTree(child);
			}
			return;
		}

		if(!n.isFertile()){
			List<Integer> s = new ArrayList<Integer>();
			branchToSequence(n, s);
			System.out.print(n.getDepth() * n.getOccurence() + " => ");
			s.forEach(i -> System.out.print(i+" "));
			System.out.println();
		}else{
			for(Node child: n.childrens){
				evaluateTree(child);
			}
		}
		return;
		
	}
}