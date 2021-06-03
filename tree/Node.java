package tree;

import java.util.*;

import org.graphstream.graph.Graph;
import org.graphstream.graph.implementations.*;

public class Node{

    private static int node_count = 0;
	public int id;
	
	List<Node> childrens;
	Node parent;
    private int occurence;
    private boolean fertile;
    private int value;
    private int depth;
    public static Graph graph = new SingleGraph("pattern");

	
	public Node(Node parent, int value){
        this.id = ++node_count;

		this.childrens = new ArrayList<Node>();
		this.parent = parent;
        this.occurence = 1;
        this.fertile = true;
        this.value = value;

        if(this.parent == null){
            depth = 0;
            Node.graph.addNode(Integer.toString(this.id)).setAttribute("ui.class", "start");
        }else{
            depth = parent.depth + 1;
        }
	}

    public int getDepth(){ return this.depth; }

	public Node addChild(int v){

        for(Node child: this.childrens){
            if(child.getValue() == v){
                child.setOccurence(child.getOccurence() + 1);
                return child;
            }
        }

        Node n = new Node(this, v);
        Node.graph.addNode(Integer.toString(n.id)).setAttribute("ui.label", v+" ("+n.occurence+")");
        Node.graph.addEdge(this.id + "" + (n.id) , Integer.toString(this.id), Integer.toString(n.id), true);
        this.childrens.add(n);

        if(this.countChildrens() > 1 && !this.isRoot()){
            Node.graph.getNode(Integer.toString(this.id)).setAttribute("ui.class", "division");
        }
        return n;
	}

    public int getOccurence(){ return this.occurence; }
    public void setOccurence(int occurence){
        this.occurence = occurence;
        Node.graph.getNode(Integer.toString(this.id)).setAttribute("ui.label", ""+value+" ("+this.occurence+")");
    }

    public int getValue(){ return this.value; }

    public boolean isRoot(){ return this.parent == null; }

    public boolean isFertile(){ return this.fertile; }

    public void unFertilised(){
        Node.graph.getNode(Integer.toString(this.id)).setAttribute("ui.class", "unfertile");
        this.fertile = false;
    }


    public int countChildrens(){
        return this.childrens.size();
    }
    

    public int computeChildrenOccurence(){
        int ret = 0;
        for(Node child: this.childrens){
            ret += child.getOccurence();
        }
        return ret;
    }

    @Override
    public String toString(){
        return this.value + "(" + this.occurence + ") " + this.countChildrens() + " enfants" ;
    }
}