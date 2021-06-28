# Deep Graph Attention Neural Network (DGANN) 

## Citation:
```
TEX CODE
```
## What is DGANN?
We presented a Deep Graph Attention Neural Network (DGANN) to evaluate and rank protein docking candidate models. DGANN learns inter-residue physio-chemical properties and structural fitness across the two protein monomers in a docking model and generates their probabilities of near-native models. On the ZDOCK decoy benchmark, our DGANN outperformed the ranking provided by ZDOCK in terms of ranking good models into the top selections. Furthermore, we conducted comparative experiments on an independent testing dataset, and the results also demonstrated the superiority and generalization of our proposed method.

**Flowchart of DGANN method.** 
 
![](https://github.com/coffee19850519/PPDocking/blob/master/figure/FIG1.jpg)   

The pipeline of DGANN includes three stages:  
(1) Data preprocessing stage (in blue): The PDB files of candidate docking models were transformed into graph structures, where each node is a residue, and each edge connects the two residues carrying any atoms within 5 Å interatomic distance. Then, we encoded each residue (node) by its physico-chemical properties and conservation profiles (detailed in Method C). All candidate docking models were labeled as positive or negative samples according to the Critical Assessment of PRedicted Interactions (CAPRI) criteria (defined in Methods E). We split all collected protein complexes into training and test sets for 5-fold cross-validation.

(2) DGANN modeling stage (in orange): Aiming to address the imbalanced issue in QA, we bootstrapped 100 balanced training sub-sets to train our proposed DGANN to get 100 classifiers.

(3) Ensemble learning stage (in green): An ensemble learning strategy was employed to integrate the outputs from the 100 classifiers. When assessing a protein docking model, we applied all classifiers to predict its quality scores and took their maximum score as the final prediction.  

**DGANN architecture.**  

![](https://github.com/coffee19850519/PPDocking/blob/master/figure/FIG2.jpg)

 DGANN consists of three modules:  
(1) GAT module: Docking models are first represented as graphs (blue nodes denote the residues from the protein in blue, and red nodes represent the residues from the protein in red), where the nodes have 26-dimensional attributes. Then, two stacked GAT layers are designed to model neighboring residue interactions and local structural information. For instance, the node embedding of GLU at the second GAT layer comes from the attention weighted aggregation of its neighbors ARG, PRO and VAL, whose embeddings are also aggregated by their neighbors at the first GAT layer. Through these processes, residue interactions from internal and external residues are modeled in each node embedding. Furthermore, at the first GAT layer, each aggregated node embedding is mapped to 32 dimensions, while at the second GAT layer the aggregated 32-dimensional node embeddings are mapped to a scalar for each node, which is deemed as the importance of the node over the whole graph.
 
(2) Top-k-pooling module: To obtain a fixed size of graph-level representations, all residues at protein-protein interfaces are sorted by their importance, and the top k nodes are kept. And the two GAT outputs of the selected k nodes are concatenated to form graph-level representations of protein-protein interfaces.

(3) QA scoring module: The graph representation of a docking model is fed into a 1-D convolutional layer and a fully connected layer to generate a flattened feature vector. Finally, a sigmoid function is applied to compute its probability of a native-like model.

## Usage
```
python imput_data.py
```
### Environment
conda 10.2  
python 3.7  
pytorch 1.7.0  
torch-geometric 1.7.0  

