----------------------------------------------------------------------------
# Understanding the network formation pattern for better link prediction
----------------------------------------------------------------------------

## Authors:
yujiating@amss.ac.cn

lywu@amss.ac.cn


## Overview:
- Link prediction using Multiple Order Local Information (MOLI) exploits the local information from the neighbors of different distances, with parameters that can be a prior-driven based on prior knowledge, or data-driven by solving an optimization problem on observed networks. 
- MOLI defined a local network diffusion process via random walks on the graph, resulting in better use of network information.
- We show that MOLI outperforms the other 11 widely used link prediction algorithms on 11 different types of simulated and real-world networks. We also conclude that there are different patterns of local information utilization for different networks, including social networks, communication 
networks, biological networks, etc.

You can reproduce the results of following five experiments of our paper, noting that the data should be decompressed before running the corresponding code.

1. Simulated networks;
2. Online Social networks;
3. European Email networks;
4. Drug-Drug Interaction (DDI) networks;
5. Protein-Protein Interaction (PPI) networks.


## Notes: 
1. The results of all experiments have been saved in the corresponding folder of  ./Results/
2. When solving the optimization problem, we can take several values for x first to see the value of the objective function, so that we can choose the initial point x0 reasonably, which can make the algorithm converge faster.
