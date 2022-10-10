# JMTiE
A Joint Multi-trajectory Search in Encrypted Outsourced Database

This is the source code for the paper: [Xin, Zhang, et al. "JMTiE: A Joint Multi-trajectory Search in Encrypted Outsourced Database" ] 


## Data
Our 3 datasets are NYK dataset, Shanghai dataset and Beijing dataset.
NYK dataset is [here](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)[1].
Shanghai dataset is [here](https://cse.hkust.edu.hk/scrg/)[2].
Beijing dataset is [here](https://www.microsoft.com/en-us/download/details.aspx?id=52367)[3-5].
All 3 datasets are download to file DatePreprocess.

## Prerequisites:
Python 3.9


## How to use

### Data preprocessing

* For NYK database, rename "dataset_TSMC2014_NYC.txt" by "dataset_TSMC2014_NYC.xlsx" and stored it in file 'DataPreprocess'.

* For Shanghai database, run the following code in file 'DataPreprocess' to preprocess the dataset:

 ```
 shanghaipredapro.py
 ```
 Then you will get a file named 'shanghai.feather'.
 
* For Beijing database, run the following code in file 'DataPreprocess' to preprocess the dataset:

 ```
 Beijingpredapro.py
 ```
 Then you will get a file named ''beijing.feather''.

### Partition data space and secure knn search

* Run the following code in file 'RBFtreeBuildandSearch' to partition dataset to several blocks. (1 block in NYK dataset, 4 blocks in Shanghai dataset, 64 blocks in Beijing dataset.)
```
parBlockStep.py
```

* Run the following code to in file 'RBFtreeBuildandSearch' to build a RBF tree for points in every block, random sample a query point and generate a search range, and search the knn points for the query points.

```
RBFindexbuidandsearch.py
```

### JMT(ie) search in table of knn points
* Run the following code in file '(e-)jmt(ie)' to sample the query trajectory and the corresponding table includes knn points for all points in the query trajectory.
```
QueryTable.py
```

* JMT search without expanding search region
 + If the limited number of sub-trajectories is 1, run the following code in file '(e-)jmt(ie)' to  get a trajectrory that has the max one-to-one trajectory similarity with the query trajectory.
 ```
 JMT(iE).py
 ```

 + If the limited number of sub-trajectories is more than 1, run the following code in file '(e-)jmt(ie)' to get the joint trajectory that has the max one-to-multi trajectory simialrity with the query trajectory.
 ```
 single(iE).py
 ```
* JMT search with expanding search region
 + If the limited number of sub-trajectories is 1, run the following code in file '(e-)jmt(ie)' to  get a trajectrory that has the max one-to-one trajectory similarity with the query trajectory.
 ```
 e-JMT(iE).py
 ```

 + If the limited number of sub-trajectories is more than 1, run the following code in file '(e-)jmt(ie)' to get the joint trajectory that has the max one-to-multi trajectory simialrity with the query trajectory.
 ```
 singleIKNN(iE).py
 ```

### Evaluation the time cost of SSED and SXOR
* Run the follow code to evaluate the time cost of once SSED. 
```
 SSED.py
 ```

* Run the follow code to evaluate the time cost of once SXOR. 
```
 SXOR.py
 ```

* Run the follow code in file '(e-)jmt(ie)' to evaluate the number of times that SSED and SXOR need to be used in current table of knn points.
```
 computeSS.py
 ```

Note that we import file 'computeSS.py' to files 'single(iE).py', 'singleIKNN(iE).py', 'JMT(iE).py' and 'e-JMT(iE).py' to evalution the number of times that SSED and SXOR need to be used
in every file. In our evalution the time cost of search in encrypted database is roughly equal to the sum of the time cost of search in plain-text database and the time cost of SSED and SXOR.

References
[1]Dingqi Yang, Daqing Zhang, Vincent W. Zheng, Zhiyong Yu. Modeling User Activity Preference by Leveraging User Spatial Temporal Characteristics in LBSNs. IEEE Trans. on Systems, Man, and Cybernetics: Systems, (TSMC), 45(1), 129-142, 2015. 
[2]iyuan Liu, Yunhuai Liu, Lionel M. Ni, Jianping Fan, and Minglu Li. Towards Mobility-based Clustering. In Proc. Of ACM KDD 2010.
[3]Yu Zheng, Lizhu Zhang, Xing Xie, Wei-Ying Ma. Mining interesting locations and travel sequences from GPS trajectories. In Proceedings of International conference on World Wild Web (WWW 2009), Madrid Spain. ACM Press: 791-800. 
[4] Yu Zheng, Quannan Li, Yukun Chen, Xing Xie, Wei-Ying Ma. Understanding Mobility Based on GPS Data. In Proceedings of ACM conference on Ubiquitous Computing (UbiComp 2008), Seoul, Korea. ACM Press: 312-321.
[5] Yu Zheng, Xing Xie, Wei-Ying Ma, GeoLife: A Collaborative Social Networking Service among User, location and trajectory. Invited paper, in IEEE Data Engineering Bulletin. 33, 2, 2010, pp. 32-40.
