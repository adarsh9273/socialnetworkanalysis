# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as  sns

#reading the data
df = pd.read_csv('fbtrain.csv')

#Checking the missing and duplicate values
df.isnull().sum() #no missing data
df.duplicated().sum() #no duplicate values

#df.to_csv('fbtrainwoheader.csv',header=False,index=False)

import networkx as nx
g = nx.read_edgelist('fbtrainwoheader.csv',delimiter =',',create_using = nx.DiGraph(),nodetype=int)
print(nx.info(g))
'''
Type: DiGraph
Number of nodes: 1862220 #unique person
Number of edges: 9437519
Average in degree:   5.0679
Average out degree:   5.0679
'''
# =============================================================================
# VISUAL REPRESENTATION OF SUB-GRAPH 
# =============================================================================
import networkx as nx
#pd.read_csv('fbtrain.csv',nrows=50).to_csv('fbtrainwoheader_sample.csv',header=False,index=False)
subgraph = nx.read_edgelist('fbtrainwoheader_sample.csv',delimiter =',',create_using = nx.DiGraph(),nodetype=int)
print(nx.info(subgraph))
'''
Type: DiGraph
Number of nodes: 66
Number of edges: 50
Average in degree:   0.7576
Average out degree:   0.7576
'''
pos = nx.spring_layout(subgraph)
nx.draw(subgraph,pos,node_color = '#A0CBE2',edge_color='#00bb5e',width =1 ,edge_cmap =plt.cm.Blues,with_labels =True)

# =============================================================================
# EXPLORATORY DATA ANALYSIS
# =============================================================================
# =============================================================================
#no of followers for each person
# =============================================================================
indegree_dist = list(dict(g.in_degree()).values())
indegree_dist.sort()   
plt.plot(indegree_dist)     
plt.xlabel('Index No')
plt.ylabel('No Of Followers')        
        
list(g.in_degree())[:5]        
        
indegree_dist = list(dict(g.in_degree()).values())
indegree_dist.sort()   
plt.plot(indegree_dist[:1500000])     
plt.xlabel('Index No')
plt.ylabel('No Of Followers')        
                
sns.boxplot(data=indegree_dist,)       
plt.ylabel('No Of Followers')
        
#90-100 percentile
for i in range(11):
     print(90+i,'percentile value is',np.percentile(indegree_dist,90+i)) 
#99% of data having followers of 40 only.  
        
#99-100 percentile
for i in range(11):
     print(99+0.1*i,'percentile value is',np.percentile(indegree_dist,99+0.1*i))       
        
sns.distplot(indegree_dist)        
plt.xlabel('PDF of Indegree')
        
# =============================================================================
#No of people each person is following
# =============================================================================
outdegree_dist = list(dict(g.out_degree()).values())
outdegree_dist.sort()   
plt.plot(outdegree_dist)     
plt.xlabel('Index No')
plt.ylabel('No Of Followers')        
        
list(g.outdegree_dist())[:5]        
        
outdegree_dist = list(dict(g.out_degree()).values())
outdegree_dist.sort()   
plt.plot(outdegree_dist[:1500000])     
plt.xlabel('Index No')
plt.ylabel('No Of Followers')        
                
sns.boxplot(data=outdegree_dist)       
plt.ylabel('No Of Followers')
        
#90-100 percentile
for i in range(11):
     print(90+i,'percentile value is',np.percentile(outdegree_dist,90+i)) 
#99% of data having followers of 40 only.  
        
#99-100 percentile
for i in range(11):
     print(99+0.1*i,'percentile value is',np.percentile(outdegree_dist,99+0.1*i))       
        
sns.distplot(outdegree_dist)        
plt.xlabel('PDF of Indegree')
        
# =============================================================================
# No of persons those are not following anyone are        
# =============================================================================
sum(np.array(outdegree_dist)==0)#274512
100 * sum(np.array(outdegree_dist)==0)/len(outdegree_dist) #14.74%

# =============================================================================
#  No of persons having zero followers are       
# =============================================================================
sum(np.array(indegree_dist)==0)#188043
100 * sum(np.array(indegree_dist)==0)/len(indegree_dist) #10.1%       
        
# =============================================================================
# No of persons those are not following anyone and having zero followerss are        
# =============================================================================
count =0
for i in g.nodes():
    if len(list(g.predecessors(i)))==0:
        if len(list(g.successors(i)))==0:
            count+=1
#0

# =============================================================================
# both followers + following            
# =============================================================================
from collections import Counter
dict_in = dict(g.in_degree())
dict_out = dict(g.out_degree())
d = Counter(dict_in) + Counter(dict_out)
in_out_degree = np.array(list(d.values()))
        
in_out_degree.sort()   
plt.plot(in_out_degree)     
plt.xlabel('Index No')
plt.ylabel('No Of Followers')  

in_out_degree.sort()  
plt.plot(in_out_degree[:1500000])     
plt.xlabel('Index No')
plt.ylabel('No Of Followers')      
                
sns.boxplot(data=in_out_degree)       
plt.ylabel('No Of Followers')
        
#90-100 percentile
for i in range(11):
     print(90+i,'percentile value is',np.percentile(in_out_degree,90+i)) 
#99% of data having followers of 40 only.  
        
#99-100 percentile
for i in range(11):
     print(99+0.1*i,'percentile value is',np.percentile(sorted(in_out_degree),99+0.1*i))       
        
sns.distplot(in_out_degree)        
plt.xlabel('PDF of Indegree')       
        
#Min of no of followers + following is
in_out_degree.min()#1

#Max of no of followers + following is    
in_out_degree.max()#1579

#No of persons having followers + following less than 10 are
sum(in_out_degree<10)#1320326
     
#No of weakly connected components
len(list(nx.weakly_connected_components(g))) #45558
       
#weakly connected components wit 2 nodes
count =0
for i in list(nx.weakly_connected_components(g)):
    if len(i)==2:
        count+=1        
#32195

# =============================================================================
# POSING A PROBLEM AS CLASSIFICATION PROBLEM        
# =============================================================================
# =============================================================================
# Generating some edges which are not present in graph for supervised learning       
# =============================================================================
import random
import pickle
import csv
from tqdm import tqdm
r = csv.reader(open('fbtrainwoheader.csv','r'))
edges = dict()
for edge in r:
    edges[(edge[0], edge[1])] = 1
    
missing_edges =set([])
while(len(missing_edges)<9437519):
    a = random.randint(1,1862220)
    b = random.randint(1,1862220)
    tmp =edges.get((a,b),-1)
    if tmp==-1 and a!=b:
        try:
            if nx.shortest_path_length(g,source =a,target =b) >2:
                missing_edges.add((a,b))
            else:
                continue
        except:
            missing_edges.add((a,b))
    else:
        continue
'''Pickling the list into pickle'''
pickle.dump(missing_edges,open('missing_edges_final.p','wb'))
        
'''Loading the pickle'''
missing_edges = pickle.load(open('missing_edges_final.p','rb'))        
        
# =============================================================================
# Training and Test data split        
# =============================================================================
from sklearn.model_selection import train_test_split
X = pd.read_csv('fbtrain.csv')
y = pd.DataFrame(list(missing_edges),columns=['source_node', 'destination_node'])

# Splitting the dataset into the Training set and Test set
X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(X, np.ones(len(X)), test_size=0.2, random_state=9)
X_train_neg, X_test_neg, y_train_neg, y_test_neg = train_test_split(y, np.zeros(len(y)), test_size=0.2, random_state=9)


#removing header and saving
X_train_pos.to_csv('train_pos_after_eda.csv',header=False, index=False)
X_test_pos.to_csv('test_pos_after_eda.csv',header=False, index=False)
X_train_neg.to_csv('train_neg_after_eda.csv',header=False, index=False)
X_test_neg.to_csv('test_neg_after_eda.csv',header=False, index=False)

#visualising
train_graph = nx.read_edgelist('train_pos_after_eda.csv',delimiter=',',create_using=nx.DiGraph(), nodetype=int)
test_graph = nx.read_edgelist('test_pos_after_eda.csv',delimiter=',',create_using=nx.DiGraph(), nodetype=int)
print(nx.info(train_graph))
'''
Type: DiGraph
Number of nodes: 1780722
Number of edges: 7550015
Average in degree:   4.2399
Average out degree:   4.2399
'''
print(nx.info(test_graph))
'''
Type: DiGraph
Number of nodes: 1144623
Number of edges: 1887504
Average in degree:   1.6490
Average out degree:   1.6490
'''
# finding the unique nodes in the both train and test graphs
train_nodes_pos = set(train_graph.nodes())
test_nodes_pos = set(test_graph.nodes())

#no of people common in train and test
trY_teY =len(train_nodes_pos.intersection(test_nodes_pos)) #1063125

#no of people present in train but not present in test
trY_teN =len(train_nodes_pos - test_nodes_pos) #717597

#no of people present in test but not present in train
teY_trN =len(test_nodes_pos - train_nodes_pos) #81498

#% of people not there in Train but exist in Test in total Test data
100 * teY_trN/len(test_nodes_pos) #7.12 ->which creates a cold start problem

#final train and test data sets
X_train =X_train_pos.append(X_train_neg,ignore_index=True)
y_train = np.concatenate((y_train_pos,y_train_neg))
X_test =X_test_pos.append(X_test_neg,ignore_index=True)
y_test = np.concatenate((y_test_pos,y_test_neg))

#Saving it for future use
X_train.to_csv('train_after_eda.csv',header=False,index=False)
X_test.to_csv('test_after_eda.csv',header=False,index=False)
pd.DataFrame(y_train.astype(int)).to_csv('train_y.csv',header=False,index=False)
pd.DataFrame(y_test.astype(int)).to_csv('test_y.csv',header=False,index=False)

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
# =============================================================================
# Jaccard Distance
# =============================================================================
import networkx as nx
train_graph=nx.read_edgelist('train_pos_after_eda.csv',delimiter=',',create_using=nx.DiGraph(),nodetype=int)
#For followees
def jaccard_for_followees(a,b):
    try:
        if len(set(train_graph.successors(a)))==0 | len(set(train_graph.successors(b)))==0:
            return 0
        sim = (len(set(train_graph.successors(a)).intersection(set(train_graph.successors(b)))))/(len(set(train_graph.successors(a)).union(set(train_graph.successors(b)))))
        return sim
    except:
        return 0

#one test case
jaccard_for_followees(273084,1505602) #0
#node 1635354 not in graph 
jaccard_for_followees(273084,1505602) #0

#For followers
def jaccard_for_followers(a,b):
    try:
        if(len(set(train_graph.predecessors(a)))==0 |set(train_graph.predecessors(b)))==0:
            return 0
        sim = (len(set(train_graph.predecessors(a)).intersection(set(train_graph.predecessors(b)))))/(len(set(train_graph.predecessors(a)).union(set(train_graph.predecessors(b)))))
        return sim
    except:
        return 0

#one test case
jaccard_for_followers(273084,470294) #0
#node 1635354 not in graph 
jaccard_for_followers(669354,1635354) #0

# =============================================================================
# Cosine distance
# =============================================================================
#For followees
def cosine_for_followees(a,b):
    try:
        if len(set(train_graph.successors(a)))==0 | len(set(train_graph.successors(b)))==0:
            return 0
        sim = (len(set(train_graph.successors(a)).intersection(set(train_graph.successors(b)))))/(math.sqrt(len(set(train_graph.successors(a)))*len((set(train_graph.successors(b))))))
        return sim
    except:
        return 0

#one test case
cosine_for_followees(273084,1505602) #0
#node 1635354 not in graph 
cosine_for_followees(273084,1635354) #0

#For followers
import math
def cosine_for_followers(a,b):
    try:
        if len(set(train_graph.predecessors(a)))==0 |len(set(train_graph.predecessors(b)))==0:
            return 0
        sim = (len(set(train_graph.predecessors(a)).intersection(set(train_graph.predecessors(b)))))/(math.sqrt(len(set(train_graph.predecessors(a))))*(len(set(train_graph.predecessors(b)))))
        return sim 
    except:
        return 0

#one test case
cosine_for_followers(2,470294) #0.02886751345948129
#node 1635354 not in graph 
cosine_for_followers(669354,1635354) #0

# =============================================================================
# Pagerank
# =============================================================================
import pickle
pgrank = nx.pagerank(train_graph)
pgrank[max(pgrank,key = pgrank.get)] #max value ->2.7098251341935827e-05
pgrank[min(pgrank,key = pgrank.get)] #min value ->1.6556497245737814e-07
sum(pgrank.values())/len(pgrank) #mean value ->5.615699699389075e-07

pickle.dump(pgrank,open('page_rank.p','wb'))
#fro reading
pgrank = pickle.load(open('page_rank.p','rb'))
mean_pr = np.mean(list(pgrank.values())) #mean value ->5.615699699389075e-07
# =============================================================================
# SHortest path
# =============================================================================
#if has direct edge then deleting that edge and calculating shortest path
def compute_shortest_path_lenth(a,b):
    p=-1
    try:
        ''' path between twoo nodes directly connected then we remove that edge and calculate path'''
        if train_graph.has_edge(a,b):
            train_graph.remove_edge(a,b)
            p = nx.shortest_path_length(train_graph,source=a, target=b)
            train_graph.add_edge(a,b)
        else:
            p = nx.shortest_path_length(train_graph,source=a, target=b)
        return p
    except:
        '''For there is no path'''
        return -1
    
#test case
compute_shortest_path_lenth(77697, 826021) #10
compute_shortest_path_lenth(669354,1635354) #-1

# =============================================================================
# Checking for same community
# =============================================================================
#getting weekly connected edges from graph 
wcc = list(nx.weakly_connected_components(train_graph))
def belongs_to_same_wcc(a,b):
    '''
        Input two nodes : a , b .
        Output : Boolean (1 : They belong to same community (Weakly connected components), 0 They do not belong to same Weakly connected components)
    '''    
    index =[]
    if train_graph.has_edge(b,a):
        return 1
    if train_graph.has_edge(a,b):
        for i in wcc:
            if a in i:
                index = i
                break
        if b in index:
            train_graph.remove_edge(a,b)
            if compute_shortest_path_lenth(a,b)== -1:
                train_graph.add_edge(a,b)
                return 0
            else:
                train_graph.add_edge(a,b)
                return 1
        else:
            return 0
    else:
        for i in wcc:
            if a in i:
                index = i
                break
        if b in index:
            return 1
        else:
            return 0
        
#test case             
belongs_to_same_wcc(861, 1659750)  #0            
belongs_to_same_wcc(669354,1635354) #0 

# =============================================================================
# Adamic/Adar Index
# =============================================================================
def cal_adar_index(a,b):
    sum= 0
    try:
        n = list(set(train_graph.successors(a)).intersection(set(train_graph.successors(b))))
        if len(n)!=0:
            for i in n:
                sum+=(1/np.log10(len(list(train_graph.predecessors(i)))))
            return sum
        else:
            return 0
    except:
        return 0
        
#test case             
cal_adar_index(1,189226) #0
cal_adar_index(669354,1635354) #0

# =============================================================================
# Is persion was following back
# =============================================================================
def follows_me_back(a,b):
    if train_graph.has_edge(b,a):
        return 1
    else:
        return 0
            
#test case             
follows_me_back(1,189226) #1->(yes)
follows_me_back(669354,1635354) #0 -> (no)

# =============================================================================
# Katz Centrality(similar to pagerank)
# =============================================================================
katz = nx.katz_centrality(train_graph,alpha=0.005)
katz[max(katz,key = katz.get)] #max value ->0.003394554981699122
katz[min(katz,key = katz.get)] #min value ->0.0007313532484065916
sum(katz.values())/len(katz) #mean value ->0.0007483800935562018

pickle.dump(katz,open('katz.p','wb'))
#for reading
katz = pickle.load(open('katz.p','rb'))
mean_katz = np.mean(list(katz.values())) #mean value ->0.0007483800935562018
# =============================================================================
# Hits Score
# =============================================================================
hits = nx.hits(train_graph,max_iter=100, tol=1.0e-8, nstart=None)
hits[max(hits,key = hits.get)] #max value ->0.003394554981699122
hits[min(hits,key = hits.get)] #min value ->0.0007313532484065916
sum(hits.values())/len(hits) #mean value ->0.0007483800935562018

pickle.dump(hits,open('hits.p','wb'))
#for reading
hits = pickle.load(open('hits.p','rb'))
mean_hits = np.mean(list(hits[0].values())) #mean value ->0.0007483800935562018

# =============================================================================
# FEATURIZATION
# =============================================================================
# =============================================================================
# Reading a sample of Data from both train and test¶
# =============================================================================
import random
#Train data 
df = pd.read_csv('train_after_eda.csv')
skip_train = sorted(random.sample(range(1,len(df)+2),len(df)-100000 ))
df_final_train  = pd.read_csv('train_after_eda.csv', skiprows=skip_train, names=['source_node', 'destination_node'])
df_final_train['indicator_link'] = pd.read_csv('train_y.csv',skiprows=skip_train, names=['indicator_link'])
df_final_train.head(2)
#Test data
df1 = pd.read_csv('test_after_eda.csv')
skip_test = sorted(random.sample(range(1,len(df1)+2),len(df1)-50000  ))
df_final_test = pd.read_csv('test_after_eda.csv', skiprows=skip_test, names=['source_node', 'destination_node'])
df_final_test['indicator_link'] = pd.read_csv('test_y.csv', skiprows=skip_test, names=['indicator_link'])
df_final_test.head(2)


# =============================================================================
# Adding a set of features
# =============================================================================
df_final = pd.concat([df_final_train,df_final_test])
# jaccard_followers
df_final['jaccard_followers'] = df_final.apply(lambda row : jaccard_for_followers(row['source_node'],row['destination_node']),axis =1)
# jaccard_followees
df_final['jaccard_followees'] = df_final.apply(lambda row : jaccard_for_followees(row['source_node'],row['destination_node']),axis =1)
# cosine_followers
df_final['cosine_followers'] = df_final.apply(lambda row : cosine_for_followers(row['source_node'],row['destination_node']),axis =1)
# cosine_followees
df_final['cosine_followees'] = df_final.apply(lambda row : cosine_for_followees(row['source_node'],row['destination_node']),axis =1)

# num_followers_s, num_followees_s,num_followers_d, num_followees_d,inter_followers,inter_followees
def compute_features_stage1(x):
    num_followers_s = []
    num_followees_s = []
    num_followers_d = []
    num_followees_d = []
    inter_followers = []
    inter_followees = []
    for i ,row in x.iterrows():
        try:
            s1 = set(train_graph.predecessors(row['source_node']))
            s2 = set(train_graph.successors(row['source_node']))
        except:
            s1 = set()
            s2 = set()
        
        try:
            d1 = set(train_graph.predecessors(row['destination_node']))
            d2 = set(train_graph.successors(row['destination_node']))
        except:
            d1 = set()
            d2 = set()    
        
        num_followers_s.append(len(s1))
        num_followees_s.append(len(s2))
        
        num_followers_d.append(len(d1))
        num_followees_d.append(len(d2))

        inter_followers.append(len(s1.intersection(d1)))
        inter_followees.append(len(s2.intersection(d2)))
    return num_followers_s, num_followees_s,num_followers_d, num_followees_d,inter_followers,inter_followees

df_final['num_followers_s'],df_final['num_followees_s'],df_final['num_followers_d'], df_final['num_followees_d'],df_final['inter_followers'],df_final['inter_followees'] = compute_features_stage1(df_final)

#adar_index
df_final['adar_index'] = df_final.apply(lambda row : cal_adar_index(row['source_node'],row['destination_node']),axis =1)
#followback 
df_final['follows_back'] = df_final.apply(lambda row : follows_me_back(row['source_node'],row['destination_node']),axis =1)
#same_comp
df_final['same_comp'] = df_final.apply(lambda row : belongs_to_same_wcc(row['source_node'],row['destination_node']),axis =1)
#shortest path
df_final['shortest_path'] = df_final.apply(lambda row : compute_shortest_path_lenth(row['source_node'],row['destination_node']),axis =1)

#saving the whole final data  as hdfs 
hdf = pd.HDFStore('storage_sample_stage.h5')
hdf.put('train_test_df',df_final,format ='table',data_columns=True)
hdf.close()

#for reading 
df_final1 = pd.read_hdf('storage_sample_stage.h5','train_test_df',mode='r')

# =============================================================================
# Weight Features
# =============================================================================
#weight for source and destination of each link
from tqdm import tqdm
Weight_in = {}
Weight_out = {}
for i in tqdm(train_graph.nodes()):
    
    s1 = set(train_graph.predecessors(i))
    w_in = 1/(math.sqrt(1+len(s1)))
    Weight_in[i] = w_in
    
    s2 = set(train_graph.successors(i))
    w_out = 1/(math.sqrt(1+len(s2)))
    Weight_out[i] = w_out
    
#for imputing with mean
mean_weight_in =np.mean(list(Weight_in.values())) #0.5896525798139404
mean_weight_out =np.mean(list(Weight_out.values())) #0.6027837700482165

# weight of incoming edges
df_final['weight_in'] = df_final.destination_node.apply(lambda x : Weight_in.get(x,mean_weight_in))
# weight of outgoing edges
df_final['weight_out'] = df_final.source_node.apply(lambda x : Weight_out.get(x,mean_weight_out))
# weight of incoming edges + weight of outgoing edges
df_final['weight_f1'] = df_final.weight_in + df_final.weight_out
# weight of incoming edges * weight of outgoing edges
df_final['weight_f2'] = df_final.weight_in * df_final.weight_out
# 2*weight of incoming edges + weight of outgoing edges
df_final['weight_f3'] = 2*df_final.weight_in + 1*df_final.weight_out
# weight of incoming edges + 2*weight of outgoing edges
df_final['weight_f4'] = 1*df_final.weight_in + 2*df_final.weight_out

#page rank for source and destination
df_final['page_rank_s'] = df_final.source_node.apply(lambda x : pgrank.get(x,mean_pr))
df_final['page_rank_d'] = df_final.destination_node.apply(lambda x : pgrank.get(x,mean_pr))

#Katz centrality score for source and destination
df_final['katz_s'] = df_final.source_node.apply(lambda x : katz.get(x,mean_katz))
df_final['katz_d'] = df_final.destination_node.apply(lambda x : katz.get(x,mean_katz))

#Hits algorithm score for source and destination
df_final['hubs_s'] = df_final.source_node.apply(lambda x : hits[0].get(x,0))
df_final['hubs_d'] = df_final.destination_node.apply(lambda x : hits[0].get(x,0))

df_final['authorities_s'] = df_final.source_node.apply(lambda x : hits[1].get(x,0))
df_final['authorities_d'] = df_final.destination_node.apply(lambda x : hits[1].get(x,0))

#saving the whole final data  as hdfs 
hdf = pd.HDFStore('storage_sample_stage1.h5')
hdf.put('train_test_df',df_final,format ='table',data_columns=True)
hdf.close()

#for reading 
df_final2 = pd.read_hdf('storage_sample_stage1.h5','train_test_df',mode='r')

# =============================================================================
# Singular Value Decomposition features
# =============================================================================
from scipy.sparse.linalg import svds
#for svd features to get feature vector creating a dict node val and index in svd vector
sadj_col = sorted(train_graph.nodes())
sadj_dict = {val:idx for idx,val in enumerate(sadj_col) }
def svd(x,S):
    try:
        z = sadj_dict[x]
        return S[z]
    except:
        return [0,0,0,0,0,0]

#createing a adjanceny matrix
Adj = nx.adjacency_matrix(train_graph,nodelist = sadj_col).asfptype()
U,s,V =svds(Adj, k=6)

#SVD features for both source and destination
df_final[['svd_u_s_1', 'svd_u_s_2','svd_u_s_3', 'svd_u_s_4', 'svd_u_s_5', 'svd_u_s_6']] = df_final.source_node.apply(lambda x : svd(x,U)).apply(pd.Series)
df_final[['svd_u_d_1', 'svd_u_d_2','svd_u_d_3', 'svd_u_d_4', 'svd_u_d_5', 'svd_u_d_6']] = df_final.destination_node.apply(lambda x : svd(x,U)).apply(pd.Series)

df_final[['svd_v_s_1', 'svd_v_s_2','svd_v_s_3', 'svd_v_s_4', 'svd_v_s_5', 'svd_v_s_6']] = df_final.source_node.apply(lambda x : svd(x,V.T)).apply(pd.Series)
df_final[['svd_v_d_1', 'svd_v_d_2','svd_v_d_3', 'svd_v_d_4', 'svd_v_d_5', 'svd_v_d_6']] = df_final.destination_node.apply(lambda x : svd(x,V.T)).apply(pd.Series)

#saving the whole final data  as hdfs 
hdf = pd.HDFStore('storage_sample_stage2.h5')
hdf.put('train_test_df',df_final,format ='table',data_columns=True)
hdf.close()

#for reading 
df_final3 = pd.read_hdf('storage_sample_stage2.h5','train_test_df',mode='r')

# =============================================================================
# Svd_dot
# =============================================================================
# svd_dot -> Dot product between sourse node svd and destination node svd features
    df_final['svd_u_s_d'] = df_final['svd_u_s_1']*df_final['svd_u_d_1'] \
    + df_final['svd_u_s_2']*df_final['svd_u_d_2'] \
    + df_final['svd_u_s_3']*df_final['svd_u_d_3'] \
    + df_final['svd_u_s_4']*df_final['svd_u_d_4'] \
    + df_final['svd_u_s_5']*df_final['svd_u_d_5'] \
    + df_final['svd_u_s_6']*df_final['svd_u_d_6']
    
    df_final['svd_v_s_d'] = df_final['svd_v_s_1']*df_final['svd_v_d_1'] \
    + df_final['svd_v_s_2']*df_final['svd_v_d_2'] \
    + df_final['svd_v_s_3']*df_final['svd_v_d_3'] \
    + df_final['svd_v_s_4']*df_final['svd_v_d_4'] \
    + df_final['svd_v_s_5']*df_final['svd_v_d_5'] \
    + df_final['svd_v_s_6']*df_final['svd_v_d_6']
# =============================================================================

##alternative of svd_dot 
source_u=df_final[['svd_u_s_1','svd_u_s_2','svd_u_s_3','svd_u_s_4','svd_u_s_5','svd_u_s_6']]
source_v=df_final[['svd_v_s_1','svd_v_s_2','svd_v_s_3','svd_v_s_4','svd_v_s_5','svd_v_s_6']]
Distination_u=df_final[['svd_u_d_1','svd_u_d_2','svd_u_d_3','svd_u_d_4','svd_u_d_5','svd_u_d_6']]
Distination_v=df_final[['svd_v_d_1','svd_v_d_2','svd_v_d_3','svd_v_d_4','svd_v_d_5','svd_v_d_6']]

from tqdm import tqdm
svd_dot_u=[]
svd_dot_v=[]
for i  in tqdm(range(len(source_u))):
    svd_dot_u.append(np.dot(source_u.values[i],Distination_u.values[i]))
    svd_dot_v.append(np.dot(source_v.values[i],Distination_v.values[i]))
df_final['svd_dot_u']=svd_dot_u
df_final['svd_dot_v']=svd_dot_v 

# =============================================================================
 #Preferential attachment
# =============================================================================
#for followees
def preferential_for_followees(a,b):
    try:
        if len(set(train_graph.successors(a))) == 0  | len(set(train_graph.successors(b))) == 0:
            return 0
        sim = (len(set(train_graph.successors(a)))*len((set(train_graph.successors(b)))))
        return sim
    except:
        return 0
#for followees
def preferential_for_followers(a,b):
    try:
        if len(set(train_graph.predecessors(a))) == 0  | len(set(train_graph.predecessors(b))) == 0:
            return 0
        sim = (len(set(train_graph.predecessors(a)))*len((set(train_graph.predecessors(b)))))
        return sim
    except:
        return 0
    
df_final['preferential_for_followees'] = df_final.apply(lambda row : preferential_for_followees(row['source_node'],row['destination_node']),axis =1)
df_final['preferential_for_followers'] = df_final.apply(lambda row : preferential_for_followers(row['source_node'],row['destination_node']),axis =1)

#saving the whole final data  as hdfs 
hdf = pd.HDFStore('storage_sample_stage3.h5')
hdf.put('train_test_df',df_final,format ='table',data_columns=True)
hdf.close()

#for reading 
df_final = pd.read_hdf('storage_sample_stage3.h5','train_test_df',mode='r')

#now we split it back to train and test data 
df_final_train = df_final[:100002]
df_final_test = df_final[100002:150004]

# =============================================================================
# MODEL BUILDING
# =============================================================================
y_train = df_final_train.indicator_link
y_test = df_final_test.indicator_link

df_final_train.drop(['source_node', 'destination_node','indicator_link'],axis=1,inplace=True)
df_final_test.drop(['source_node', 'destination_node','indicator_link'],axis=1,inplace=True)

# =============================================================================
# Random forest
# =============================================================================
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import f1_score
estimators = [10,50,100,250,450]
train_scores = []
test_scores = []
for i in estimators:
    clf = RandomForestClassifier(n_estimators=i,max_depth=5,min_samples_leaf=52, min_samples_split=120,n_jobs=-1,random_state=25)
    clf.fit(df_final_train,y_train)
    train_sc = f1_score(y_train,clf.predict(df_final_train))
    test_sc = f1_score(y_test,clf.predict(df_final_test))
    train_scores.append(train_sc)
    test_scores.append(test_sc)
    print('Estimators:',i,'Train Score:',train_sc,'Test Score:',test_sc)
    '''
    Estimators: 10 Train Score: 0.9008430433236407 Test Score: 0.8689215060625399
    Estimators: 50 Train Score: 0.9202715404699738 Test Score: 0.9155498477535885
    Estimators: 100 Train Score: 0.9203284949546597 Test Score: 0.915424739195231
    Estimators: 250 Train Score: 0.9211152206028729 Test Score: 0.916995390939667
    Estimators: 450 Train Score: 0.9213544282605056 Test Score: 0.9172265074230738
    '''
plt.plot(estimators,train_scores,label='Train Score')
plt.plot(estimators,test_scores,label='Test Score')
plt.xlabel('Estimators')
plt.ylabel('Score')
plt.title('Estimators vs score at depth of 5')    
# =============================================================================
depths = [3,9,11,15,20,35,50,70,130]
train_scores = []
test_scores = []
for i in depths:
    clf = RandomForestClassifier(n_estimators=115,max_depth=i,min_samples_leaf=52, min_samples_split=120,n_jobs=-1,random_state=25)
    clf.fit(df_final_train,y_train)
    train_sc = f1_score(y_train,clf.predict(df_final_train))
    test_sc = f1_score(y_test,clf.predict(df_final_test))
    train_scores.append(train_sc)
    test_scores.append(test_sc)
    print('Depth:',i,'Train Score:',train_sc,'Test Score:',test_sc)
    '''
    Depth: 3 Train Score: 0.8725173308382597 Test Score: 0.8533196528579368
    Depth: 9 Train Score: 0.9567589102387299 Test Score: 0.92336837722494
    Depth: 11 Train Score: 0.9599861529689663 Test Score: 0.9239856082191211
    Depth: 15 Train Score: 0.9646333021134502 Test Score: 0.9255933156918718
    Depth: 20 Train Score: 0.9651345940837543 Test Score: 0.9231154311459681
    Depth: 35 Train Score: 0.9653175079008607 Test Score: 0.9259544367003018
    Depth: 50 Train Score: 0.9653175079008607 Test Score: 0.9259544367003018
    Depth: 70 Train Score: 0.9653175079008607 Test Score: 0.9259544367003018
    Depth: 130 Train Score: 0.9653175079008607 Test Score: 0.9259544367003018
    '''
plt.plot(depths,train_scores,label='Train Score')
plt.plot(depths,test_scores,label='Test Score')
plt.xlabel('Depth')
plt.ylabel('Score')
plt.title('Depth vs score at depth of 5')    

# =============================================================================
# Random search CV
# =============================================================================
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
param_dist = {'n_estimators':sp_randint(105,125) ,
              'max_depth':sp_randint(10,15),
              'min_samples_leaf':sp_randint(25,65),
              'min_samples_split':sp_randint(25,65)}
clf = RandomForestClassifier(random_state=25,n_jobs= -1)
rf_random = RandomizedSearchCV(clf,param_distributions = param_dist,n_iter =5,cv=10,scoring='f1',random_state=25,return_train_score=True)
rf_random.fit(df_final_train,y_train)
rf_random.cv_results_['mean_test_score']
rf_random.cv_results_['mean_train_score']
rf_random.best_estimator_

clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=13, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=26, min_samples_split=47,
            min_weight_fraction_leaf=0.0, n_estimators=121, n_jobs=-1,
            oob_score=False, random_state=25, verbose=0, warm_start=False)
clf.fit(df_final_train,y_train)
train_sc = f1_score(y_train,clf.predict(df_final_train))
test_sc = f1_score(y_test,clf.predict(df_final_test))

#Making the confusion matrix and report for train data 
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_train,clf.predict(df_final_train)))
confusion_matrix(y_train,clf.predict(df_final_train))

#Making the confusion matrix and report for test data 
print(classification_report(y_test,clf.predict(df_final_test)))
confusion_matrix(y_test,clf.predict(df_final_test))

#ROC curve
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,clf.predict(df_final_test)) #0.9300684644249815

#Plotting the ROC curve
from sklearn.metrics import roc_curve
fpr,tpr,thresolds = roc_curve(y_test,clf.predict(df_final_test))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC')
plt.plot(fpr, tpr) #good graph

features = df_final_train.columns
importances = clf.feature_importances_
indices = (np.argsort(importances))[-25:]
plt.figure(figsize=(10,12))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='r', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')

# =============================================================================
# XGBoost
# =============================================================================
import xgboost as xgb
param_dist = {"n_estimators":sp_randint(105,125),"max_depth": sp_randint(10,15)}
clf = xgb.XGBClassifier()
xgbmodel = RandomizedSearchCV(clf, param_distributions=param_dist,n_jobs=-1, n_iter=5,cv=4,scoring='f1',random_state=25,return_train_score=True)
xgbmodel.fit(df_final_train,y_train)
xgbmodel.cv_results_['mean_test_score']
xgbmodel.cv_results_['mean_train_score']
xgbmodel.best_estimator_

clf = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.1,
       max_delta_step=0, max_depth=14, min_child_weight=1, missing=None,
       n_estimators=123, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
       subsample=1, verbosity=1)
clf.fit(df_final_train,y_train)
train_sc = f1_score(y_train,clf.predict(df_final_train))
test_sc = f1_score(y_test,clf.predict(df_final_test))

#Making the confusion matrix and report for train data 
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_train,clf.predict(df_final_train)))
confusion_matrix(y_train,clf.predict(df_final_train))

#Making the confusion matrix and report for test data 
print(classification_report(y_test,clf.predict(df_final_test)))
confusion_matrix(y_test,clf.predict(df_final_test))

#ROC curve
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,clf.predict(df_final_test)) #0.9313988856090472 -> little better than the previous

#Plotting the ROC curve
from sklearn.metrics import roc_curve
fpr,tpr,thresolds = roc_curve(y_test,clf.predict(df_final_test))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC')
plt.plot(fpr, tpr) #good graph

features = df_final_train.columns
importances = clf.feature_importances_
indices = (np.argsort(importances))[-25:]
plt.figure(figsize=(10,12))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='r', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')

#here we found out The Most Important featurres are Follows back, cosine follwers , jaccard distance,weight features, shortest path.......
# =============================================================================
param_dist ={'max_depth':[1,2,3,4,5],
            'n_estimators' : [100, 300, 500, 700, 900, 1000,1100],
            'learning_rate' : [0.0001, 0.001, 0.01, 0.1,1,10],
            'colsample_bytree' : [0.1, 0.3, 0.5, 0.7 , 0.9, 1]}
clf = xgb.XGBClassifier(random_state=25,n_jobs=-1)
xg_random = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=5,cv=10,scoring='f1',random_state=25)
xg_random.fit(df_final_train,y_train)
xg_random.cv_results_['mean_test_score']
xg_random.cv_results_['mean_train_score']
xg_random.best_estimator_

clf = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=0.1, gamma=0, learning_rate=1,
       max_delta_step=0, max_depth=1, min_child_weight=1, missing=None,
       n_estimators=700, n_jobs=-1, nthread=None,
       objective='binary:logistic', random_state=25, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
       subsample=1, verbosity=1)
clf.fit(df_final_train,y_train)
train_sc = f1_score(y_train,clf.predict(df_final_train))
test_sc = f1_score(y_test,clf.predict(df_final_test))

#Making the confusion matrix and report for train data 
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_train,clf.predict(df_final_train)))
confusion_matrix(y_train,clf.predict(df_final_train))

#Making the confusion matrix and report for test data 
print(classification_report(y_test,clf.predict(df_final_test)))
confusion_matrix(y_test,clf.predict(df_final_test))

#ROC curve
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,clf.predict(df_final_test)) #0.9313988856090472 -> little better than the previous

#Plotting the ROC curve
from sklearn.metrics import roc_curve
fpr,tpr,thresolds = roc_curve(y_test,clf.predict(df_final_test))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC')
plt.plot(fpr, tpr) #good graph

features = df_final_train.columns
importances = clf.feature_importances_
indices = (np.argsort(importances))[-25:]
plt.figure(figsize=(10,12))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='r', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
# =============================================================================
from prettytable import PrettyTable
x = PrettyTable()
x.field_names = ["Model","Train f1_score","test f1_score"]

x.add_row(["RandomForest Classifier",96.53,92.67])
x.add_row(["XGBClassifier",99.98,92.74])
print(x)

# =============================================================================
# 1.The data contains a snapshot of social link prediction with only two columns source node and destination node.
# 
# 2.Our objective is to convert the data into a Classification problem and predict if a link between two users exists.
# 
# 3.ter EDA we split the data using Random Split of 80-20.
# 
# 4.we create various Graph based and Non- Graph Based features from the data such as Jaccard Distance, Cosine Distance, Page Ranking, Shortest path, Adar Index, etc.
# 
# 5.Preferrential feature is added to both train and test data for followers and followees.
# 
# 6.built two models i.e Random forest and XGBClassifier.
# 
# 7.if we compare Random Forest performed better than XGBClasssifier.
# =============================================================================










































