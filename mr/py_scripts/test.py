#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt #, pandas as pd
import gc
from scipy.stats import spearmanr
from scipy import sparse
from memory_profiler import profile
from datetime import datetime
import gc
import sys
import json
import operator



import sys



import scipy.stats as ss

def get_mappings(n_nodes, n_pairs):
    """ mappings for nodes to pairs """

    #S = np.zeros((n_nodes, n_pairs), dtype= int)
    S=sparse.dok_matrix((n_nodes,n_pairs))
    k = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            S[i, k] = 1
            S[j, k] = 1
            k += 1


    triu_ix = np.triu_indices(n_nodes,1)
    return sparse.csr_matrix(S), triu_ix

def build_arrays(file_name):
    with open(file_name) as f:
        users_id=[]
        movies_id=[]
        u=0
        for line in f:
            user, movie= line.split(" ")[0], line.split(" ")[1]
            user=int(user)
            movie=int(movie)
            if len(users_id)!=0 :
                #print("len(users_id)= ",len(users_id)," len(movies_id)= ",len(movies_id))
                if users_id[u-1] == user :
                    sub_movies_list.append(movie)
                else:
                    users_id.append(user)
                    movies_id.append(sub_movies_list)
                    sub_movies_list=[]
                    sub_movies_list.append(movie)
                    u+=1

            else:
                users_id.append(user)
                sub_movies_list=[]
                sub_movies_list.append(movie)
                u+=1
        movies_id.append(sub_movies_list)

        print("list_users= ",len(users_id)," list_movies= ",len(movies_id))
        #print("list_users= ",users_id," list_movies= ",movies_id)

        return users_id, movies_id

def jaccard_simu(list_users, list_movies):
    z=np.zeros((len(list_users),len(list_users)))

    for i in range(len(list_movies)):

        for j in range(i, len(list_movies)):
            intesection= len(set(list_movies[i]).intersection(set(list_movies[j])))
            union= len(set(list_movies[i]).union(set(list_movies[j])))
            score= intesection / float(union)
            z[i,j]=score
            z[j,i]=score
    return z




def objective_function(w, z, S, mu, la):
    d = S.dot(w)

    if np.any(d < 0):
        return np.inf

    return (mu / 2) * (w.dot(z) - np.log(d).sum() + la * (mu / 2) * w.dot(w))


@profile
def graph_discovery(list_user, list_movies, mu=1, la=1):
    n_nodes=len(list_user)
    n_pairs = n_nodes * (n_nodes - 1) // 2
    stop_thresh = 10e-2 / n_pairs
    print("n_nodes= ",n_nodes, " n_pairs= ",n_pairs)
    ss, triu_ix= get_mappings(n_nodes, n_pairs)
    # classifier distances
    z = 1-jaccard_simu(list_user, list_movies)
    zz = 1 - z
    #np.savetxt('matrix_jaccard.'+str(len(list_user))+'txt', zz)

    z = z[triu_ix]

    # init similarity vector (upper triangular matrix)
    #w = np.ones(n_pairs)

    w=np.random.randint(2, size=n_pairs)
    # degree vector

    d = ss.dot(w)
    print('ddd= ', d)

    # init step

    mult= (ss.T.dot(ss))
    mult=(sparse.linalg.norm(mult))
    print('mult= ', mult)

    gamma = 1 / ((mu / 2) * (
              np.linalg.norm((z)) + mult + 2 * la * (mu / 2)))

    #gamma = 0.5

    obj = objective_function(w, z, ss, mu, la)

    for k in range(2000):
        #print('(1. / d).dot(ss) ', ss.T.dot((1. / d)))
        grad = (mu / 2) * (z - ss.T.dot((1. / d)) + 2 * la * (mu / 2) * w)

        new_w = w - gamma * grad
        new_w[new_w < 0]  = 0

        new_obj = objective_function(new_w, z, ss, mu, la)

        if new_obj > obj:
            gamma /= 2

        elif abs(obj - new_obj) > abs(stop_thresh * obj):
            obj = new_obj
            w = new_w.copy()
            gamma *= 1.05

        else:
            obj = new_obj
            w = new_w.copy()
            break

        d = ss.dot(w)

    print("it=", k, "obj=", obj, "gamma=", gamma)
    del(ss, d, obj, new_obj)
    gc.collect()
    similarities = np.zeros((n_nodes, n_nodes))
    print("")
    similarities[triu_ix] = similarities.T[triu_ix] = w

    return similarities,zz


def produitMatriciel(A, B):
    n = len(A); p = len(A[0]);  q = len(B[0])
    return [[sum([A[i][k]*B[k][j] for k in range(p)]) for j in range(q)] for i in range(n)]

def non_symetric_graph_discovery(losses, list_user, list_movies, mu=1, la=1):
    n_nodes = len(list_user)
    n_pairs = n_nodes * (n_nodes - 1) // 2
    stop_thresh = 10e-2 / n_pairs
    print("n_nodes= ", n_nodes, " n_pairs= ", n_pairs)
    S, triu_ix,mx = get_mappings(n_nodes, n_pairs)
    print("S_dim= ", S.size)
    # classifier distances
    z = 1 - jaccard_simu(list_user, list_movies)
    zz=[]
    for i in range(len(z)) :
        for j in range(len(z)) :
            if j!=i :
                zz.append(z[i,j])

        # init similarity vector (upper triangular matrix)
    #w = np.ones(n_pairs*2)

    w=np.random.randint(2, size=n_pairs)

    S=np.concatenate((S,S), axis=1)
    #print("S= ",S.size,"\n",S)
    # degree vector

    d = S.dot(w)

    mult =  S.T.dot(S)

    gamma = 1 / (np.linalg.norm(losses.dot(S)) + (mu / 2) * (
            np.linalg.norm((zz)) + np.linalg.norm(mult) + 2 * la * (mu / 2)))
    # gamma = 0.5

    obj = objective_function(w, zz, S, losses, mu, la)
    cost_values=[]
    for k in range(2000):

        grad = losses.dot(S)+ (mu / 2) * (zz - (1. / d).dot(S) + 2 * la * (mu / 2) * w)

        new_w = w - gamma * grad
        new_w[new_w < 0] = 0

        new_obj = objective_function(new_w, zz, S, losses, mu, la)

        if new_obj > obj:
            gamma /= 2

        elif abs(obj - new_obj) > abs(stop_thresh * obj):
            obj = new_obj
            w = new_w.copy()
            gamma *= 1.05

        else:
            obj = new_obj
            w = new_w.copy()
            break

        d = S.dot(w)
        # print("it=", k, "obj=", obj, "w= ", w, "gamma=", gamma)
        cost_values.append(objective_function(w, zz, S, losses, mu, la))

    print("it=", k, "obj=", obj, "gamma=", gamma)
    del (S, d, obj)
    gc.collect()
    print("")
    similarity=np.zeros((n_nodes,n_nodes))
    c=0
    # 2we reconstitue the full w matrix
    for i in range(n_nodes) :
        for j in range(n_nodes) :
            if i!=j :
                similarity[i,j]=w[c]
                c+=1
    return similarity, (1-z), cost_values




def block_graph_discovery(losses, list_user, list_movies, mu=1, la=1, kappa=1, max_iter=1e4):
    n_nodes= len(list_user)
    n_pairs = n_nodes * (n_nodes - 1) // 2

    S, triu_ix, map_idx = get_mappings(n_nodes, n_pairs)

    # classifier distances
    z = 1-jaccard_simu(list_user, list_movies)
    print('Jaccard simularties: \n',z)
    z = z[triu_ix]

    # init similarity vector (upper triangular matrix)
    w = np.ones(n_pairs)

    # degree vector
    d = S.dot(w)

    # init step
    gamma = 0.5

    obj = objective_function(w, z, S, losses, mu, la)

    new_w = w.copy()

    for k in range(int(max_iter)):

        # pick indices of weights to update
        rnd_j = np.random.choice(n_nodes, 1 + kappa, replace=False)
        i, others = rnd_j[0], rnd_j[1:]

        # select corresponding blocks of vectors
        idx_block = map_idx[np.minimum(i, others), np.maximum(i, others)]
        d_block = S[rnd_j, :].dot(new_w)
        S_block = S[rnd_j, :][:, idx_block]

        grad = losses[rnd_j].dot(S_block) + (mu / 2) * (
                    z[idx_block] - (1. / d_block).dot(S_block) + 2 * la * (mu / 2) * new_w[idx_block])

        new_w[idx_block] = new_w[idx_block] - gamma * grad
        new_w[new_w < 0] = 0

        new_obj = objective_function(new_w, z, S, losses, mu, la)

        # update step only every n_nodes iterationss, for stability
        if k % n_nodes == 0:

            if new_obj > obj or not np.isfinite(new_obj):
                gamma /= 2
                new_w = w.copy()
                new_obj = obj

            elif obj - new_obj < abs(obj) / 1e6:
                obj = new_obj
                break

            else:
                gamma *= 1.05
                w = new_w.copy()
                obj = new_obj

    print("itF=", k, "objF=", obj, "gammaF=", gamma)
    similarities = np.zeros((n_nodes, n_nodes))
    similarities[triu_ix] = similarities.T[triu_ix] = w
    np.savetxt('matrix_simu_'+str(size)+".txt", similarities)

    return similarities



def non_symetric_block_graph_discovery(losses, list_user, list_movies, mu=1, la=1, kappa=1, max_iter=1e4):
    n_nodes= len(list_user)
    n_pairs = n_nodes * (n_nodes - 1) // 2

    S, triu_ix, map_idx = get_mappings(n_nodes, n_pairs)

    # classifier distances
    z = 1 - jaccard_simu(list_user, list_movies)
    zz = []
    for i in range(len(z)):
        for j in range(len(z)):
            if j != i:
                zz.append(z[i, j])

    # init similarity vector (upper triangular matrix)
    w = np.ones(n_pairs * 2)
    S = np.concatenate((S, S), axis=1)

    # degree vector
    d = S.dot(w)

    # init step
    gamma = 0.5

    obj = objective_function(w, zz, S, losses, mu, la)

    new_w = w.copy()

    for k in range(int(max_iter)):

        # pick indices of weights to update
        rnd_j = np.random.choice(n_nodes, 1 + kappa, replace=False)
        i, others = rnd_j[0], rnd_j[1:]

        # select corresponding blocks of vectors
        idx_block = map_idx[np.minimum(i, others), np.maximum(i, others)]
        d_block = S[rnd_j, :].dot(new_w)
        S_block = S[rnd_j, :][:, idx_block]
        print("idx_block= \n",idx_block)
        grad = losses[rnd_j].dot(S_block) + (mu / 2) * (
                    zz[idx_block] - (1. / d_block).dot(S_block) + 2 * la * (mu / 2) * new_w[idx_block])

        new_w[idx_block] = new_w[idx_block] - gamma * grad
        new_w[new_w < 0] = 0

        new_obj = objective_function(new_w, zz, S, losses, mu, la)

        # update step only every n_nodes iterationss, for stability
        if k % n_nodes == 0:

            if new_obj > obj or not np.isfinite(new_obj):
                gamma /= 2
                new_w = w.copy()
                new_obj = obj

            elif obj - new_obj < abs(obj) / 1e6:
                obj = new_obj
                break

            else:
                gamma *= 1.05
                w = new_w.copy()
                obj = new_obj

    print("itF=", k, "objF=", obj, "gammaF=", gamma)
    similarity = np.zeros((n_nodes, n_nodes))
    c = 0
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                similarity[i, j] = w[c]
                c += 1
    return similarity


def spearman_without_null(simu_matr,jaccard_mat,la,size):
    coef_array = np.zeros(size)
    for i in range(len(simu_matr)):
        sj_index=[]
        #in this loop we check null values and we save theirs indices
        for j in range(len(simu_matr[i])) :
            if simu_matr[i][j]==0 or jaccard_mat[i][j]==0 :
                sj_index.append(j)
        # Here we delete values that matches the indices already gathred
        new_simu=np.delete(simu_matr[i],sj_index)
        new_jacc=np.delete(jaccard_mat[i],sj_index)
       # print('second size S= ', len(new_simu),' J= ', len(new_jacc))

        coef, p = spearmanr(new_simu,new_jacc)
        coef_array[i] = coef
    # we rank the correlation values
    coef_array.sort()
    return coef_array

def bruteForce(jacc_matrix,k) :
    knnG=[]
    for i in range(len(jacc_matrix)):
        knn=sorted(enumerate(jacc_matrix[i]), key=operator.itemgetter(1))
        if k < len(knn) :
            knnG.append(knn[len(knn)-k-1:len(knn)-1])
        elif k==len(knn):
            knnG.append(knn[len(knn)-k:])
        else:
            print("verify your inputs")

        #print("knn_user: ",i+1,": ",knn)
        # for k_ in range(k):
        #     print(str((knn[len(knn) - k_-1][0])+1),":", knn[len(knn) - k_-1][1])
    a=np.ones(len(jacc_matrix))
    mat_bf=np.diag(a)
    for i in range(len(knnG)):
        for j in range(len(knnG[i])):
            mat_bf[i,knnG[i][j][0]]=knnG[i][j][1]

    #print("final matric\n",mat_bf)
    return mat_bf


def discovery_build_graph(simu_matrix):
    knn_graph = {}
    k=len(simu_matrix)
    for i in range(len(simu_matrix)):
        knn_graph[str(i + 1)] = []
        sorted_list = sorted(enumerate(simu_matrix[i]), key=operator.itemgetter(1))
        #print("sorted_list=\n", sorted_list)
        for m in range(0, k):
            if sorted_list[len(sorted_list) - m-1][1] != 0 :
                knn_graph[str(i + 1)].append({
                    str((sorted_list[len(sorted_list) - m-1][0])+1): sorted_list[len(sorted_list) - m-1][1]
                })


    path_parent = "C:/Users/Moham/git/mahout/mr/Datasets/"

    with open(path_parent+'KNNG_Brute_force.txt', 'w+') as outfile:
        json.dump(knn_graph, outfile,indent=2)



if __name__== "__main__":
    start_time= datetime.now()
    print(" Start program............ ",start_time)
    size= 500
    lambdaa= 0.35

    users_array, movies_array=build_arrays(sys.argv[0])
    #users_array, movies_array=build_arrays("../Datasets/TestSet0.data")
    #losses = np.zeros(len(users_array))  # local losses. it can be set to 0

    #simu,z = graph_discovery(users_array[:size], movies_array[:size], mu=1,la=lambdaa)
    #simu,jacc,cost = non_symetric_graph_discovery(losses, users_array, movies_array, mu=1,la=lambdaa)

    z= bruteForce(jaccard_simu(users_array, movies_array), 175)
    discovery_build_graph(z)
    #np.savetxt('C:/Users/Moham/git/KNNG_FULL_DECENT.txt', simu)

    #print(simu)
    end_time= datetime.now()
    print(" program finished............ ", end_time)
    print('Duration: {}'.format(end_time - start_time))

    #print("simu = \n",simu)
    #print("Jacc = \n",jacc)


    # z=jaccard_simu(users_array[:size], movies_array[:size])
    # #print("jaccard: \n",z)
    # n_nodes = len(users_array)
    # n_pairs = n_nodes * (n_nodes - 1) // 2
    # print("n_nodes= ", n_nodes, " n_pairs= ", n_pairs)
    # S, triu_ix = get_mappings(n_nodes, n_pairs)
    # print("triu_ix\n",triu_ix)
    # z=z[triu_ix]
    # print("z[triu_ix]",z)
    #


    #print("similarity_matrix : \n",similarities)

    # here we draw the distrubution of correlation values,
    # lambda_array=[0.1,0.5,1,10,50,100]
    # for i in lambda_array :
    #     similarities, z = graph_discovery(losses, users_array[:size], movies_array[:size], mu=1, la=i)
    #     spea_values= spearman_without_null(similarities,z,i,size)
    #     plt.scatter(spea_values, np.arange(1, len(spea_values) + 1))
    #
    #     # we save the diagram in a file
    # plt.title('Spearman correlation W_(1-J) ' + str(size) + ' users' + ' lambda = ' )
    # plt.ylabel('orders of spearman frequences')
    # plt.xlabel('Spearman correlations values')
    # plt.savefig('spearman_W_(1-J)_la11_n0_' + str(size))

    #spearman_calc(similarities,z,1)
    #lambda_array=lambda_calc(np.linspace(0.1,1,10),losses,movies_array)
    #print( 'Full gradient_decent W matrix similarities \n', similarities)
    #np.savetxt('matrix_lambda_'+str(size)+".txt", lambda_array)
    #np.savetxt('matrix_simu_'+str(size)+".txt", similarities)

    ###################### Spearman ########################
    #spearmanr(similarities,z)
    ########################################################
