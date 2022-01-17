import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
########## we use 50 cores to run the algorithm ##########
nbinl=6 # number of \ell bin
number_bin=5 # number of redshift bin
gg1_zp_mat=np.load('input_powerspectrum.npy') #gg1_zp_mat is the powerspectrum between photo-z bins with the shape (nbinl,number_bin,number_bin)
######################## Algorithm 1-FP #######################
def update_p(p0,CR_l,Ql,CP_l):
    for i in range(nbinl):
        CR_l[i]=np.dot(np.dot(p0.T.I,CP_l[i]),p0.I)
        Ql[i]=np.dot(CP_l[i],p0.I)
        Ql_all=np.mat(Ql.sum(axis=0)) 
        CR_l_all=np.mat(CR_l.sum(axis=0))
        p1_T=np.abs(np.dot(Ql_all,CR_l_all.I))
        vj=np.zeros(number_bin) 
        for j in range(number_bin):
            vj[j]=p1_T[j,:].sum()
        epsilon=1e-10
        p2_T=np.zeros_like(p0)
        for j in range(number_bin):
            p2_T[:,j]=(p1_T[:,j]+epsilon)/(vj[j]+epsilon) 
        p0=p2_T.T
    return p0,CR_l
def a1_iter(seed):
    np.random.seed(seed)
    times=0
    J_all=np.zeros(n_times)
    p0_all=np.zeros((n_times,number_bin,number_bin))

    CP_l=np.zeros((nbinl,number_bin,number_bin))
    CP_l[:]=gg1_zp_mat[:] # input power spectrum
    while(times<n_times):# judge_P>0):
        niter=0
        CR_l=np.zeros_like(CP_l)
        Ql=np.zeros_like(CP_l)
        p_random=np.random.rand(number_bin,number_bin)
        for i in range(number_bin):
            p_random[i,i]=2*p_random[i,:].sum()
            p_random[:,i]=p_random[:,i]/p_random[:,i].sum()
        p0=np.mat(p_random)
        JJ=[]
        pp0=[]
        while(niter<n_iter):
            p0,CR_l=update_p(p0,CR_l,Ql,CP_l)
            J=0
            diag_CR=np.zeros(number_bin)
            for i in range(nbinl):
                diag_CR[:]=np.diagonal(CR_l[i])
                CR_l[i]=0
                for j in range(number_bin):
                    CR_l[i,j,j]=abs(diag_CR[j])
                J+=0.5*np.linalg.norm(CP_l[i]-np.dot(np.dot(p0.T,CR_l[i]),p0))**2
            JJ.append(J)
            pp0.append(p0)
            if niter>=3:
                judge_J=(JJ[niter-1]+JJ[niter-2]+JJ[niter-3])/3*1.5
                if JJ[niter]>=judge_J:
                    break
            niter+=1
        J_all[times]=JJ[np.argmin(JJ)]
        p0_all[times]=pp0[np.argmin(JJ)]
        times+=1
    return J_all,p0_all
sum_times=1000 # number of runs
n_a1=10 # 10 threads, each use 5 cores
n_times=int(sum_times/n_a1) # number of runs per thread
n_iter=int(1e4) # number of iterations per run
pool = Pool(n_a1)
rel = pool.map(a1_iter, range(n_a1))
pool.close()
pool.join()
rel1=[]
rel2=[]
for i in range(n_a1):
    rel1.append(rel[i][0])
    rel2.append(rel[i][1])
J_all=np.array(rel1).reshape(-1,)
p0_all=np.array(rel2).reshape(-1,number_bin,number_bin)

J_all_sort=np.copy(J_all[J_all.argsort()]) #output J (Algorithm 1)
p0_all_sort=np.copy(p0_all[J_all.argsort()]) # output the scattering matrix (Algorithm 1) and input to the Algorithm 2

np.save('J_all_a1.npy',J_all_sort)
np.save('p0_all_a1.npy',p0_all_sort)

######################## Algorithm 2-NMF #######################
def update_CR_l(W,Vl,nbinl,number_bin):
    U=np.zeros((number_bin*number_bin,number_bin))
    Vec_Vl=np.zeros((nbinl,number_bin*number_bin,1))
    CL=np.zeros((nbinl,number_bin,number_bin))
    for i in range(number_bin):
        U[:,i]=np.kron(W[:,i],W[:,i])
    for i in range(nbinl):
        Vec_Vl[i]=Vl[i,:,:].reshape((-1,1),order='F')
    for i in range(nbinl):
        for j in range(number_bin):
            CL[i,j,j]=abs(np.dot(np.mat(U).I,Vec_Vl[i])[j])
    return CL

def update_W(W,H,Vl,nbinl,number_bin):
    delta1=np.zeros((number_bin,number_bin))
    delta2=np.zeros((number_bin,number_bin))
    A=np.zeros((number_bin,number_bin))
    B=np.zeros((number_bin,number_bin))
    W1=np.zeros_like(W)
    for n in range(nbinl):
        delta1[:]+=np.dot(Vl[n],H[n].T)
        delta2[:]+=np.dot(np.dot(W,H[n]),H[n].T)
    for i in range(number_bin):
        for j in range(number_bin):
            for b in range(number_bin):
                A[i,j]+=W[i,b]/delta2[i,b]
                B[i,j]+=W[i,b]*delta1[i,b]/delta2[i,b]
    for i in range(number_bin):
        for j in range(number_bin):
            W1[i,j]=W[i,j]*(delta1[i,j]*A[i,j]+1-B[i,j])/(delta2[i,j]*A[i,j])
            if W1[i,j]<0:
                W1[i,j]=W[i,j]*(delta1[i,j]*A[i,j]+1)/(delta2[i,j]*A[i,j]+B[i,j])
    return W1

def a2_iter(nl):
    P_all_test1=[]
    CR_all_test1=[]
    min_J_all=[]
    for ip0 in range(n_a2):
        ip=nl*n_a2+ip0

        p0=np.array(p0_all_sort[ip])
        H=np.zeros((nbinl,number_bin,number_bin))
        Vl=np.zeros((nbinl,number_bin,number_bin))

        Vl[:]=gg1_zp_mat[:] # input power spectrum

        J_all=[]
        p0_all=[]
        CR_all=[]
        W=p0.T
        judge_W=1
        niter=0
        CL=update_CR_l(W,Vl,nbinl,number_bin)
        H[:]=np.dot(CL[:],p0)
        while(niter<n_iter and judge_W>0):
            W1=update_W(W,H,Vl,nbinl,number_bin)
            H[:]=np.dot(CL[:],W1.T)
            CL=update_CR_l(W1,Vl,nbinl,number_bin)
            judge_W=0
            for i in range(number_bin):
                for j in range(number_bin):
                    if(abs(W1[i,j]-W[i,j])>=1e-8):
                        judge_W+=1
            J=0
            for i in range(nbinl):
                J+=0.5*np.linalg.norm(Vl[i]-np.dot(np.dot(W1,CL[i]),W1.T))**2
            J_all.append(J)
            p0_all.append(W1.T)
            CR_all.append(CL)
            W=W1
            if niter>=3:
                judge_J=(J_all[niter-1]+J_all[niter-2]+J_all[niter-3])/3*1.5
                if J_all[niter]>=judge_J:
                    break
            niter+=1
        P_all_test1.append(p0_all[np.argmin(J_all)])
        CR_all_test1.append(CR_all[np.argmin(J_all)])
        min_J_all.append(min(J_all))
    return min_J_all,P_all_test1,CR_all_test1
sum_times=1000 # number of runs
npool=50 # 50 threads, each use 1 core
n_a2=int(sum_times/npool) # number of runs per thread
n_iter=int(1e4) # number of iterations per run
pool = Pool(npool)
rel = pool.map(a2_iter, range(npool))
pool.close()
pool.join()
rel1=[]
rel2=[]
rel3=[]
for i in range(npool):
    rel1.append(rel[i][0])
    rel2.append(rel[i][1])
    rel3.append(rel[i][2])
J_all_min=np.array(rel1).reshape(-1,) #output J (Algorithm 2)
p0_all_min=np.array(rel2).reshape(-1,number_bin,number_bin)# output the scattering matrix (Algorithm 2)
CR_all_min=np.array(rel3).reshape(-1,nbinl,number_bin,number_bin) # output power spectrum between true-z bins (Algorithm 2)

np.save('J_all_a2.npy',J_all_min)
np.save('p0_all_a2.npy',p0_all_min)
np.save('CR_all_a2.npy',CR_all_min)
