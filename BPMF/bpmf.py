"""
Bayesian PMF (BPMF) Implementation in Python.

"""
import sys
import pandas as pd
import numpy as np

from util import predict, wishrnd, read_data, make_matrix


def cov(x):
    return np.cov(x, rowvar=0)


# rand('state',0);
# randn('state',0);

np.random.seed(1234)

# if restart==1 
#   restart=0; 
#   epoch=1; 
#   maxepoch=50; 

restart = True
if restart:
    restart = False
    epoch = 1
    max_epoch = 50

#   iter=0; 
#   num_m = 3952;
#   num_p = 6040;
#   num_feat = 10;

    iter_ = 0   # track iteration number
    n = 6040    # number of users
    m = 3952    # number of items
    nf = 10     # number of features (latent factor vector dimensionality)

#   % Initialize hierarchical priors 
#   beta=2; % observation noise (precision) 
#   mu_u = zeros(num_feat,1);
#   mu_m = zeros(num_feat,1);
#   alpha_u = eye(num_feat);
#   alpha_m = eye(num_feat);  

    beta = 2  # observation noise (precision)
    mu_u = np.zeros((nf, 1))
    mu_i = np.zeros((nf, 1))
    alpha_u = np.eye(nf)
    alpha_i = np.eye(nf)

#   % parameters of Inv-Whishart distribution (see paper for details) 
#   WI_u = eye(num_feat);
#   b0_u = 2;
#   df_u = num_feat;
#   mu0_u = zeros(num_feat,1);

    # Parameters of Inverse-Wishart distribution (User).
    W_u = np.eye(nf)
    b0_u = 2
    df_u = nf  # degrees of freedom
    mu0_u = np.zeros((nf, 1))

#   WI_m = eye(num_feat);
#   b0_m = 2;
#   df_m = num_feat;
#   mu0_m = zeros(num_feat,1);

    # Parameters of Inverse-Wishart distribution (Item).
    W_i = np.eye(nf)
    b0_i = 2
    df_i = nf  # degrees of freedom
    mu0_i = np.zeros((nf, 1))

#   load moviedata
#   mean_rating = mean(train_vec(:,3));
#   ratings_test = double(probe_vec(:,3));

#   pairs_tr = length(train_vec);
#   pairs_pr = length(probe_vec);

    header_names, train, probe = read_data()
    uid, iid, target = header_names

    mean_rating = train[target].mean()
    ratings_test = probe[target].values.astype(np.double)

#   fprintf(1,'Initializing Bayesian PMF using MAP solution found by PMF \n'); 
#   makematrix

    print 'Converting training triples to matrix format'
    args = header_names + [False]
    train_mat = make_matrix(train, *args)
    # train_mat = row_to_mat(train, *header_names)

    print 'Initializing BPMF using MAP solution from PMF'

#   load pmf_weight
#   err_test = cell(maxepoch,1);
# 
#   w1_P1_sample = w1_P1; 
#   w1_M1_sample = w1_M1; 
#   clear w1_P1 w1_M1;

    w_u_sample = np.loadtxt('w_u.csv', delimiter=',')
    w_i_sample = np.loadtxt('w_i.csv', delimiter=',')
    # err_test = np.empty((max_epoch, 1), dtype=object)

#   % Initialization using MAP solution found by PMF. 
#   %% Do simple fit
#   mu_u = mean(w1_P1_sample)';
#   d=num_feat;
#   alpha_u = inv(cov(w1_P1_sample));
# 
#   mu_m = mean(w1_M1_sample)';
#   alpha_m = inv(cov(w1_P1_sample));
# 
#   count=count';
#   probe_rat_all = pred(w1_M1_sample,w1_P1_sample,probe_vec,mean_rating);
#   counter_prob=1; 
# 
# end

    # Initialization using MAP solution found by PMF.
    mu_u = w_u_sample.mean(axis=0)
    alpha_u = np.linalg.inv(cov(w_u_sample))

    mu_i = w_i_sample.mean(axis=0)
    alpha_i = np.linalg.inv(cov(w_i_sample))

    ratings_mat = train_mat.T
    probe_rat_all = predict(w_u_sample, w_i_sample, mean_rating, probe)
    counter_prob = 1
    print 'Done initializing'


# for epoch = epoch:maxepoch
# 
#   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   %%% Sample from movie hyperparams (see paper for details)  
#   N = size(w1_M1_sample,1);
#   x_bar = mean(w1_M1_sample)'; 
#   S_bar = cov(w1_M1_sample); 
# 
#   WI_post = inv(inv(WI_m) + N/1*S_bar + ...
#             N*b0_m*(mu0_m - x_bar)*(mu0_m - x_bar)'/(1*(b0_m+N)));
#   WI_post = (WI_post + WI_post')/2;
# 
#   df_mpost = df_m+N;
#   alpha_m = wishrnd(WI_post,df_mpost);   
#   mu_temp = (b0_m*mu0_m + N*x_bar)/(b0_m+N);  
#   lam = chol( inv((b0_m+N)*alpha_m) ); lam=lam';
#   mu_m = lam*randn(num_feat,1)+mu_temp;

ngibbs = 2
overall_err = np.zeros(max_epoch * ngibbs)
prev_rmse = np.inf
stopping_threshold = 0.00001

for epoch in xrange(max_epoch):
    print '\nEpoch %d' % (epoch + 1)

    # Sample from movie hyperparameters.
    N = w_i_sample.shape[0]
    x_bar = np.mean(w_i_sample, 0).reshape(nf, 1)
    S_bar = cov(w_i_sample)

    tmp = mu0_i - x_bar
    W_post = np.linalg.inv(
        np.linalg.inv(W_i) + (N * S_bar) +
        N * b0_i * tmp.dot(tmp.T) / (b0_i + N))
    W_post = (W_post + W_post.T) / 2

    df_i_post = df_i + N
    alpha_i = wishrnd(W_post, df_i_post)
    mu_tmp = (b0_i * mu0_i + N * x_bar) / (b0_i + N)
    lam = np.linalg.cholesky(np.linalg.inv((b0_i + N) * alpha_i)).T
    mu_i = lam.dot(np.random.randn(nf)) + mu_tmp.reshape(nf)

#   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   %%% Sample from user hyperparams
#   N = size(w1_P1_sample,1);
#   x_bar = mean(w1_P1_sample)';
#   S_bar = cov(w1_P1_sample);
# 
#   WI_post = inv(inv(WI_u) + N/1*S_bar + ...
#             N*b0_u*(mu0_u - x_bar)*(mu0_u - x_bar)'/(1*(b0_u+N)));
#   WI_post = (WI_post + WI_post')/2;
#   df_mpost = df_u+N;
#   alpha_u = wishrnd(WI_post,df_mpost);
#   mu_temp = (b0_u*mu0_u + N*x_bar)/(b0_u+N);
#   lam = chol( inv((b0_u+N)*alpha_u) ); lam=lam';
#   mu_u = lam*randn(num_feat,1)+mu_temp;

    N = w_u_sample.shape[0]
    x_bar = np.mean(w_u_sample, 0).reshape(nf, 1)
    S_bar = cov(w_u_sample)

    tmp = mu0_u - x_bar
    W_post = np.linalg.inv(
        np.linalg.inv(W_u) + (N * S_bar) +
        N * b0_u * tmp.dot(tmp.T) / (b0_u + N))
    W_post = (W_post + W_post.T) / 2

    df_u_post = df_u + N
    alpha_u = wishrnd(W_post, df_u_post)
    mu_tmp = (b0_u * mu0_u + N * x_bar) / (b0_u + N)
    lam = np.linalg.cholesky(np.linalg.inv((b0_u + N) * alpha_u)).T
    mu_u = lam.dot(np.random.randn(nf)) + mu_tmp.reshape(nf)

#   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   % Start doing Gibbs updates over user and 
#   % movie feature vectors given hyperparams.  
# 
#   for gibbs=1:2 
#     fprintf(1,'\t\t Gibbs sampling %d \r', gibbs);

    # Start Gibbs updates over user/item feature vectors given hyperparams.
    for gibbs in range(ngibbs):
        print '\tGibbs sampling %d' % (gibbs + 1)

#     %%% Infer posterior distribution over all movie feature vectors 
#     count=count';
#     for mm=1:num_m
#        fprintf(1,'movie =%d\r',mm);
#        ff = find(count(:,mm)>0);
#        MM = w1_P1_sample(ff,:);
#        rr = count(ff,mm)-mean_rating;
#        covar = inv((alpha_m+beta*MM'*MM));
#        mean_m = covar * (beta*MM'*rr+alpha_m*mu_m);
#        lam = chol(covar); lam=lam'; 
#        w1_M1_sample(mm,:) = lam*randn(num_feat,1)+mean_m;
#      end

        # Infer posterior distribution over all item feature vectors.
        ratings_mat = ratings_mat.T
        for mm in range(m):
            # print 'item =%d' % mm
            # uids = (ratings_mat[:, mm] > 0).toarray().reshape(n)
            uids = ratings_mat[:, mm] > 0
            MM = w_u_sample[uids]
            # rr = (ratings_mat[uids, mm].toarray() -
            #       mean_rating).reshape(MM.shape[0])
            rr = ratings_mat[uids, mm] - mean_rating
            covar = np.linalg.inv(alpha_i + beta * MM.T.dot(MM))
            mean_i = covar.dot(beta * MM.T.dot(rr) + alpha_i.dot(mu_i))
            lam = np.linalg.cholesky(covar).T
            w_i_sample[mm] = lam.dot(np.random.randn(nf)) + mean_i

#     %%% Infer posterior distribution over all user feature vectors 
#      count=count';
#      for uu=1:num_p
#        fprintf(1,'user  =%d\r',uu);
#        ff = find(count(:,uu)>0);
#        MM = w1_M1_sample(ff,:);
#        rr = count(ff,uu)-mean_rating;
#        covar = inv((alpha_u+beta*MM'*MM));
#        mean_u = covar * (beta*MM'*rr+alpha_u*mu_u);
#        lam = chol(covar); lam=lam'; 
#        w1_P1_sample(uu,:) = lam*randn(num_feat,1)+mean_u;
#      end
#    end 

        # Infer posterior distribution over all user feature vectors.
        ratings_mat = ratings_mat.T
        for uu in range(n):
            # print 'user =%d' % uu
            # iids = (ratings_mat[:, uu] > 0).toarray().reshape(m)
            iids = ratings_mat[:, uu] > 0
            MM = w_i_sample[iids]
            # rr = (ratings_mat[iids, uu].toarray() -
            #       mean_rating).reshape(MM.shape[0])
            rr = ratings_mat[iids, uu] - mean_rating
            covar = np.linalg.inv(alpha_u + beta * MM.T.dot(MM))
            mean_u = covar.dot(beta * MM.T.dot(rr) + alpha_u.dot(mu_u))
            lam = np.linalg.cholesky(covar).T
            w_u_sample[uu] = lam.dot(np.random.randn(nf)) + mean_u

#    probe_rat = pred(w1_M1_sample,w1_P1_sample,probe_vec,mean_rating);
#    probe_rat_all = (counter_prob*probe_rat_all + probe_rat)/(counter_prob+1);
#    counter_prob=counter_prob+1;

        probe_rat = predict(w_u_sample, w_i_sample, mean_rating, probe)
        probe_rat_all = ((counter_prob * probe_rat_all + probe_rat) /
                         (counter_prob + 1))
        counter_prob += 1

#    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#    %%%%%%% Make predictions on the validation data %%%%%%%
#    temp = (ratings_test - probe_rat_all).^2;
#    err = sqrt( sum(temp)/pairs_pr);
# 
#    iter=iter+1;
#    overall_err(iter)=err;
# 
#   fprintf(1, '\nEpoch %d \t Average Test RMSE %6.4f \n', epoch, err);
# 
# end 

    # Make predictions on the validation data.
    error = ratings_test - probe_rat_all
    rmse = np.sqrt((error ** 2).sum() / probe.shape[0])

    iter_ += 1
    overall_err[iter_] = rmse

    print '\tAverage Test RMSE: %6.4f' % rmse

    if (prev_rmse - rmse) <= stopping_threshold:
        print '\tStopping threshold reached'
        break
    else:
        prev_rmse = rmse
