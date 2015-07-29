"""
PMF Implementation in Python.

"""
import pandas as pd
import numpy as np

from util import predict, read_data


# rand('state',0); 
# randn('state',0); 

np.random.seed(1234)

# if restart==1 
#   restart=0;
#   epsilon=50; % Learning rate 
#   lambda  = 0.01; % Regularization parameter 
#   momentum=0.8; 

restart = True
if restart:
    restart = False
    epsilon = 50    # Learning rate
    lambda_  = 0.01  # Regularization parameter
    momentum = 0.8

#   epoch=1; 
#   maxepoch=50; 

    epoch = 1
    max_epoch = 50

#   load moviedata % Triplets: {user_id, movie_id, rating} 
#   mean_rating = mean(train_vec(:,3)); 
#  
#   pairs_tr = length(train_vec); % training data 
#   pairs_pr = length(probe_vec); % validation data 

    header_names, train, probe = read_data()
    uid, iid, target = header_names

    mean_rating = train[target].mean()
    ratings_test = probe[target].values.astype(np.double)

    nd_train = train.shape[0]  # number of training dyads
    nd_probe = probe.shape[0]  # number of probe dyads

#   numbatches= 9; % Number of batches  
#   num_m = 3952;  % Number of movies 
#   num_p = 6040;  % Number of users 
#   num_feat = 10; % Rank 10 decomposition 

    num_batches = 9
    N = 100000  # number of training triplets processed per batch
    n = 6040    # number of users
    m = 3952    # number of items
    nf = 10     # number of features (latent factor vector dimensionality)

#   w1_M1     = 0.1*randn(num_m, num_feat); % Movie feature vectors
#   w1_P1     = 0.1*randn(num_p, num_feat); % User feature vecators
#   w1_M1_inc = zeros(num_m, num_feat);
#   w1_P1_inc = zeros(num_p, num_feat);

    # Randomly initialize user and item feature vectors.
    w_u = 0.1 * np.random.randn(n, nf)  # User
    w_i = 0.1 * np.random.randn(m, nf)  # Item

    # Allocate space for feature vector update vectors.
    w_u_update = np.zeros((n, nf))
    w_i_update = np.zeros((m, nf))


# for epoch = epoch:maxepoch
#   rr = randperm(pairs_tr);
#   train_vec = train_vec(rr,:);
#   clear rr 

err_train = np.zeros(max_epoch)
err_valid = np.zeros(max_epoch)
for epoch in range(max_epoch):
    rr = np.random.permutation(nd_train)  ## rr = random range
    train = train.ix[rr]
    del rr  # necessary?

#   for batch = 1:numbatches
#     fprintf(1,'epoch %d batch %d \r',epoch,batch);
#     N=100000; % number training triplets per batch 

    for batch in range(num_batches):
        print 'epoch %d, batch %d' % (epoch + 1, batch + 1)

#     aa_p   = double(train_vec((batch-1)*N+1:batch*N,1));
#     aa_m   = double(train_vec((batch-1)*N+1:batch*N,2));
#     rating = double(train_vec((batch-1)*N+1:batch*N,3));

        train_subset = train.ix[range(batch*N, (batch+1)*N)]
        uids = train_subset[uid].values
        iids = train_subset[iid].values
        ratings = train_subset[target].values

#     rating = rating-mean_rating; % Default prediction is the mean rating. 

        # Default prediction is the mean rating, so subtract it.
        ratings = ratings - mean_rating

#     %%%%%%%%%%%%%% Compute Predictions %%%%%%%%%%%%%%%%%
#     pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2);
#     f = sum( (pred_out - rating).^2 + ...
#         0.5*lambda*( sum( (w1_M1(aa_m,:).^2 + w1_P1(aa_p,:).^2),2)));

        # Compute predictions.
        pred_out = np.sum(w_i[iids] * w_u[uids], 1)
        error = pred_out - ratings
        regular = np.sum(w_i[iids] ** 2 + w_u[uids] ** 2, 1)
        f = np.sum(error ** 2 + 0.5 * lambda_ * regular)

#     %%%%%%%%%%%%%% Compute Gradients %%%%%%%%%%%%%%%%%%%
#     IO = repmat(2*(pred_out - rating),1,num_feat);
#     Ix_m=IO.*w1_P1(aa_p,:) + lambda*w1_M1(aa_m,:);
#     Ix_p=IO.*w1_M1(aa_m,:) + lambda*w1_P1(aa_p,:);
# 
#     dw1_M1 = zeros(num_m,num_feat);
#     dw1_P1 = zeros(num_p,num_feat);
# 
#     for ii=1:N
#       dw1_M1(aa_m(ii),:) =  dw1_M1(aa_m(ii),:) +  Ix_m(ii,:);
#       dw1_P1(aa_p(ii),:) =  dw1_P1(aa_p(ii),:) +  Ix_p(ii,:);
#     end

        # Compute gradients.
        IO = np.repeat(2 * error, nf).reshape(error.shape[0], nf)
        Ix_u = IO * w_i[iids] + lambda_ * w_u[uids]
        Ix_i = IO * w_u[uids] + lambda_ * w_i[iids]

        dw_u = np.zeros((n, nf))
        dw_i = np.zeros((m, nf))

        for ii in range(N):
            dw_u[uids[ii]] += Ix_u[ii]
            dw_i[iids[ii]] += Ix_i[ii]

#     %%%% Update movie and user features %%%%%%%%%%%
# 
#     w1_M1_inc = momentum*w1_M1_inc + epsilon*dw1_M1/N;
#     w1_M1 =  w1_M1 - w1_M1_inc;
# 
#     w1_P1_inc = momentum*w1_P1_inc + epsilon*dw1_P1/N;
#     w1_P1 =  w1_P1 - w1_P1_inc;
#   end 

        # Update user and item feature vectors.
        w_u_update = momentum * w_u_update + epsilon * (dw_u / N)
        w_u -= w_u_update

        w_i_update = momentum * w_i_update + epsilon * (dw_i / N)
        w_i -= w_i_update

#   %%%%%%%%%%%%%% Compute Predictions after Paramete Updates %%%%%%%%%%%%%%%%%
#   pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2);
#   f_s = sum( (pred_out - rating).^2 + ...
#         0.5*lambda*( sum( (w1_M1(aa_m,:).^2 + w1_P1(aa_p,:).^2),2)));
#   err_train(epoch) = sqrt(f_s/N);

    # Compute predictions after parameter updates.
    pred_out = np.sum(w_i[iids] * w_u[uids], 1)
    error = pred_out - ratings
    regular = np.sum(w_i[iids] ** 2 + w_u[uids] ** 2, 1)
    f_s = np.sum(error ** 2 + 0.5 * lambda_ * regular)
    err_train[epoch] = np.sqrt(f_s / N)

#   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   %%% Compute predictions on the validation set %%%%%%%%%%%%%%%%%%%%%% 
#   NN=pairs_pr;
# 
#   aa_p = double(probe_vec(:,1));
#   aa_m = double(probe_vec(:,2));
#   rating = double(probe_vec(:,3));
# 
#   pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2) + mean_rating;
#   ff = find(pred_out>5); pred_out(ff)=5; % Clip predictions 
#   ff = find(pred_out<1); pred_out(ff)=1;
# 
#   err_valid(epoch) = sqrt(sum((pred_out- rating).^2)/NN);
#   fprintf(1, 'epoch %4i batch %4i Training RMSE %6.4f  Test RMSE %6.4f  \n', ...
#               epoch, batch, err_train(epoch), err_valid(epoch));
#   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Compute predictions on the validation set.
    predictions = predict(w_u, w_i, mean_rating, probe)
    error = predictions - probe[target].values

    err_valid[epoch] = np.sqrt((error ** 2).sum() / nd_probe)
    print 'epoch %4d, batch %4d, Training RMSE: %6.4f, Test RMSE: %6.4f\n' % (
        epoch, batch, err_train[epoch], err_valid[epoch])

#   if (rem(epoch,10))==0
#      save pmf_weight w1_M1 w1_P1
#   end
# 
# end 

    if (epoch + 1) % 10 == 0:
        np.savetxt('w_u.csv', w_u, delimiter=',')
        np.savetxt('w_i.csv', w_i, delimiter=',')

# Finally, save weights at the end.
np.savetxt('w_u.csv', w_u, delimiter=',')
np.savetxt('w_i.csv', w_i, delimiter=',')
