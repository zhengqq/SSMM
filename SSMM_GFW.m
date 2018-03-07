%   Distribution code Version 1.0 -- 02/20/2018 by Qingqing Zheng Copyright 2018, The Chinese University of Hong Kong.
%
%   The Code is created based on the method described in the following paper 
%   [1] "Sparse Support Matrix Machine", Qingqing Zheng, Fengyuan Zhu, Jing Qin, Badong Chen, Pheng-Ann Heng, Pattern Recognition, 2018. 
%  
%   The code and the algorithm are for non-comercial use only.

function [W,b] = SSMM_GFW(X,y,gamma,tau,ss)
% GFW_SSMM - Sparse Support Matrix Machine optimized by Generalized
% Forward-Backward algorithm. 
%   [W,b] = SSMM_GFW(X,y,gamma,tau,ss) trains a matrix classifier with
%   input matrices X, labels y, sparse parameter gamma, low rank parameter
%   tau and step size ss. 
%
%   Paras:
%   @X    : Input matrices of size m x d x n, where m and d are the sample
%   dimensions, n is the number of sample size.
%   @y    : Input labels for the input matrices, [1,-1] for binary cases. 
%   @gamma: Sparse paramter controlling the l1 norm. 
%   @tau  : Low rank parameter controlling the nuclear norm.
%   @ss   : step size. 


    sz = size(X);
    num_train = length(y);
    if sz(3)~= num_train
        fprintf('the number of input samples should be equal to the corresponding labels');
    end

    max_iter =10000;
    % ss = 1e-6; % step size

    Ng = 2;     % Num of non-differentiable term
    eps = 1e-3; 
    alpha = 3;  % Smooth parameter for smooth hinge loss


    Z1_old = zeros(sz(1),sz(2));
    Z2_old = zeros(sz(1),sz(2));
    b_old = 0;
    W_old = zeros(sz(1),sz(2));
    for iter = 1:max_iter  
        grad_F = smoothinge_gradient(X,y,W_old,b_old,alpha); 
        Z1_tmp = 2*W_old - Z1_old - ss*grad_F;
        Z2_tmp = 2*W_old - Z2_old - ss*grad_F;
        prox_Z1 = shrinkage(Z1_tmp,Ng*ss*tau);
        prox_Z2 = max(0,Z2_tmp - Ng*ss*gamma) - max(0, -Z2_tmp - Ng*ss*gamma);
        Z1_new = Z1_old + prox_Z1 - W_old;
        Z2_new = Z2_old + prox_Z2 - W_old;

        W_new = (Z1_new + Z2_new)/Ng;
        b_new = svm_kkt(X,y,W_new,b_old,ss);
        
        W_diff = W_new - W_old;
        if norm(W_diff(:)) < eps
            fprintf('converge! the iteration is %d\n',iter);
            break;
        else
            W_old = W_new;
            Z1_old = Z1_new;
            Z2_old = Z2_new;      
            b_old = b_new;      
        end
    end
    W = W_new;
    b = b_new;
end 



%% Update b with fixed step size 
% function b = svm_kkt(X,y,W,bo)
%     beta = 0.001;
%     sz0 = size(X);
%     X_vec = reshape(X,[sz0(1)*sz0(2),sz0(3)]);
%     W_vec = reshape(W,[sz0(1)*sz0(2),1]);
%     z = y.*(X_vec'*W_vec+bo);
%     bdiff = sum(-y(z<=1));
%     b = bo - bdiff*beta;
% end

function b = svm_kkt(X,y,W,bo,ss)
    beta = ss;
    alpha = 3;
    sz0 = size(X);
    X_vec = reshape(X,[sz0(1)*sz0(2),sz0(3)]);
    W_vec = reshape(W,[sz0(1)*sz0(2),1]);
    z = y.*(X_vec'*W_vec+bo);
    tmp = find(z>0 & z<1);
    bdiff = sum(-y(z<=0))+sum((z(tmp).^alpha-1).*y(tmp));
    b = bo - bdiff*beta;
end

function obj = objective_value(X,y,W,b,gamma,tau)
    sz0 = size(X);
    X_vec = reshape(X,[sz0(1)*sz0(2),sz0(3)]);
    W_vec = reshape(W,[sz0(1)*sz0(2),1]);
    term1 = tau*norm_nuc(W);
    term2 = gamma*sum(abs(W(:)));
    term3 = sum(max(0,1-y.*((X_vec'*W_vec)+b)));
    obj = term1+term2+term3;
end

function z = norm_nuc(X)
    z = sum(svd(X));
end
