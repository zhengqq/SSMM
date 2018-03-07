% Example: a simple binary demo with Caltech Face LBP features. 
% =====================
% Step 1: load the input data
% Step 2: Fine tune the free parameter
% Step 3: Train the binary matrix classifier 
% Step 4: Predict results on training data
% Step 5: Predict results on testing data


%% Load the LBP features ,dimensional 59*100
load('CaltechFace_LBP.mat');
% Feature scaling
for i = 1:size(X,3)
    tmp = X(:,:,i);
    X(:,:,i) = X(:,:,i)/max(tmp(:));
end
for j = 1:size(X_test,3)
    tmp = X_test(:,:,j);
    X_test(:,:,j) = X_test(:,:,j)/max(tmp(:));
end

%% Set the free parameter 
tau = 0.01;    % parameter for low rank term
gamma = 0.001; % parameter for sparse term 
ss = 0.001;    % step size, should be small (theta in paper)

fprintf('tau = %f,gamma = %f, ss = %f\n',tau,gamma,ss);


%% Train the binary matrix classifier
tic;
[W,b] = SSMM_GFW(X,y,gamma,tau,ss);
fprintf('Training time = %f\n',toc);

%% Predict the training accuracy
sz = size(X);
sz_test = size(X_test);
W1 = reshape(W,[sz(1)*sz(2),1]);
X1 = reshape(X,[sz(1)*sz(2),sz(3)]);
y_hat = sign(X1'*W1+b);
acc = sum(y_hat == y)/length(y);
fprintf('Training acc is %.4f\n',acc);

%% Predict the testing accuracy
tic;
X_test1 = reshape(X_test,[sz_test(1)*sz_test(2),sz_test(3)]);
y_hat_test = sign(X_test1'*W1+b);
acc_test = sum(y_hat_test == y_test)/length(y_test);

%% Print the results 
fprintf('Testing acc is %.4f\n',acc_test);
fprintf('Testing time is %f \n',toc);





