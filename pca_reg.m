%% loading data
clc

x1 = csvread('x1.csv');
x2 = csvread('x2.csv');
x3 = csvread('x3.csv');
x4 = csvread('x4.csv');
y = csvread('y.csv');

X = [x1, x2, x3, x4];
n = size(X,1);
p = size(X,2);

%% new data matrix after adding column of ones
X_new = [ones(n,1), X];
p_new = size(X_new, 2);

%% split the data into traing and testing matrix by (60% and 40%)
X_train = X_new(1:120, :);
X_test = X_new(121:end, :);
y_train = y(1:120, :);
y_test = y(121:end, :);

%% finding the beta hat for training data
beta_hat = X_train\y_train;
% display(beta_hat);

%% now applying svd decomposition on X_new
[U,S,V] = svd(X_train);

%% get the diag entry of S
sigma = diag(S);
% display(sigma);

%% compute the sigma ratios
sigma_sum = sum(sigma);
% display(sigma_sum);

%% creating a vector of size p-1 to store sigma ratios
sigma_ratio = zeros(p_new-1, 1);
sigma_ratio(1,1) = sigma(1,1);

for i = 2:p_new-1
    sigma_ratio(i,1) = sigma_ratio(i-1,1) + sigma(i,1);
end

sigma_ratio = sigma_ratio/sigma_sum;
% display(sigma_ratio);

%% calculating informative part and non informative part (removing lower sigma)
r = size(sigma,1) - 2;
U_I = U(:,1:r);
S_I = S(1:r,1:r);
V_I = V(:, 1:r);

%% calculating X informative and X non informative
X_I = U_I * S_I * V_I';
X_N = X_train - X_I;

%% calculating Wr

Wr = X_train * V_I;
% display(size(Wr));

%% calculating gamma hat for Wr
gamma_hat = Wr\y_train;

%% calculating beta hat pca using gamma hat
beta_hat_pca = V_I * gamma_hat;

%% testing prediction

y_pred = X_test * beta_hat;
y_pred_pca = X_test * beta_hat_pca;

% display(y_pred);
% display(y_pred_pca);

%% ploting prediction and actual value on testing data plot
test_data_point = 1:size(y_test,1);
test_data_point = test_data_point';

plot(test_data_point, y_test , 'r-');
hold on;
plot(test_data_point, y_pred, 'g-');
hold on
plot(test_data_point, y_pred_pca, 'b-');
hold off

title('y vs data point plot');
xlabel('X\_test data point') % x-axis label
ylabel('y') % y-axis label
legend('yTest','yPred', 'yPredPCA');

%% calculating covariance of beta hat
cov_beta_hat = cov(beta_hat);
display(cov_beta_hat);

%% calculating covariance of beta hat pca
cov_beta_hat_pca = cov(beta_hat_pca);
display(cov_beta_hat_pca);

%%  calculating rmse 
rmse_test_pred = sqrt(mean((y_test-y_pred).^2));
rmse_test_pred_pca = sqrt(mean((y_test-y_pred_pca).^2));
display(rmse_test_pred);
display(rmse_test_pred_pca);

%% save the prediction in separate csv file
csvwrite('y_pred.csv', y_pred);
csvwrite('y_pred_pca.csv', y_pred_pca);

%% from the out put clearly cov(beta_hat) >= cov(beta_hat_pca)