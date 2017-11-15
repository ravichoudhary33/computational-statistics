%% loading data
clc
X1_test = csvread('Problem1_Input_Test.csv');
X1_train = csvread('Problem1_Input_Training.csv');

Y1_train = csvread('Problem1_Output_Training.csv');
Y1_test = csvread('Problem1_Output_Test.csv');

n = size(X1_train, 1);
m = size(X1_train, 2);

% display(n)
% display(m)

%% adding ones to first column
X1_train = [ones(n,1) X1_train];
X1_test = [ones(n,1) X1_test];

%% calculating beta hat for train data
beta_hat = pinv((X1_train'*X1_train))*X1_train'*Y1_train;

%% prediction on beta hat
Y1_pred = X1_train*beta_hat;

%% calculate sum of square error
mse_train_pred = sum((Y1_train - Y1_pred).^2)/n;
rmse_train_pred = sqrt(mse_train_pred);
display(rmse_train_pred)

%% applying recursive least square model
% formula beta_hat_n_plus_1 = beta_hat_n + K_n*e_n
M_n = pinv(X1_train'*X1_train);
Y1_test_pred = zeros(n, 1);

lambda = 1;
beta_hat_n = beta_hat;

for i = 1:n
    M_n_plus_1 = M_n - (M_n * X1_test(i,:)' * X1_test(i,:) * M_n) ./((1/lambda) + X1_test(i,:) * M_n * X1_test(i,:)');
    
    K_n = (1/lambda) .* (M_n_plus_1 * X1_test(i,:)');
    e_n = (Y1_test(i) - X1_test(i,:) * beta_hat_n);
    
    beta_hat_n_plus_1 = beta_hat_n + K_n .* e_n;
    
    Y1_test_pred(i) = X1_test(i,:) * beta_hat_n_plus_1;
    % update beta hat n+1
    M_n = M_n_plus_1;
    beta_hat_n = beta_hat_n_plus_1; 
end

%% calculate rmse for test data
mse_test_pred = sum((Y1_test - Y1_test_pred).^2)/n;
rmse_test_pred = sqrt(mse_test_pred);
display(rmse_test_pred)

%% ploting prediction and actual value on testing data plot
test_data_point = 1:size(Y1_test,1);
test_data_point = test_data_point';

plot(test_data_point, Y1_test , 'r-');
hold on;
plot(test_data_point, Y1_test_pred, 'g-');
hold off
title('y vs data point plot');
xlabel('X1\_test data point') % x-axis label
ylabel('y') % y-axis label
legend('yTest','yPred');