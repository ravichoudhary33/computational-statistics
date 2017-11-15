%% loading data
clc
X1_test = csvread('Problem1_Input_Test.csv');
X1_train = csvread('Problem1_Input_Training.csv');

Y3_train = csvread('Problem3_Output_Training.csv');
Y3_test = csvread('Problem3_Output_Test.csv');

n = size(X1_train, 1);
m = size(X1_train, 2);

% display(n)
% display(m)

%% adding ones to first column
X1_train = [ones(n,1) X1_train];
X1_test = [ones(n,1) X1_test];

%% initialize y_test_pred
Y3_test_pred = ones(n, 3);

%% calculating weight for each test data
d = zeros(n, 1);
w = zeros(n, 3);

%% i for iterating through test data and j for train data
phi = 1;
epsilon = 0.0001;

for i = 1:n
   for j = 1:n
      % d(j) =  dot(X1_train(j, :), X1_test(i, :))/(norm(X1_train(j, :))*norm(X1_test(i, :)));
      d(j) = norm(X1_train(j,:) - X1_test(i,:));
      % w(j) = exp((-d(j)^2));
   end
   % normalizing d 
   d = (d - mean(d))/std(d);
   d = abs(d);
   w(:,1) = d <= 1;
   w(:,2) = exp((-d.^2)/(2*phi));
   w(:,3) = exp(-d.^2);
   for k = 1:n
      if d(k) > epsilon
          w(k,3) = 1/d(k);
      else
          w(k,3) = 10;
      end
   end
   
   % for first test data we got d and w, now compute beta_hat
   % display(d);
   
   for l = 1:3
       V = diag(w(:, l));
       Inv_V = V;
       beta_hat = pinv(X1_train' * Inv_V * X1_train) * X1_train' * Inv_V * Y3_train;
       % display(beta_hat);
       Y3_test_pred(i, l) = X1_test(i, :) * beta_hat;
   end
   
end
display(Y3_test_pred);

%% calculate rmse for test data
rmse_test_pred = zeros(3,1);

for m = 1:3
    mse_test_pred = mean((Y3_test - Y3_test_pred(:,m)).^2);
    rmse_test_pred(m) = sqrt(mse_test_pred);
end
display(rmse_test_pred);

%% ploting prediction and actual value on testing data plot
test_data_point = 1:size(Y3_test,1);
test_data_point = test_data_point';

x0=10;
y0=10;
width=550;
height=400;
set(gcf,'units','points','position',[x0,y0,width,height])

for p = 1:3
    subplot(3,1,p);
    plot(test_data_point, Y3_test , 'r-');
    hold on;
    plot(test_data_point, Y3_test_pred(:,p), 'g-');
    hold off
    if p == 1
        title('y vs data point plot for weight 1');
    elseif p == 2
        title('y vs data point plot for weight 2');
    else
        title('y vs data point plot for weight 3');
    end
    xlabel('X1\_test data point') % x-axis label
    ylabel('y') % y-axis label
    legend('yTest','yPred');
end
