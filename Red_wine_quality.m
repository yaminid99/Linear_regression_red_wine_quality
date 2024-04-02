clc;
clear all;
close all;
%% Author: Yamini Devi
% Input: Dataset.
% Output: 

%%
%load data from file
% redWineData=load('winequality_red.csv');
filename = 'winequality_red.csv';
redWineData = readtable("winequality_red.csv");
redWineData = table2array(redWineData);
% redWineData = redWineData.';
%Split the data into train and test sets
training_samples = 1500;
yTrain = (redWineData(1:training_samples,12));
redWineTrain = (redWineData(1:training_samples,1:11));
redWineTest = (redWineData(training_samples+1:1599,1:11));
yTest = (redWineData(training_samples+1:1599,12));


%% Rohit
yTrain = transpose(redWineData(1:training_samples,12));
redWineTrain = transpose(redWineData(1:training_samples,1:11));
redWineTest = transpose(redWineData(training_samples+1:1599,1:11));
yTest = transpose(redWineData(training_samples+1:1599,12));

X =  redWineTrain;
Y = yTrain;
w_star = pinv(X*X.')*X*Y.';
Mean_squared_Error_Train = mean((abs(w_star.'*X - Y)).^2);
disp("Mean Squared Error Performance on train set: ")
disp(Mean_squared_Error_Train);
Mean_squared_Error_Test = mean((abs(w_star.'*redWineTest - yTest)).^2);
disp("Mean Squared Error Performance on test set: ")
disp(Mean_squared_Error_Test);


%Plot error rate of data samples compared with the frequency of data
%samples in our dataset.
figure
all_data = [X,redWineTest];
all_label = [Y,yTest];
eachSqErr= (all_label-w_star.'*all_data).^2;
h = hist(eachSqErr,0:.01:3);
plot(0:.01:3,h,'linewidth',3);
title("Histogram of Squared error");
grid on;
xlabel('AvgSquare Test Error'); 
ylabel('# of occurances');


figure
plot(Y,w_star.'*X,'+');
xlabel("Actual Value")
ylabel("Predicted value")
title(['Average Train error = ' num2str(Mean_squared_Error_Train) ' & Average Test error = ' num2str(Mean_squared_Error_Test)]);
%%



%Convert the train dataset such that it can accomodate for patterns in data
%with higher degree polynomial terms.
%Note: number of columns of M <=30. First column of all ones is the bias term.

%Different model configurations with different costs:
%M=[ones(length(redWineTrain),1) redWineTrain(:,1:11)];  %0.4168
%M=[ones(length(redWineTrain),1) redWineTrain(:,1:11) redWineTrain(:,1:11).^2];  %0.3996
% M=[ones(length(redWineTrain),1) ...
%     redWineTrain(:,1:11).^0.1 ...
%     redWineTrain(:,11).^24.*redWineTrain(:,5) ...
%     sin(redWineTrain(:,2)).^1.3.*redWineTrain(:,7).*redWineTrain(:,11) ...
%     redWineTrain(:,5).^-60.*redWineTrain(:,2) ...
%     sin(redWineTrain(:,1)).^9.*redWineTrain(:,5).*redWineTrain(:,4) ...
%     redWineTrain(:,7).^6.*redWineTrain(:,4).*redWineTrain(:,6) ...
%     redWineTrain(:,1).*redWineTrain(:,7).*redWineTrain(:,11).*redWineTrain(:,2) ...
%     redWineTrain(:,1:11).^0.4.*redWineTrain(:,8).*redWineTrain(:,8) ...
%     sin(redWineTrain(:,4)).^315.*redWineTrain(:,8).*redWineTrain(:,10) ...
%     ];
% 
% %Normal Equations method to learn the model coefficients.
% w = ((M'*M)\M')*yTrain;
% 
% %Convert the test set into the same form as train set. 
% M_test=[ones(length(redWineTest),1) ...
%     redWineTest(:,1:11).^0.1 ...
%     redWineTest(:,11).^24.*redWineTest(:,5) ...
%     sin(redWineTest(:,2)).^1.3.*redWineTest(:,7).*redWineTest(:,11) ...
%     redWineTest(:,5).^-60.*redWineTest(:,2) ...
%     sin(redWineTest(:,1)).^9.*redWineTest(:,5).*redWineTest(:,4) ...
%     redWineTest(:,7).^6.*redWineTest(:,4).*redWineTest(:,6) ...
%     redWineTest(:,1).*redWineTest(:,7).*redWineTest(:,11).*redWineTest(:,2) ...
%     redWineTest(:,1:11).^0.4.*redWineTest(:,8).*redWineTest(:,8) ...
%     sin(redWineTest(:,4)).^315.*redWineTest(:,8).*redWineTest(:,10) ...
%     ];
% 
% %check performance on train set
% disp("Performance on train set: ")
% avgTrainSqErr=sum((yTrain-M*w).^2)./length(redWineTrain)
% 
% %check performance on test set
% disp("Performance on test set: ")
% avgTestSqErr=sum((yTest-M_test*w).^2)./length(redWineTest)
% 
% 
% %Visualize error or performance for overall data
% %Plot test set predictions vs the actual labels
% figure
% disp("here: ")
% plot(yTest,M_test*w,'+');
% title(sprintf('avgSqErr=%6.4f; avDevErr=%6.4f',avgTrainSqErr,avgTestSqErr));
% print -dpng error_hist.png
% 
% %Plot error rate of data samples compared with the frequency of data
% %samples in our dataset.
% figure
% all_data = [M;M_test];
% all_label = [yTrain;yTest];
% eachSqErr= (all_label-all_data*w).^2;
% h = hist(eachSqErr,0:.01:3);
% plot(0:.01:3,h,'linewidth',3); 
% grid on;
% xlabel('AvgSquare Test Error'); 
% ylabel('# of occurances');
% print -dpng errror_vs_occurrence_frequency.png