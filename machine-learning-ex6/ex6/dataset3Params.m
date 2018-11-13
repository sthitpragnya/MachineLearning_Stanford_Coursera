function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_set = [0.1; 0.3; 1; 3; 10; 30];
sigma_set = [0.1; 0.3; 1; 3; 10; 30];
min_error = Inf;
best_set = [0; 0];

for i = 1:length(C_set)
  for j = 1:length(sigma_set)
    c1 = C_set(i);
    sigma1 = sigma_set(j);
    model= svmTrain(X, y, c1, @(x1, x2) gaussianKernel(x1, x2, sigma1));
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
    fprintf("C\t|\tsigma\t|\terror\t|\tmin_error\n");
    fprintf("%d\t|\t%d\t|\t%d\t|\t%d\n", c1, sigma1, error, min_error);
    if error < min_error
      min_error = error;
      best_set = [C_set(i); sigma_set(j)];
    endif
  endfor
endfor

C = best_set(1);
sigma = best_set(2);




% =========================================================================

end
