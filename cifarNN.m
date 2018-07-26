clear; close all; clc;

% the data of 10000 x 3072 is too big to perform any operation, the fmincg doesn't give proper output
% so the data set have been reduced to 100 x 3072 matrix, in the other words consider only 1000 images
% >> X = data(1:1000,:);
% >> Y = labels(1:1000,:);
% >> save('cifarmini','X','Y');
% above given is the code to extract a part from the given matrix.

input_layer_size = 3072; % 32x32 color image, RGB 1024 each
num_labels = 10;	% ten various objects
hidden_layer_size = 25;

fprintf('loading the dataset...')

% load('data_batch_1.mat'); % data stored in arrays 'data' and 'lables'
% m = size(data,1);	% number of examples
load('cifarmini.mat');
m = size(X,1);

%==============================================================

% data = double(data);
% labels = double(labels);
X = double(X);
Y = double(Y);

% ============== display a random image (row 5) ====================

R = X(5,1:1024);
G = X(5,1025:2048);
B = X(5,2049:3072);

A = zeros(32,32,3,'uint8');

A(:,:,1) = reshape(R,32,32);
A(:,:,2) = reshape(G,32,32);
A(:,:,3) = reshape(B,32,32);

imshow(A)

% ==============================================================
% forward propagation operation with the pre initialized weights

fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
load('fwdpropw.mat');
%theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
%theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters 
nn_params = [theta1(:) ; theta2(:)];

% computing the feedforward cost

lambda = 1;

J = nnCostFunction1(nn_params, input_layer_size, hidden_layer_size,num_labels, X, Y, lambda);
  


fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


% now the training need to be performed.

fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 10);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction1(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, Y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
%[nn_params, cost] = fminunc(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%save('newweights','theta1','theta2');

pred = predict(theta1, theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == Y)) * 100);


%  To give you an idea of the network's output, you can also run
%  through the examples one at the a time to see what it is predicting.

%  Randomly permute examples
rp = randperm(m);

for i = 1:m
    % Display 
    fprintf('\nDisplaying Example Image\n');
    %displayData(X(rp(i), :));
	
	R = X(i,1:1024);
	G = X(i,1025:2048);
	B = X(i,2049:3072);

	Z = zeros(32,32,3,'uint8');

	Z(:,:,1) = reshape(R,32,32);
	Z(:,:,2) = reshape(G,32,32);
	Z(:,:,3) = reshape(B,32,32);

	imshow(Z)
	
    pred = predict(theta1, theta2, X(rp(i),:));
    fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
    
    % Pause
    fprintf('Program paused. Press enter to continue.\n');
    pause;
end