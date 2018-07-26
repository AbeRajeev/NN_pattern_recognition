clear; close all; clc;

input_layer_size = 3072; % 32x32 color image, RGB 1024 each
num_labels = 10;	% ten various objects
hidden_layer_size = 25;

fprintf('load the data...')

load('cifarmini.mat');
m = size(X,1);

X = double(X);
Y = double(Y);

R = X(5,1:1024);
G = X(5,1025:2048);
B = X(5,2049:3072);

A = zeros(32,32,3,'uint8');

A(:,:,1) = reshape(R,32,32);
A(:,:,2) = reshape(G,32,32);
A(:,:,3) = reshape(B,32,32);

imshow(A)
%================================================

fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
load('fwdpropw.mat');

pred = predict(theta1, theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == Y)) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;

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