function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, Y, lambda)
								   
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%theta1 = uint8(theta1);
%theta2 = uint8(theta2);				 
				 
% Setup some useful variables
m = size(X, 1);
 
		 
% You need to return the following variables correctly 
J = 0;
theta1_grad = zeros(size(theta1));
theta2_grad = zeros(size(theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1 = [ones(m,1) X];

z2 = a1 * theta1';

a2 = sigmoid (z2);

a2 = [ones(size(a2,1),1) a2];

z3 = a2 * theta2';

h = sigmoid(z3);

yTerm = zeros(m,num_labels);

for i = 1:m

	%yTerm(i,labels(i)) = 1;
	yTerm = (1:num_labels) == Y(i);
	
end


J = (1/m) * sum(sum(-1 * yTerm .* log(h) - (1-yTerm) .* log(1 - h) ));
%J = (1/m) * sum(sum(-1 * (yTerm * log(h)) - ((1-yTerm) * log(1 - h)) ));
reg = (lambda/(2*m)) * (sum(sum(theta1(:,2:end).^2)) + sum(sum(theta2(:,2:end).^2)));

J = J + reg;

% -------------------------------------------------------------

for t = 1:m

	a1 = X(t,:)';
	
	a1 = [1 ; a1];
	
	z2 = theta1 * a1;
	
	
	a2 = [1 ; sigmoid(z2)];
	
	z3 = theta2 * a2;
	
	a3 = sigmoid(z3);
	
	yy = ([1:num_labels] == Y(t))';
	
	delta3 = a3 - yy;
	
	delta2 = (theta2' * delta3) .* [1;(sigmoidGradient(z2))];
	
	delta2 = delta2(2:end);
	
	theta1_grad = theta1_grad + (delta2 * a1');
	theta2_grad = theta2_grad + (delta3 * a2');
	
end

theta1_grad = (1/m) * theta1_grad + (lambda/m) * [zeros(size(theta1,1),1) theta1(:,2:end)];
theta2_grad = (1/m) * theta2_grad + (lambda/m) * [zeros(size(theta2,1),1) theta2(:,2:end)];


% =========================================================================

% Unroll gradients
grad = [theta1_grad(:) ; theta2_grad(:)];


end
