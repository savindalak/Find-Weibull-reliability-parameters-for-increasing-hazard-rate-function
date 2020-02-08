function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    [J_history(iter), gradient] = computeCost(X, y, theta);
    theta(1) =theta(1) - alpha*gradient(1);
    theta(2) =theta(2) - alpha*gradient(2);
        
  

end
J_history
end
