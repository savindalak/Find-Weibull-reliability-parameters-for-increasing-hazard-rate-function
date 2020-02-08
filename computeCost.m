function [J, gradient] = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
% Initialize some useful values
m = length(y); % number of training examples


J = 0;

J= sum(((exp(-(X.*(1/theta(1))).^theta(2)))-y).^2)/(2*m);

gradient1= [((exp(-(X.*(1/theta(1))).^theta(2)))-y)].*[exp(-(X.*(1/theta(1))).^theta(2))].*[X.^theta(2)*theta(2)*(1/theta(1).^(theta(2)+1))].*(1/m);
gradient2= [((exp(-(X.*(1/theta(1))).^theta(2)))-y)].*[exp(-(X.*(1/theta(1))).^theta(2))].*[-((X./theta(1)).^theta(2)).*log(X/theta(1))].*(1/m);

gradient=[sum(sum(gradient1)),sum(sum(gradient2))];

end
