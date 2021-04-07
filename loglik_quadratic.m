function result = loglik_quadratic(X, inv_Sigma) 

% help calculate the joint density of X with inverse covariance matrix inv_Sigma
% rows of X are independently distributed as N(0, inv_Sigma^-1)
% rewrite the formula in a matrix version to avoid
% using for loop

% X: n x p, data
% inv_Sigma: p x p
% return: \sum_{i = 1}^n x_i inv_Sigma x_i', x_i is a row of X, inv_Sigma is inverse covariance matrix

    n = size(X, 1);
    p = size(X, 2);
    result = reshape(inv_Sigma' * X', 1, n*p) * reshape(X', n*p, 1);
end

