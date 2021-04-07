function [loglik_w, logdet_Sigma_w] = loglik_K_j_new(X, Z, ii, w, inv_Sigma, logdet_Sigma, sigmasq, sigmasq_a) 

%help calculate the ratio between the current probability of 
%X given Z (current means Z is updated) and the original probability of X given Z
    
    p = size(Z, 1);
    n = size(X, 1);
    E = zeros(p, p);
    E(ii, ii) = 1;
    
    % Z_new = [Z A], where A is p x w matrix with only ii-th row being all 1
    % Sigma_new = sigmasq_a * Z_new Z_new' + sigmasq I
    % Sigma = sigmasq_a * Z Z' + sigmasq I
    % inv_Sigma_diff = Sigma_new^{-1} - Sigma^{-1}
    % derived based on the Woodbury inverse lemma
    
    inv_Sigma_diff = - inv_Sigma(:, ii) * inv_Sigma(ii, :) / (1 / (sigmasq_a * w) + inv_Sigma(ii, ii)); 
    
    logdet_Sigma_w = log(det(sigmasq_a * Z * Z' + sigmasq_a * w * E + sigmasq * eye(p)));
    % logdet_Sigma = log(det(sigmasq_a * Z * Z' + sigmasq * eye(p)));
    
    %if w == 5
    %    fprintf('logdet_Sigma: %.2f.\n', logdet_Sigma);
    %    fprintf('logdet_Sigma_w: %.2f.\n', logdet_Sigma_w);
    %end
    
    % log p(X | Z_new) - log p(X | Z)
    loglik_w = -(1./2) * loglik_quadratic(X, inv_Sigma_diff) - (n/2) * (logdet_Sigma_w - logdet_Sigma);
    
    % p(X | Z_new) / p(X | Z)
    % y = exp(y);
    
end
