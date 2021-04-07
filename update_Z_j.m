function [Z_new, Sigma_new, inv_Sigma_new, logdet_Sigma_new] = update_Z_j(ii, X, Z, Sigma, inv_Sigma, logdet_Sigma, sigmasq, sigmasq_a, alpha, T)
% ii: row index
% T: temprature

    K = size(Z, 2);
    p = size(Z, 1);
    k1 = 0;
    
    K_active = K;
    
    deleted_column = false;
    
    for k = 1:K
        k1 = k1 + 1;
        
        % 1. delete the k-th column if blabla
        if sum(Z(:, k1)) - Z(ii, k1) == 0 
            
            %update Sigma, inv_Sigma, logdet_Sigma
            if Z(ii, k1) == 1
                Sigma(ii, ii) = Sigma(ii, ii) - sigmasq_a;
            end
            
            % remove the k1-th column
            Z(:, k1) = [];
            deleted_column = true;
            k1 = k1 - 1;
            K_active = K_active - 1;
            
            % seems no need to calculate inv_Sigma and logdet_Sigma now??
            % if the next k is still delete, then delete
            % without calculating inv_Sigma
            % inv_Sigma = inv_covmat(Z, sigmasq, sigmasq_a);
            
        % 2. otherwise, update z_{ii, k} = 1 or 0
        else 
            if deleted_column
                inv_Sigma = inv_covmat(Z, sigmasq, sigmasq_a);
                logdet_Sigma = log(det(Sigma));
                deleted_column = false;
            end
            
            % prepare for calculating p(z_{jk} = 1 | ..)
            if Z(ii, k1) == 0
                Sigma0 = Sigma;
                inv_Sigma0 = inv_Sigma;
                logdet_Sigma0 = logdet_Sigma;
                
                % if Z_{jk} changes from 0 to 1, how Z, Sigma and inv_Sigma will change
                Z1 = Z;
                Z1(ii, k1) = 1;
                
                Sigma1 = sigmasq_a * (Z1 * Z1') + sigmasq * eye(p);
                %Sigma1 = Sigma;
                %Sigma1(ii, :) = Sigma1(ii, :) + sigmasq_a * Z(:, k1)';
                %Sigma1(:, ii) = Sigma1(:, ii) + sigmasq_a * Z(:, k1);
                %Sigma1(ii, ii) = Sigma1(ii, ii) + sigmasq_a; 
                
                inv_Sigma1 = inv_covmat(Z1, sigmasq, sigmasq_a);
                logdet_Sigma1 = log(det(Sigma1));
                
            else
                Sigma1 = Sigma;
                inv_Sigma1 = inv_Sigma;
                logdet_Sigma1 = logdet_Sigma;
                
                % if Z_{jk} changes from 1 to 0, how Z, Sigma and inv_Sigma will change
                Z0 = Z;
                Z0(ii, k1) = 0;
                
                Sigma0 = sigmasq_a * (Z0 * Z0') + sigmasq * eye(p);
                %Sigma0 = Sigma;
                %Sigma0(ii, :) = Sigma0(ii, :) - sigmasq_a * Z(:, k1)'; 
                %Sigma0(:, ii) = Sigma0(:, ii) - sigmasq_a * Z(:, k1);
                %Sigma0(ii, ii) = Sigma0(ii, ii) + sigmasq_a; %update Sigma
                
                inv_Sigma0 = inv_covmat(Z0, sigmasq, sigmasq_a);
                logdet_Sigma0 = log(det(Sigma0));
                
            end
            
            %update z_{ii,k}, i.e. Z
            relement = binornd(1, prob_z_jk_1(X, Sigma0, Sigma1, inv_Sigma0, inv_Sigma1, logdet_Sigma0, logdet_Sigma1, Z, ii, k1, T)); 
            Z(ii, k1) = relement;
            
            %update Sigma, inv_Sigma
            if relement == 0
                Sigma = Sigma0;
                inv_Sigma = inv_Sigma0;
                logdet_Sigma = logdet_Sigma0;
            else
                Sigma = Sigma1;
                inv_Sigma = inv_Sigma1;
                logdet_Sigma = logdet_Sigma1;
            end
        end
    end
    
    if deleted_column
        inv_Sigma = inv_covmat(Z, sigmasq, sigmasq_a);
        logdet_Sigma = log(det(Sigma));
        deleted_column = false;
    end
    
    % only add new columns if current # of columns < 76 (truncate at K = 80..)
    if K_active < 76
    
        % 3. add new columns
        % generate the number of new features added to the iith row
        % find: locate nonzero elements
        % mnrnd: returns a vector e.g. (0, 0, 1, 0) if p = (x, x, x, x)
        % Kn: # of new columns to add to Z
        [prob_Kn_all, logdet_Sigma_Kn_all] = prob_K_j_new(X, Z, ii, inv_Sigma, logdet_Sigma, sigmasq, sigmasq_a, alpha, T);
    
        Kn = find(mnrnd(1, prob_Kn_all)) - 1; 
    
        if Kn > 0
            % update Z, adding Kn new features (columns) to the ii-th row
            E = zeros(p, Kn);
            E(ii, :) = 1;
            Z = [Z, E];
        
            %update Sigma
            Sigma(ii, ii) = Sigma(ii, ii) + sigmasq_a * Kn;
        
            % update inv_Sigma
            % still used that Woodbury trick in "loglik_K_j_new.m"
            inv_Sigma = inv_Sigma - inv_Sigma(:, ii) * inv_Sigma(ii, :) / (1 / (sigmasq_a * Kn) + inv_Sigma(ii, ii));
            logdet_Sigma = logdet_Sigma_Kn_all(Kn + 1);
        end
    end
    
    Z_new = Z;
    Sigma_new = Sigma;
    inv_Sigma_new = inv_Sigma;
    logdet_Sigma_new = logdet_Sigma;
end
