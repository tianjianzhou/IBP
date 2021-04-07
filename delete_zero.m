function Z1= delete_zero(Z)

    Z1 = Z(:, sum(Z, 1) ~= 0);

end

