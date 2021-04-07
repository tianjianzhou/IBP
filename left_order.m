function Z_lof = left_order(Z)
    cell_col = num2cell(Z, 1); 
    binary_col = cellfun(@(x) str2double(strrep(num2str(reshape(x, 1, [])), ' ', '')), cell_col);
    [~, index_col] = sort(binary_col, 'descend');
    Z_lof = Z(:, index_col);
end