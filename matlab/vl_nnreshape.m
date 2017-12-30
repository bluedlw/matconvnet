function y = vl_nnreshape(x, n, backWard)

inputSize = [size(x, 1), size(x, 2), size(x, 3), size(x, 4)];

if ~backWard
    
    tmp2 = [];
    m = inputSize(3) / n; %chn per group, for coord, n = 4;
    for num = 1 : inputSize(4)
        tmp = [];
        for c = 1 : n
            tmp = cat(2, tmp, x(:,:,(c-1)*m+1:c*m, num));
        end
        tmp2 = cat(4, tmp2, tmp);
    end
    y = tmp2;
else
    tmp2 = [];
    m = inputSize(2) / n;
    for num = 1 : inputSize(4)
        tmp = x(:,1:m,1); % cat(3, [], gpuArray(ones(3,3,4))), get 3x3x5!???
        for c = 1 : n
            tmp = cat(3, tmp, x(:, (c-1)*m+1:c*m, :, num));
        end
        tmp(:,:,1) = [];
        tmp2 = cat(4, tmp2, tmp);
    end
    y = tmp2;
end

