function [y, normalizer] = vl_nnfocalloss(x, c, dzdy, varargin)

opts.alpha = 0.25;
opts.gamma = 2;
opts.lossType = 'sigmoid';

opts = vl_argparse(opts, varargin);

%c = gather(c);

if isa(x, 'gpuArray')
  switch classUnderlying(x) ;
    case 'single', cast = @(z) single(z) ;
    case 'double', cast = @(z) double(z) ;
  end
else
  switch class(x)
    case 'single', cast = @(z) single(z) ;
    case 'double', cast = @(z) double(z) ;
  end
end

ignoreMask = (c==0);
instanceMask = ~ignoreMask;
if strcmp(opts.lossType, 'sigmoid')
    bgLabel = -1;
    instanceWeight = zerosLike(c);
    bgMask = (c == bgLabel);
    fgMask = (~bgMask & instanceMask);
    instanceWeight(bgMask) = 1-opts.alpha;
    instanceWeight(fgMask) = opts.alpha;
    
elseif strcmp(opts.lossType, 'softmaxlog')
    bgLabel = 1;
    instanceWeight = cast(~ignoreMask);
else
    error('unsupported loss type for focal loss');
end


%normalizer = sum(fgMask(:)) / size(x, 4);
normalizer = sum(fgMask(:));
%normalizer = sum(tmp(:)' * mask(:)) / size(x, 4); % compatible with gradient update
if opts.gamma == 0
    %normalizer = normalizer + sum(bgMask(:)) / size(x,4);
    normalizer = normalizer + sum(bgMask(:));
end

if normalizer == 0
    normalizer = 1;
end


%fprintf('%f\n', normalizer);

if isempty(dzdy)
    if strcmp(opts.lossType, 'sigmoid')
        xt = x .* c;
        pt = sigmoid(xt);
        y = -(1-pt).^opts.gamma .* log(pt);
        y = sum(y(:)' * instanceWeight(:)) / normalizer;
    else if strcmp(opts.lossType, 'softmaxlog')
        ;
    end
    
else
    dzdy = dzdy * instanceWeight;
    if strcmp(opts.lossType, 'sigmoid')
        xt = x .* c;
        pt = sigmoid(xt);
        y = c .* (1-pt).^opts.gamma .* (opts.gamma * pt .* log(pt) + pt - 1);
        y = bsxfun(@times, y, dzdy);
        y = y / normalizer;
    else if strcmp(opts.lossType, 'softmaxlog')
        ;
    end
    
end


function y = sigmoid(x)
y = 1 ./ (1 + exp(-x));

%function normalizer = GetNormalizer(c)

% --------------------------------------------------------------------
function y = zerosLike(x)
% --------------------------------------------------------------------
if isa(x,'gpuArray')
  y = gpuArray.zeros(size(x),classUnderlying(x)) ;
else
  y = zeros(size(x),'like',x) ;
end
