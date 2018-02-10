function [y, normalizer] = vl_nnfocalloss(x, c, varargin)

if ~isempty(varargin) && ~ischar(varargin{1})  % passed in dzdy
  dzdy = varargin{1} ;
  varargin(1) = [] ;
else
  dzdy = [] ;
end

opts.instanceWeights = [];
opts.alpha = 0.25;
opts.gamma = 2;
opts.loss = 'softmaxlog'; % softmax

opts = vl_argparse(opts, varargin, 'nonrecursive') ;

inputSize = [size(x,1) size(x,2) size(x,3) size(x,4)] ;

% Form 1: C has one label per image. In this case, get C in form 2 or
% form 3.
c = gather(c) ;
if numel(c) == inputSize(4)
  c = reshape(c, [1 1 1 inputSize(4)]) ;
  c = repmat(c, inputSize(1:2)) ;
end


hasIgnoreLabel = any(c(:) == 0);

% --------------------------------------------------------------------
% Spatial weighting
% --------------------------------------------------------------------

% work around a bug in MATLAB, where native cast() would slow
% progressively
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

labelSize = [size(c,1) size(c,2) size(c,3) size(c,4)] ;
assert(isequal(labelSize(1:2), inputSize(1:2))) ;
assert(labelSize(4) == inputSize(4)) ;

%instanceWeights = [] ;
bgLabel = cast(1);
instanceMask = c(:,:,1, :) ~= 0;
bgMask = (c(:,:,1,:) == bgLabel) & instanceMask;
fgMask = (c(:,:,1,:) ~= bgLabel) & instanceMask;
instanceWeights = cast(instanceMask) ; % always have instanceWeights
%instanceWeights(bgMask) = cast(1 - opts.alpha);
%instanceWeights(fgMask) = cast(opts.alpha);


switch lower(opts.loss)
  case {'softmaxlog'}
    % there must be one categorical label per prediction vector
    assert(labelSize(3) == 1) ;

    if hasIgnoreLabel
      % null labels denote instances that should be skipped
      % dlw
    end
    instanceWeights(bgMask) = cast(1 - opts.alpha);
    instanceWeights(fgMask) = cast(opts.alpha);
  case {'sigmoid'}
    instanceWeights = repmat(instanceWeights, 1, 1, inputSize(3)) .* (1-opts.alpha);
  otherwise
    error('Unknown loss ''%s''.', opts.loss) ;
end

if ~isempty(opts.instanceWeights)
  % important: this code needs to broadcast opts.instanceWeights to
  % an array of the same size as c
  if isempty(instanceWeights)
    instanceWeights = bsxfun(@times, onesLike(c), opts.instanceWeights) ;
  else
    instanceWeights = bsxfun(@times, instanceWeights, opts.instanceWeights);
  end
end


fgNum = cast(sum(fgMask(:)));
if fgNum < 1
    fgNum = 1;
end

normalizer = fgNum;
% normalizer = inputSize(1) * inputSize(2) * inputSize(4);
% if ~isempty(instanceWeights)
%     normalizer = sum(instanceWeights(:) ~= 0);
% end

% --------------------------------------------------------------------
% Do the work
% --------------------------------------------------------------------

switch lower(opts.loss)
  case {'sigmoid', 'softmaxlog'}
    % from category labels to indexes
    numPixelsPerImage = prod(inputSize(1:2)) ;
    numPixels = numPixelsPerImage * inputSize(4) ;
    imageVolume = numPixelsPerImage * inputSize(3) ;

    n = reshape(0:numPixels-1,labelSize) ;
    offset = 1 + mod(n, numPixelsPerImage) + ...
             imageVolume * fix(n / numPixelsPerImage) ;
    ci = offset + numPixelsPerImage * max(c - 1,0) ;
end

switch lower(opts.loss)
    case {'sigmoid'}
      instanceWeights(ci) = instanceWeights(ci) .* (opts.alpha / (1-opts.alpha)); % check alpha (0.1, 0.9)
end

if nargin <= 2 || isempty(dzdy)
  switch lower(opts.loss)
    case 'classerror'
      [~,chat] = max(x,[],3) ;
      t = cast(c ~= chat) ;
    case 'topkerror'
      [~,predictions] = sort(x,3,'descend');
      t = 1 - sum(bsxfun(@eq, c, predictions(:,:,1:opts.topK,:)), 3);

    case 'softmaxlog'
      Xmax = max(x,[],3) ;
      ex = exp(bsxfun(@minus, x, Xmax));
      p = bsxfun(@rdivide, ex, sum(ex, 3));
      p = p(ci);
      t = -(1 - p).^opts.gamma .* log(p);
      %t = bsxfun(@times, -(1 - p).^opts.gamma, log(p));
      %t = Xmax + log(sum(ex,3)) - x(ci);
    case 'sigmoid'
      p = 1 ./ (1 + exp(-x));
      %tmp = -onesLike(x);
      %tmp(ci) = 1;
      %x_ = x .* tmp;
      %b = max(0, x_) ;
      %xp = x > 0;
      %t = -p.^opts.gamma .* (xp .* (log(p) - x) - ~xp .* log(1+exp(x)));
      %t(ci) = -(1 - p(ci)).^opts.gamma .* log(p(ci)); % TODO
      t = 1 - p;
      t(ci) = p(ci);
      t = -(1 - t).^opts.gamma .* log(max(t, 1e-10));
  end
  if ~isempty(instanceWeights)
    y = instanceWeights(:)' * t(:) ;
  else
    y = sum(t(:));
  end
  if normalizer > 0
      y = y / normalizer;
  end
else
  if ~isempty(instanceWeights)
    dzdy = dzdy * instanceWeights ;
  end
  
  if normalizer > 0
      dzdy = dzdy / normalizer;
  end
  
  switch lower(opts.loss)
    case {'classerror', 'topkerror'}
      y = zerosLike(x) ;
    case 'softmaxlog'
      Xmax = max(x,[],3) ;
      ex = exp(bsxfun(@minus, x, Xmax)) ;
      y = bsxfun(@rdivide, ex, sum(ex,3)) ;
      
      one_p = 1 - y(ci);
      tmp = -(one_p.^opts.gamma) + opts.gamma * one_p.^(opts.gamma-1) .* y(ci) .* log(y(ci));
      
      y(ci) = y(ci) - 1;
      
      y = bsxfun(@times, tmp, -y);
      y = bsxfun(@times, dzdy, y) ;
    case 'sigmoid'
      %x_ = -x;
      %x_(ci) = -x_(ci);
      p = 1 ./ (1 + exp(-x));
      %y = -p.^opts.gamma .* (-p + opts.gamma .* (1-p) .*log(1-p));
      %y(ci) = -(1 - p(ci)).^opts.gamma .* (1 - p(ci)-opts.gamma .* p(ci) .* log(p(ci)));
      %y = bsxfun(@times, dzdy, y);
      y = 1 - p;
      y(ci) = p(ci);
      y = (1 - y).^opts.gamma .* (1 - y - opts.gamma .* y .* log(max(y,1e-10)));
      
      y(ci) = -y(ci);
      y = bsxfun(@times, dzdy, y);
  end
  
end

% --------------------------------------------------------------------
function y = zerosLike(x)
% --------------------------------------------------------------------
if isa(x,'gpuArray')
  y = gpuArray.zeros(size(x),classUnderlying(x)) ;
else
  y = zeros(size(x),'like',x) ;
end

% --------------------------------------------------------------------
function y = onesLike(x)
% --------------------------------------------------------------------
if isa(x,'gpuArray')
  y = gpuArray.ones(size(x),classUnderlying(x)) ;
else
  y = ones(size(x),'like',x) ;
end

% %c = gather(c);
% bgLabel = -1;
% if size(x, 3) == 1
%     lossType = 'sigmoid'; % binary
% else
%     lossType = 'softmax';
%     bgLabel = 1;          % set fixed to 1, foreground as 2,3,4....
% end
% 
% if isa(x, 'gpuArray')
%   switch classUnderlying(x) ;
%     case 'single', cast = @(z) single(z) ;
%     case 'double', cast = @(z) double(z) ;
%   end
% else
%   switch class(x)
%     case 'single', cast = @(z) single(z) ;
%     case 'double', cast = @(z) double(z) ;
%   end
% end
% 
% ignoreMask = (c==0);
% instanceMask = ~ignoreMask;
% if strcmp(opts.lossType, 'sigmoid')
%     bgLabel = -1;
%     instanceWeight = zerosLike(c);
%     bgMask = (c == bgLabel);
%     fgMask = (~bgMask & instanceMask);
%     instanceWeight(bgMask) = 1 - opts.alpha;
%     instanceWeight(fgMask) = opts.alpha;
%     
% elseif strcmp(opts.lossType, 'softmaxlog')
%     bgLabel = 1;
%     instanceWeight = cast(~ignoreMask);
% else
%     error('unsupported loss type for focal loss');
% end
% 
% 
% %normalizer = sum(fgMask(:)) / size(x, 4);
% normalizer = sum(fgMask(:));
% %normalizer = sum(tmp(:)' * mask(:)) / size(x, 4); % compatible with gradient update
% if opts.gamma == 0
%     %normalizer = normalizer + sum(bgMask(:)) / size(x,4);
%     normalizer = normalizer + sum(bgMask(:));
% end
% 
% if normalizer == 0
%     normalizer = 1;
% end
% 
% 
% %fprintf('%f\n', normalizer);
% 
% if isempty(dzdy)
%     if strcmp(opts.lossType, 'sigmoid')
%         xt = x .* c;
%         pt = sigmoid(xt);
%         y = -(1-pt).^opts.gamma .* log(pt);
%         y = sum(y(:)' * instanceWeight(:)) / normalizer;
%     elseif strcmp(opts.lossType, 'softmaxlog')
%         ;
%     end
%     
% else
%     dzdy = dzdy * instanceWeight;
%     if strcmp(opts.lossType, 'sigmoid')
%         xt = x .* c;
%         pt = sigmoid(xt);
%         y = c .* (1-pt).^opts.gamma .* (opts.gamma * pt .* log(pt) + pt - 1);
%         y = bsxfun(@times, y, dzdy);
%         y = y / normalizer;
%     elseif strcmp(opts.lossType, 'softmaxlog')
%         ;
%     end
%     
% end
% 
% % --------------------------------------------------------------------
% function y = sigmoid(x)
% y = 1 ./ (1 + exp(-x));
% 
% % --------------------------------------------------------------------
% function y = zerosLike(x)
% if isa(x,'gpuArray')
%   y = gpuArray.zeros(size(x),classUnderlying(x)) ;
% else
%   y = zeros(size(x),'like',x) ;
% end
