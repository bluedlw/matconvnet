function net = AddConv(net, layerName, inputs, outputs, varargin)
opts.size = [1,1,1,1];
opts.pad = [0,0,0,0];
opts.stride = [1,1];
opts.dilate = [1,1];
opts.hasBias = true;
opts.weightDecay = 1;
opts.biasDecay = 0;
opts.weightLr = 1;
opts.biasLr = 2;
opts.params = {[layerName, '_weight'], [layerName, '_bias']};
opts.opts = {'cuDNN'};

opts = vl_argparse(opts, varargin);

if ~opts.hasBias
    if numel(opts.params) == 2
        opts.params(2) = [];
    end
end

convBlock = dagnn.Conv('size', opts.size, 'pad', opts.pad, ...
    'stride', opts.stride, 'dilate', opts.dilate, 'hasBias', ...
    opts.hasBias, 'opts', opts.opts);

net.addLayer(layerName, convBlock, inputs, outputs, opts.params);

idx = net.getParamIndex(opts.params{1});
net.params(idx).learningRate = opts.weightLr;
net.params(idx).weightDecay = opts.weightDecay;
if numel(opts.params) == 2
    idx = net.getParamIndex(opts.params{2});
    net.params(idx).learningRate = opts.biasLr;
    net.params(idx).weightDecay = opts.biasDecay;
end
