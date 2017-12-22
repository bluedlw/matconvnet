function net = AddBatchNorm(net, layerName, inputs, outputs, varargin)
opts.epsilon = 1e-5;
opts.numChannels = 1;
opts.params = {[layerName, '_gamma'], [layerName, '_beta'], [layerName, '_moments']};
opts.weightLr = 1;
opts.biasLr = 1;
opts.weightDecay = 1;
opts.biasDecay = 1;
opts.momentLr = 0.1;
opts.momentDecay = 1;

opts = vl_argparse(opts, varargin);


bnormBlock = dagnn.BatchNorm('numChannels', opts.numChannels, ...
    'epsilon', opts.epsilon);

net.addLayer(layerName, bnormBlock, inputs, outputs, opts.params);

idx = net.getParamIndex(opts.params{1});
net.params(idx).learningRate = opts.weightLr;
net.params(idx).weightDecay = opts.weightDecay;

idx = net.getParamIndex(opts.params{2});
net.params(idx).learningRate = opts.biasLr;
net.params(idx).weightDecay = opts.biasDecay;

idx = net.getParamIndex(opts.params{3});
net.params(idx).learningRate = opts.momentLr;
net.params(idx).weightDecay = opts.momentDecay;


% if numel(opts.params) == 2
%     idx = net.getParamIndex(opts.params{1});
%     net.params(idx).learningRate = opts.biasLr;
%     net.params(idx).weightDecay = opts.biasDecay;
% end