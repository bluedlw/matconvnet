function net = AddLoss(net, layerName, inputs, outputs, varargin)
opts.loss = 'softmaxlog'; % classerror, 
opts.instanceWeights = [];
opts.classWeights = [];
opts.threshold = 0;
opts.topK = 5;


opts = vl_argparse(opts, varargin, 'nonrecursive');

lossBlock = dagnn.Loss('loss', opts.loss, 'opts', ...
    {'instanceWeights', opts.instanceWeights, ...
     'classWeights', opts.classWeights,...
     'threshold', opts.threshold,...
     'topK', opts.topK});

net.addLayer(layerName, lossBlock, inputs, outputs, {});
