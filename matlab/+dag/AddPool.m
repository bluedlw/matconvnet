function net = AddPool(net, layerName, inputs, outputs, varargin)
opts.poolSize = [1,1];
opts.pad = [0,0];
opts.stride = [1,1];
opts.dilate = [1,1];
opts.method = 'max';

opts = vl_argparse(opts, varargin);

maxpoolBlock = dagnn.Pooling('poolSize', opts.poolSize, 'pad', opts.pad, ...
    'stride', opts.stride, 'method', opts.method, 'dilate', opts.dilate);

net.addLayer(layerName, maxpoolBlock, inputs, outputs, {});