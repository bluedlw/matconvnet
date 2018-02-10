function net = AddReLU(net, layerName, inputs, outputs, varargin)

opts.leak = 0;

opts = vl_argparse(opts, varargin);

reluBlock = dagnn.ReLU('leak', opts.leak);

net.addLayer(layerName, reluBlock, inputs, outputs, {});
