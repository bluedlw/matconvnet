function net = AddConcat(net, layerName, inputs, outputs, varargin)
opts.dim = 3;

opts = vl_argparse(opts, varargin);

concatBlock = dagnn.Concat('dim', opts.dim);

net.addLayer(layerName, concatBlock, inputs, outputs, {});