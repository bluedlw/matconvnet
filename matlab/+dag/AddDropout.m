function net = AddDropout(net, layerName, inputs, outputs, varargin)

opts.rate = 0.5;

opts = vl_argparse(opts, varargin);

dropBlock = dagnn.DropOut('rate', opts.rate);

net.addLayer(layerName, dropBlock, inputs, outputs, {});
