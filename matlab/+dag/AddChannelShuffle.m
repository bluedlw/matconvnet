function net = AddChannelShuffle(net, lname, inputs, outputs, varargin)
opts.groups = 1;

opts = vl_argparse(opts, varargin);

shuffleBlock = dagnn.ChannelShuffle('groups', opts.groups);
net.addLayer(lname, shuffleBlock, inputs, outputs, {});