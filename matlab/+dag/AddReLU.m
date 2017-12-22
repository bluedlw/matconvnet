function net = AddReLU(net, layerName, inputs, outputs, varargin)

reluBlock = dagnn.ReLU();

net.addLayer(layerName, reluBlock, inputs, outputs, {});
