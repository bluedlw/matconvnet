function net = AddSum(net, lname, inputs, outputs)

sumLayer = dagnn.Sum();

net.addLayer(lname, sumLayer, inputs, outputs, {});