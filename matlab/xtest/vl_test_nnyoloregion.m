function vl_test_nnyoloregion()

iters = 0;
anchors=[1,1; 2,2];
bias_match=1;
classes=3;
coord=4;
softmax=1;
jitter=0.3;
rescore=1;
object_scale=5;
noobject_scale=1;
class_scale=1;
coord_scale=1;
absolute=1;
thresh=0.6;
random=1;

train = 1;

gts = {[0.1, 0.1, 0.2, 0.2, 0], [0.3, 0.3, 0.45, 0.5, 1]};

batchSize = numel(gts);

featH = 3; featW = 5; featC = size(anchors, 1) * (coord + 1 + classes);

data = single(rand(featH, featW, featC, batchSize));

data_gpu = gpuArray(data);

der = vl_nnyoloregion(data_gpu, gts, anchors, bias_match, classes, coord, softmax, jitter, rescore,...
    object_scale, noobject_scale, class_scale, coord_scale, absolute, thresh, random, train, iters);
asdf = 100;
debug = 0;
