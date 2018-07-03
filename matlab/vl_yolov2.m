function y = vl_yolov2(x, gt, anchors, bias_match, classes, coord, softmax, jitter, rescore, ...
                object_scale, noobject_scale, class_scale, coord_scale,...
                absolute, thresh, random, train, iters)
useGpu = 0;
if isa(x, 'gpuArray')
    useGpu = 1;
end
x = gather(x);
gt = gather(gt);

[y, ~] = vl_yolov2mex(x, gt, anchors, bias_match, classes, coord, softmax, jitter, rescore, ...
                object_scale, noobject_scale, class_scale, coord_scale,...
                absolute, thresh, random, train, iters);          
if useGpu == 1
    y = gpuArray(y);
end