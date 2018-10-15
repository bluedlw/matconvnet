classdef YoloRegion < dagnn.ElementWise
    
    properties
        iters = 0
        anchors=[1,1]
        bias_match=1
        classes=1
        coord=4
        softmax=1
        jitter=0.3
        rescore=1
        object_scale=5
        noobject_scale=1
        class_scale=1
        coord_scale=1
        absolute=1
        thresh=0.6
        random=1
    end
    
    
    methods
        function outputs = forward(obj, inputs, params)
            train = 1;
            if strcmp(obj.net.mode, 'test')
                train = 0;
            else
                obj.iters = obj.iters + 1;
            end
            outputs{1} = 1;
            
%             outputs{1} = vl_yolov2(inputs{1}, inputs{2}, obj.anchors, obj.bias_match, ...
%                 obj.classes, obj.coord, obj.softmax, obj.jitter, obj.rescore, ...
%                 obj.object_scale, obj.noobject_scale, obj.class_scale, obj.coord_scale,...
%                 obj.absolute, obj.thresh, obj.random, train, obj.iters);
            %outputs{1} = [];
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            train = 1;
            [derInputs{1}, avg_iou, avg_cls, avg_obj, avg_anyobj, recall] = ...
            vl_nnyoloregion(inputs{1}, inputs{2}, obj.anchors, obj.bias_match, ...
                obj.classes, obj.coord, obj.softmax, obj.jitter, obj.rescore, ...
                obj.object_scale, obj.noobject_scale, obj.class_scale, obj.coord_scale,...
                obj.absolute, obj.thresh, obj.random, train, obj.iters);
            derInputs{1} = derInputs{1} / size(derInputs{1},4);
            %fprintf('loss: %f\n', sum(derInputs{1}(:).^2)/(size(derInputs{1},1) * size(derInputs{1},2) * ...
            %    size(derInputs{1}, 4) * size(obj.anchors, 1)));
            %fprintf('  loss: %f ', sum(derInputs{1}(:).^2));

            featH = size(inputs{1}, 1);
            featW = size(inputs{1}, 2);
            featC = size(inputs{1}, 3);
            bs = size(inputs{1}, 4);
            
            numAnchors = size(obj.anchors, 1);

            numObjs = 0;
            for i = 1 : numel(inputs{2})
                gt = inputs{2}{i};
                numObjs = numObjs + size(gt, 1);
            end

            numObjs = max(numObjs, 1);

            numLocs = featH * featW * numAnchors * bs;

            avg_anyobj_ = sum(avg_anyobj(:)) / numLocs;

            avg_iou_ = sum(avg_iou(:)) / numObjs;

            avg_cls_ = sum(avg_cls(:)) / numObjs;

            avg_obj_ = sum(avg_obj(:)) / numObjs;

            recall_ = sum(recall(:)) / numObjs;

            fprintf(' Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d ',...
                    avg_iou_, avg_cls_, avg_obj_, avg_anyobj_, recall_, numObjs);

            derInputs{2} = [];
            derParams = {};
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
            outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
        end
        
        function rfs = getReceptiveFields(obj)
            numInputs = numel(obj.net.layers(obj.layerIndex).inputs) ;
            rfs = getReceptiveFields@dagnn.ElementWise(obj) ;
            rfs = repmat(rfs, numInputs, 1) ;
        end
        
        function obj = YoloRegion(varargin)
            obj.load(varargin);
        end
    end
    
    
end