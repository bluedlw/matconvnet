classdef YoloV2 < dagnn.ElementWise
    
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
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            train = 1;
            derInputs{1} = vl_yolov2(inputs{1}, inputs{2}, obj.anchors, obj.bias_match, ...
                obj.classes, obj.coord, obj.softmax, obj.jitter, obj.rescore, ...
                obj.object_scale, obj.noobject_scale, obj.class_scale, obj.coord_scale,...
                obj.absolute, obj.thresh, obj.random, train, obj.iters);
            derInputs{2} = [];
            derParams = {};
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
            outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
        end
        
        
        function obj = YoloV2(varargin)
            obj.load(varargin);
        end
    end
    
    
end