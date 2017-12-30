classdef Reshape < dagnn.ElementWise
    
    properties
        n = 4 % divide channels into n groups
    end
    
    methods
        function outputs = forward(obj, inputs, params)
          outputs{1} = vl_nnreshape(inputs{1}, obj.n, 0);
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
          derInputs{1} = vl_nnreshape(derOutputs{1}, obj.n, 1);
          derParams = [];
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes)
            %outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
            %outputSizes{1}(3) = obj.size(4) ;
            outputSizes{1} = inputSizes{1} .* [1, obj.n, 1/obj.n, 1];
        end

        
        function obj = Reshape(varargin)
            obj.load(varargin);
        end
    end
    
end