classdef FocalLoss < dagnn.Loss
    
   properties
       lossType = 'sigmoid'
       alpha = 0.25
       gamma = 2
       instanceWeight = 1.
       instanceMask = []
       
       warmup = 1000
       
       normalizer = 1
       iter = 0
   end
   
   properties(Transient)
       
   end
   
   methods
       function outputs = forward(obj, inputs, params)
           obj.iter = obj.iter + 1;
           if obj.iter < obj.warmup
               gamma_ = 0;
           else
               gamma_ = obj.gamma;
           end
           [outputs{1}, normalizer_] = vl_nnfocalloss(inputs{1}, inputs{2}, [], ...
               'alpha', obj.alpha, 'gamma', gamma_, 'lossType', obj.lossType);
           outputs{1} = outputs{1} * obj.instanceWeight;
           
           obj.normalizer = normalizer_;
           
           obj.accumulateAverage(inputs, outputs);
       end
       
       function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
           [derInputs{1}, normalizer_]= vl_nnfocalloss(inputs{1}, inputs{2}, derOutputs{1}, ...
               'alpha', obj.alpha, 'gamma', obj.gamma, 'lossType', obj.lossType);
           derInputs{1} = derInputs{1} * obj.instanceWeight;
           derInputs{2} = [];
           derParams = {};
       end
       
       function accumulateAverage(obj, inputs, outputs)
           if obj.ignoreAverage, return; end;
           n = obj.numAveraged;
           %m = n + gather(obj.normalizer) * size(inputs{1}, 4); %
           m = n + gather(obj.normalizer);
           obj.average = (n * obj.average +  gather(outputs{1}) * gather(obj.normalizer)) / m;
           obj.numAveraged = m;
       end
       
       function rfs = getReceptiveFields(obj)
           % the receptive field depends on the dimension of the variables
           % which is not known until the network is run
           rfs(1,1).size = [NaN NaN];
           rfs(1,1).stride = [NaN NaN];
           rfs(1,1).offset = [NaN NaN];
           rfs(2,1) = rfs(1,1) ;
           rfs(3,1) = rfs(1,1);
       end
       
       function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
           outputSizes{1} = [1 1 1 inputSizes{1}(4)];
       end
       
       function obj = FocalLoss(varargin)
           obj.load(varargin);
           obj.loss = 'focalloss';
       end
   end
    
    
end