classdef FocalLoss < dagnn.Loss
    
   properties
       alpha = 0.25
       gamma = 2
       instanceWeights = 1.
       instanceMask = []
       warmup = 100
       normalizer = 1
       iter = 0
   end
   
   properties(Transient)
       
   end
   
   methods
       function outputs = forward(obj, inputs, params)
           obj.iter = obj.iter + 1;
           S = 1;
           if obj.iter < obj.warmup
                S = 0.1;
           end
           gamma_ = obj.gamma;
           [outputs{1}, normalizer_] = vl_nnfocalloss(inputs{1}, inputs{2}, [], ...
               'alpha', obj.alpha, 'gamma', gamma_, 'loss', obj.loss);
           outputs{1} = outputs{1} * obj.instanceWeights * S;
           
           obj.normalizer = normalizer_;
           
           obj.accumulateAverage(inputs, outputs);
       end
       
       function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
           S = 1;
           if obj.iter < obj.warmup
               S = 0.1;
           end
           gamma_ = obj.gamma;
           [derInputs{1}, normalizer_]= vl_nnfocalloss(inputs{1}, inputs{2}, derOutputs{1}, ...
               'alpha', obj.alpha, 'gamma', gamma_, 'loss', obj.loss);
           derInputs{1} = derInputs{1} * obj.instanceWeights * S;
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
           %obj.loss = 'focalloss';
       end
   end
    
    
end