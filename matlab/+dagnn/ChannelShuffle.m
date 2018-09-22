classdef ChannelShuffle < dagnn.ElementWise
    
  %   ChannelShuffle layer
  %   Please refer to: ShuffleNet An Extremely Efficient Convolutional
  %   Neural Network for Mobile Devices

  properties
    groups = 1
  end

  methods
    function outputs = forward(obj, inputs, params)
      s = [size(inputs{1}, 1), size(inputs{1}, 2), size(inputs{1}, 3), ...
          size(inputs{1}, 4)];
      if obj.groups == 1
          outputs{1} = inputs{1};
          return;
      end
      if mod(s(3), obj.groups) ~= 0
          error('input channel should be divided by groups');
      end
      n = s(3) / obj.groups;
      tmp = reshape(inputs{1}, s(1), s(2), n, obj.groups, s(4));
      tmp = permute(tmp, [1,2,4,3,5]);
      outputs{1} = reshape(tmp, s(1), s(2), s(3), s(4));
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      s = [size(inputs{1}, 1), size(inputs{1}, 2), size(inputs{1}, 3), ...
          size(inputs{1}, 4)];
      if obj.groups == 1
          derInputs{1} = derOutputs{1};
          return;
      end
      if mod(s(3), obj.groups) ~= 0
          error('input channel should be divided by groups');
      end
      n = s(3) / obj.groups;
      tmp = reshape(derOutputs{1}, s(1), s(2), obj.groups, n, s(4));
      tmp = permute(tmp, [1,2,4,3,5]);
      derInputs{1} = reshape(tmp, s(1), s(2), s(3), s(4));
      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = inputSizes{1} ;
    end

    function obj = ChannelShuffle(varargin)
      obj.load(varargin) ;
    end
  end
    
end