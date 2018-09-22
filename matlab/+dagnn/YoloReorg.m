classdef YoloReorg < dagnn.ElementWise
    
    properties
        stride = [1, 1] %strideH, strideW
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            x = inputs{1};
            h = size(x,1); w = size(x, 2); 
            chn = size(x,3); n = size(x, 4);
            remH = mod(h, obj.stride(1));
            remW = mod(w, obj.stride(2));

            h_ = floor(h / obj.stride(1)); w_ = floor(w / obj.stride(2));
            chn_ = chn * (obj.stride(1)*obj.stride(2));
            hIdx = [ones(1, h_)*obj.stride(1), ones(1, remH)];
            wIdx = [ones(1, w_)*obj.stride(2), ones(1, remW)];
            
            if n > 1
                xcell = mat2cell(x, hIdx, wIdx, chn, n);
            else
                xcell = mat2cell(x, hIdx, wIdx, chn);
            end
            if remH > 0
                xcell = xcell(1:end-1,:);
            end
            if remW > 0
                xcell = xcell(:, 1:end-1);
            end
            
            b = cellfun(@(z) reshape(permute(z, [3 2 1 4]), 1, 1, chn_, n), ...
                xcell, 'un', false);
            
            y = cat(1, b{:, 1});
            for i = 2 : w_
                y = cat(2, y, cat(1, b{:, i}));
            end
            
            %outputs{1} = cell2mat(b);
            outputs{1} = y;
        end
    
   
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            der = derOutputs{1};
            h = size(der,1);   w = size(der, 2); 
            chn = size(der,3); n = size(der, 4);
            
            chn_ = chn / (obj.stride(1)*obj.stride(2));
            
            hIdx = ones(1, h);
            wIdx = ones(1, w);
            
            if n > 1
                derCell = mat2cell(der, hIdx, wIdx, chn, n);
            else
                derCell = mat2cell(der, hIdx, wIdx, chn);
            end
            
            b = cellfun(@(z) permute(reshape(z, chn_, obj.stride(2), obj.stride(1),...
                n), [3, 2, 1, 4]), derCell, 'un', false);
            
            y = cat(1, b{:, 1});
            for i = 2 : w
                y = cat(2, y, cat(1, b{:, i}));
            end
            
            derInputs{1} = inputs{1};
            derInputs{1}(:) = 0;
            %derInputs{1}(1:h*obj.stride(1), 1:w*obj.stride(2),:,:) = cell2mat(b);
            derInputs{1}(1:h*obj.stride(1), 1:w*obj.stride(2),:,:) = y;
            
            derParams = {};
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
            
            outputSizes{1} = [floor(inputSizes{1}(1)/obj.stride(1)), ...
                floor(inputSizes{1}(2) / obj.stride(2)), ...
                inputSizes{1}(3) * obj.stride(1)*obj.stride(2), inputSizes{1}(4)] ;
            
        end
        
        function rfs = getReceptiveFields(obj)
            rfs(1,1).size = obj.stride ;
            rfs(1,1).stride = obj.stride ;
            rfs(1,1).offset = (obj.stride + 1) / 2 ;
        end
        
        function obj = YoloReorg(varargin)
            obj.load(varargin);
            if numel(obj.stride) == 1
                obj.stride = [obj.stride, obj.stride];
            end
        end
    end  
end