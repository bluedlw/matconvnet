classdef HardMine < dagnn.ElementWise
    
    properties
        num = 5
        ratio = 6
        bgLabel = -1
        warmup = 1000
        iter = 0
    end
    
    properties(Transient)
        
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            obj.iter = obj.iter + 1;
            % inputs{1} class prob, inputs{2} mask/weight, e.g. bg_mask
            bglabel = -1;
            prob = inputs{1};
            if size(inputs{1}, 3) == 1
                %prob = inputs{1}; % sigmoid
                order = 'descend';
                %bgLabel = -1;
            else
                %prob = inputs{1};
                [xmax, xid] = max(prob, [], 3);
                prob = bsxfun(@minus, prob, xmax);
                prob = exp(prob);
                bglabel = 1;
                prob = prob(:,:,1,:) ./ sum(prob, 3);
                order = 'ascend';
            end
             
            
            label = inputs{2};
            assert(size(prob, 1) == size(label, 1) && size(prob, 2) == size(label, 2));
            instanceMask = (label ~= 0);
            bgMask = instanceMask & (label == bglabel);
            fgMask = instanceMask & (label ~= bglabel);
            %wh = size(prob, 1) * size(prob, 2);
            hardLabel = label;
            if(obj.iter < obj.warmup)
                outputs{1} = hardLabel;
                return;
            end
            
%             for n = 1 : size(prob, 4)
%                 tmp = prob(:,:,:,n);
%                 [~, idx] = sort(tmp(:), 'descend');
%                 tmp2 = mask(:,:,:,n);
%                 tmp2_ = ~tmp2;
%                 idx = idx(tmp2(:));
%                 if sum(tmp2_(:)) == 0
%                     baseNum = obj.num;
%                 else
%                     baseNum = sum(tmp2_(:));
%                 end
%                 idx = idx(obj.ratio.^2 * baseNum + 1 : end);
%                 if isempty(idx)
%                     continue;
%                 else
%                     newW((n-1)*wh+idx) = 0;
%                 end
%             end

            [~, idx] = sort(prob(:), order);
                
            fgNum = sum(fgMask(:));
            if fgNum == 0
                fgNum = obj.num * size(prob, 4);
            end
            bgMask = bgMask(idx);
            idx = idx(bgMask);
            idx = idx(fgNum * obj.ratio + 1 : end);
            hardLabel(idx) = 0;
            outputs{1} = hardLabel;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutptus)
            %derInputs{1} = [];
            %derInputs{2} = [];
            %derParams = {};
        end
        
        function rfs = getReceptiveFields(obj)
            rfs(1,1).size = [1 1] ;
            rfs(1,1).stride = [1 1] ;
            rfs(1,1).offset = [1 1] ;
            rfs(2,1) = rfs(1,1);
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes)
            %outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
            %outputSizes{1}(3) = obj.size(4) ;
            outputSizes{1} = inputSizes{1};
        end
        
        function obj = HardMine(varargin)
            obj.load(varargin) ;
        end
    end
    
end
