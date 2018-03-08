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
            x = inputs{1};
            inputSize = [size(x, 1) size(x, 2) size(x, 3) size(x, 4)] ;
            labelSize = [size(x, 1), size(x, 2), 1, size(x, 4)];
            
            label = inputs{2};
            %hardLabel = label;
            assert(inputSize(1) == size(label, 1) && inputSize(2) == size(label, 2));
            
            if(obj.iter < obj.warmup)
                outputs{1} = label;
                return;
            end
            
            numPixelsPerImage = prod(inputSize(1:2));
            numPixels = numPixelsPerImage * inputSize(4);
            imageVolume = numPixelsPerImage * inputSize(3);
            n = reshape(0:numPixels-1,labelSize);
            offset = 1 + mod(n, numPixelsPerImage) + ...
                imageVolume * fix(n / numPixelsPerImage);
            ci = offset + numPixelsPerImage * max(c - 1,0);
            
            bglabel = -1;
            order = 'descend';
            if size(x, 3) == 1
                
            else
                %order = 'ascend';
                bglabel = 1;
                Xmax = max(x,[],3);
                ex = exp(bsxfun(@minus, x, Xmax));
                loss = Xmax + log(sum(ex,3)) - x(ci);
            end

            instanceMask = (label ~= 0);
            %bgMask = instanceMask & (label == bglabel);
            fgMask = instanceMask & (label ~= bglabel);

            [~, idx] = sort(loss(:), order);

            fgNum = sum(fgMask(:));
            if fgNum == 0
                fgNum = obj.num * size(x, 4);
            end
            %bgMask = bgMask(idx);
            %idx = idx(bgMask);
            %idx = idx(fgNum * obj.ratio + 1 : end);
            instanceMask = instanceMask(idx);
            idx = idx(instanceMask);
            idx = idx(fgNum * obj.ratio + 1 : end);
            label(idx) = 0;
            outputs{1} = label;
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
