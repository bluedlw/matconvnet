// @file nnnormalizelp_gpu.cu
// @brief Batch normalization block
// @author Andrea Vedaldi

/*
Copyright (C) 2017 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnyoloregion.hpp"
#include "datacu.hpp"
#include <vector>
#include <algorithm>

// -------------------------------------------------------------------
//                                                             Helpers
// -------------------------------------------------------------------
/*
struct GPUVisitPattern
{
  size_t normsVolume ;
  size_t inputVolume ;
  int dims [4] {1,1,1,1} ;
  int strides [4] {0,0,0,0} ;
  int ndims [4] {1,1,1,1} ;
  int nstrides [4] {0,0,0,0} ;
} ;
*/
/*
GPUVisitPattern getGPUVisitPatternForInput(NormalizeLp const & op, vl::Tensor input)
{
  // Compute tensor geometry.
  int n = input.getNumDimensions() ;
  auto inputDimensions = std::vector<size_t>(input.getDimensions(),
                                             input.getDimensions() + n) ;

  assert(n <= 4) ; // Todo: relax.

  size_t inputVolume = 1 ;
  size_t normsVolume = 1 ;
  auto dims = std::vector<ptrdiff_t>{} ;
  auto steps = std::vector<ptrdiff_t>{} ;
  auto ndims = std::vector<ptrdiff_t>{} ;
  auto nstrides = std::vector<ptrdiff_t>{} ;

  // Find out how to traverse the reduced results as the input is
  // scanned from first to last element.
  for (int d = 0 ; d < n ; ++d) {
    bool squashed =
    (find(op.selectedDimensions.begin(), op.selectedDimensions.end(), d) !=
     op.selectedDimensions.end()) ;

    if (squashed) {
      dims.push_back(inputDimensions[d]) ;
      steps.push_back(inputVolume) ;
    } else {
      ndims.push_back(inputDimensions[d]) ;
      nstrides.push_back(inputVolume) ;
      normsVolume *= inputDimensions[d] ;
    }
    inputVolume *= inputDimensions[d] ;
  }

  //cout << steps.size() << " " << inputVolume << endl ;
  
  for (int d = steps.size() ; d < 5 ; ++d) {
    steps.push_back(inputVolume) ;
    dims.push_back(1) ;
  }
  for (int d = 3 ; d >= 0 ; d--) {
    steps[d+1] -= steps[d] * dims[d] ;
  }

  GPUVisitPattern vp ;
  vp.inputVolume = inputVolume ;
  vp.normsVolume = normsVolume ;
  std::copy(dims.begin(),dims.end(),vp.dims) ;
  std::copy(steps.begin(),steps.end(),vp.strides) ;
  std::copy(ndims.begin(),ndims.end(),vp.ndims) ;
  std::copy(nstrides.begin(),nstrides.end(),vp.nstrides) ;
  return vp ;
}
*/


/*
template<typename type> __global__ void
computeDerInput(type * derInputData,
                type const * inputData,
                type const * normsData,
                type const * derOutputData,
                type const * scratchData,
                type exponent,
                GPUVisitPattern vp)
{
  int tid = threadIdx.x ;
  if (tid >= vp.normsVolume) { return ; }
  normsData += tid ;
  scratchData += tid ;

  int i0 = tid % vp.ndims[0] ; tid /= vp.ndims[0] ;
  int i1 = tid % vp.ndims[1] ; tid /= vp.ndims[1] ;
  int i2 = tid % vp.ndims[2] ; tid /= vp.ndims[2] ;
  int i3 = tid % vp.ndims[3] ;

  int offset =
  i0 * vp.nstrides[0] +
  i1 * vp.nstrides[1] +
  i2 * vp.nstrides[2] +
  i3 * vp.nstrides[3] ;

  derInputData += offset ;
  inputData += offset ;
  derOutputData += offset ;

  type const nv = *normsData ;
  type const sv = *scratchData ;

  for (int i3 = 0 ; i3 < vp.dims[3] ; ++i3) {
    for (int i2 = 0 ; i2 < vp.dims[2] ; ++i2) {
      for (int i1 = 0 ; i1 < vp.dims[1] ; ++i1) {
        for (int i0 = 0 ; i0 < vp.dims[0] ; ++i0) {
          type iv = *inputData ;
          type dov = *derOutputData ;

          *derInputData = dov / nv - sv * pow(iv,exponent-1) / pow(nv,exponent+1) ;

          derInputData += vp.strides[0] ;
          inputData += vp.strides[0] ;
          derOutputData += vp.strides[0] ;
        }
        derInputData += vp.strides[1] ;
        inputData += vp.strides[1] ;
        derOutputData += vp.strides[1] ;
      }
      derInputData += vp.strides[2] ;
      inputData += vp.strides[2] ;
      derOutputData += vp.strides[2] ;
    }
    derInputData += vp.strides[3] ;
    inputData += vp.strides[3] ;
    derOutputData += vp.strides[3] ;
  }
}
*/

/*
// -------------------------------------------------------------------
//                                                         GPU forward
// -------------------------------------------------------------------

template<vl::DataType dataType, bool givenNorms>
struct NormalizeLpForwardGPU
{
  vl::ErrorCode operator()(NormalizeLp & op,
                           Tensor &output,
                           typename NormAgrument<givenNorms>::type norms,
                           Tensor const &input)
  {
    assert(norms || !givenNorms) ;

    typedef typename vl::DataTypeTraits<dataType>::type type ;
    auto vp = getGPUVisitPatternForInput(op,input) ;

    // Get buffers.
    type const * inputData = (type const*)input.getMemory() ;
    type * normsData ;
    if (norms) {
      normsData = (type*)norms.getMemory() ;
    }
    else {
      normsData = (type*)op.context.getWorkspace
      (vl::VLDT_GPU, vp.normsVolume * sizeof(type)) ;
    }

    // Accumulate norms.
    if (!givenNorms) {
      computeNorms<type>
      <<< divideAndRoundUp(vp.normsVolume, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (normsData,inputData,op.exponent,op.epsilon,vp) ;
    }

    // Divide by them.
    type * outputData = (type*)output.getMemory() ;
    divideByNorms<type>
    <<< divideAndRoundUp(vp.normsVolume, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
    (outputData,inputData,normsData,vp) ;

    //cout << "n vol " << vp.normsVolume << endl ;
    return vl::VLE_Success ;
  }
} ;
*/

/*
template<vl::DataType dataType>
struct NormalizeLpForward<vl::VLDT_GPU, dataType>
: public NormalizeLpForwardGPU<dataType,false>
{ } ;

template<vl::DataType dataType>
struct NormalizeLpForwardWithNorms<vl::VLDT_GPU, dataType>
: public NormalizeLpForwardGPU<dataType,true>
{ } ;
*/

// -------------------------------------------------------------------
//                                                         GPU forward
// -------------------------------------------------------------------

template<vl::DataType dataType>
struct YoloRegionForwardGPU
{
  vl::ErrorCode operator()(YoloRegion & op,
                           Tensor &output,
                           Tensor const &input)
  {
    typedef typename vl::DataTypeTraits<dataType>::type type ;
/*
    int batchSize = input.getSize();
    int channel = input.getDepth();
    int featH = input.getHeight();
    int featW = input.getWidth();

    int numAnchors = op.numAnchors;

    // Get buffers.
    type const * inputData = (type const*)input.getMemory() ;

    int batchSize = input.getSize();
    // copy anchor to device
    type anchors = (type*)op.context.getWorkspace
    (vl::VLDT_GPU, YOLO_MAX_NUM_ANCHORS * 2 * sizeof(type)) ;
    cudaError_t err;
    err = cudaMemcpy(anchors, &op.anchors[0], op.anchors.size()*sizeof(type),
                     CudaMemcpyHostToDevice);

    // copy ground truths to device
    type anchors = (type*)op.context.getWorkspace
    (vl::VLDT_GPU, YOLO_MAX_NUM_GT_PER_IMAGE * batchSize * (op.coords + 1) * sizeof(type)) ;
*/
    

    /*
    type * normsData ;
    if (norms) {
      normsData = (type*)norms.getMemory() ;
    }
    else {
      normsData = (type*)op.context.getWorkspace
      (vl::VLDT_GPU, vp.normsVolume * sizeof(type)) ;
    }
    */

    return vl::VLE_Success ;
  }
} ;
/*
// -------------------------------------------------------------------
//                                                        GPU backward
// -------------------------------------------------------------------

template<vl::DataType dataType, bool givenNorms>
struct NormalizeLpBackwardGPU
{
  vl::ErrorCode operator()(NormalizeLp &op,
                           Tensor &derInput,
                           typename NormAgrument<givenNorms>::type norms,
                           Tensor const &input,
                           Tensor const& derOutput)
  {
    assert(norms || !givenNorms) ;

    typedef typename vl::DataTypeTraits<dataType>::type type ;
    auto vp = getGPUVisitPatternForInput(op,input) ;

    // Get buffers.
    size_t workspaceSize = vp.normsVolume * sizeof(type) ;
    type const * inputData = (type const*)input.getMemory() ;
    type * normsData ;
    if (norms) {
      normsData = (type*)norms.getMemory() ;
    }
    else {
      normsData = 0 ;
      workspaceSize *= 2 ;
    }
    type * scratchData = (type*)op.context.getWorkspace(vl::VLDT_GPU, workspaceSize) ;
    if (normsData == NULL) {
      normsData = scratchData + vp.normsVolume ;
    }

    // Accumulate norms.
    if (!givenNorms) {
      computeNorms<type>
      <<< divideAndRoundUp(vp.normsVolume, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (normsData,inputData,op.exponent,op.epsilon,vp) ;
    }

    // Compute sum(derOutput .* input).
    type const* derOutputData = (type const*)derOutput.getMemory() ;
    computeSum<type>
    <<< divideAndRoundUp(vp.normsVolume, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
    (scratchData,inputData,derOutputData,vp) ;

    // Compute derInputs.
    type * derInputData = (type*)derInput.getMemory() ;
    computeDerInput<type>
    <<< divideAndRoundUp(vp.normsVolume, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
    (derInputData,inputData,normsData,derOutputData,scratchData,op.exponent,vp) ;

    return vl::VLE_Success ;
  }
} ;
*/



/*
template<vl::DataType dataType>
struct NormalizeLpBackward<vl::VLDT_GPU, dataType>
: public NormalizeLpBackwardGPU<dataType,false>
{ } ;

template<vl::DataType dataType>
struct NormalizeLpBackwardWithNorms<vl::VLDT_GPU, dataType>
: public NormalizeLpBackwardGPU<dataType,true>
{ } ;
*/

template<typename type>
__global__ void SpatialSigmoidKernel(type *inputData, int channel, int featH, int featW, int chOffset, int numClasses, int vol)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int tmp;
  int spatialSize = featH * featW;
  int cpos = tid / spatialSize;
  tmp = tid % spatialSize;
  int xpos = tmp / featH;
  int ypos = tmp % featH;

  inputData += ((cpos * channel + chOffset) * spatialSize + xpos*featH + ypos);
  type val;
  if(tid < vol)
  {
    for(int i = 0; i < numClasses; i++)
    {
      val = *inputData;
      *inputData = 1 / (1 + exp(-val));
      inputData += spatialSize;
    }
  }
}

template<typename type>
__global__ void SpatialSoftmaxKernel(type *inputData, int channel, int featH, int featW, int chOffset, int numClasses, int vol)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid >= vol)
    return;
  int tmp;
  int spatialSize = featH * featW;
  int cpos = tid / spatialSize;
  tmp = tid % spatialSize;
  int xpos = tmp / featH;
  int ypos = tmp % featH;

  inputData += ((cpos * channel + chOffset) * spatialSize + xpos*featH + ypos);

  type maxVal = inputData[0];
  type sum = 0;
  for(int cc = 1; cc < numClasses; cc++)
  {
    if(inputData[cc*spatialSize] > maxVal)
    {
      maxVal = inputData[cc*spatialSize];
    }
  }
  for(int cc = 0; cc < numClasses; cc++)
  {
    
    inputData[cc*spatialSize] = exp(inputData[cc*spatialSize] - maxVal);
    sum += inputData[cc*spatialSize];
  }

  sum = 1 / sum;
  for(int cc = 0; cc < numClasses; cc++)
  {
    inputData[cc*spatialSize] *= sum;
  }
}

template<typename type>
struct box{
    type x, y, w, h;
};


template<typename type>
__forceinline__ __device__
box<type> get_region_box(type *data, int stride, int featH, int featW, int hidx, int wdix, type anchorH, type anchorW)
{
  box<type> b;
  b.x = (wdix + data[0]) / featW;
  b.y = (hidx + data[stride]) / featH;
  b.w = exp(data[2*stride]) * anchorW;
  b.h = exp(data[3*stride]) * anchorH;

  return b;
}


template<typename type>
__forceinline__ __device__
type overlap(type x1, type w1, type x2, type w2)
{
    type l1 = x1 - w1 / 2;
    type l2 = x2 - w2 / 2;
    type left = l1 > l2 ? l1 : l2;
    type r1 = x1 + w1 / 2;
    type r2 = x2 + w2 / 2;
    type right = r1 < r2 ? r1 : r2;
    return right - left;
}


template<typename type>
__forceinline__ __device__
type box_intersection(box<type> a, box<type> b)
{
    type w = overlap(a.x, a.w, b.x, b.w);
    type h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0) return 0;
    type area = w*h;
    return area;
}

template<typename type>
__forceinline__ __device__
type box_union(box<type> a, box<type> b)
{
    type i = box_intersection(a, b);
    type u = a.w*a.h + b.w*b.h - i;
    return u;
}


template<typename type>
__forceinline__ __device__
type box_iou(box<type> a, box<type> b)
{
    return box_intersection(a, b) / box_union(a, b);
}


template<typename type>
__forceinline__ __device__
box<type> float_to_box(type *f, int stride)
{
    box<type> b = { 0 };
    b.x = f[0];
    b.y = f[1 * stride];
    b.w = f[2 * stride];
    b.h = f[3 * stride];
    return b;
}


template<typename type>
__forceinline__ __device__
type logistic_gradient(type x)
{
    return x * (1-x);
}

template<typename type>
__forceinline__ __device__
type delta_region_box(box<type> truth, type *x, int h, int w, int index, int hidx, int widx, type ah, type aw, type *delta, type scale, int stride)
{
    box<type> pred = get_region_box(x, stride, h, w, hidx, widx, ah, aw);
    type iou = box_iou(pred, truth);

    type tx = (truth.x*w - widx);
    type ty = (truth.y*h - hidx);
    type tw = log(truth.w*w / aw);
    type th = log(truth.h*h / ah);

//     delta[index + 0 * stride] = scale * (tx - x[index + 0 * stride]);
//     delta[index + 1 * stride] = scale * (ty - x[index + 1 * stride]);
//     delta[index + 2 * stride] = scale * (tw - x[index + 2 * stride]);
//     delta[index + 3 * stride] = scale * (th - x[index + 3 * stride]);
    
    delta[index + 0 * stride] = scale * (tx - x[index + 0 * stride]) * logistic_gradient(x[index + 0 * stride]);
    delta[index + 1 * stride] = scale * (ty - x[index + 1 * stride]) * logistic_gradient(x[index + 1 * stride]);
    delta[index + 2 * stride] = scale * (tw - x[index + 2 * stride]);
    delta[index + 3 * stride] = scale * (th - x[index + 3 * stride]);

    return iou;
}

template<typename type>
__forceinline__ __device__
void delta_region_class(type *output, int stride, type scale, int numClasses, int clsId, type *derInput)
{
  int i;
  for(i = 0; i < numClasses; i++)
  {
    derInput[i*stride] = scale * (((i == clsId) ? 1 : 0) - output[i*stride]);
  }
}

// noobject ------------------------------------------------------------
template<typename type>
__global__ void NoobjectKernel(type *derInput, type *output, int channel, int featH, int featW, int numAnchors, 
                          int classes, int coords, type *biases, type *gts, YoloRegion::YoloRegionLayer layer, int vol,
                          type *avg_anyobj_data)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid >= vol)
  {
    return;
  }

  int tmp;
  int spatialSize = featW*featH;
  int cpos = tid/spatialSize;
  tmp = tid % spatialSize;
  int xpos = tmp / featH;
  int ypos = tmp % featH;
  int aid = cpos % numAnchors;
  int bid = cpos / (numAnchors);
  type aw = biases[aid*2];
  type ah = biases[aid*2+1];

  int entryIdx = cpos * channel * spatialSize + xpos*featH + ypos;

  output += entryIdx;
  derInput += entryIdx;

  box<type> pred = get_region_box(output, spatialSize, featH, featW, ypos, xpos, ah, aw);

  type best_iou = 0;
  for (int t = 0; t < YOLO_MAX_NUM_GT_PER_IMAGE; ++t){
    box<type> truth = float_to_box(gts + t * (layer.coords + 1) + bid * layer.truths, 1);
    //box truth;
    //truth.x = layer.gts[t*(layer.coords+1) + bid * layer.truths];
    if (!truth.x) break;
    type iou = box_iou(pred, truth);
    if (iou > best_iou) {
      best_iou = iou;
    }
  }

  type tmp1;
  int obj_ofs = layer.coords * spatialSize;
  avg_anyobj_data[tid] += output[obj_ofs];
  tmp1 = output[obj_ofs]; // objectness
  derInput[obj_ofs] = -layer.noobject_scale * (0 - tmp1)*logistic_gradient(tmp1);
  if (layer.background) derInput[obj_ofs] = -layer.noobject_scale * (1 - tmp1)*logistic_gradient(tmp1);
  if (best_iou > layer.thresh) {
    derInput[obj_ofs] = 0;
  }
 

  if (layer.iters < 12800){
    box<type> truth = { 0 };
    truth.x = (xpos + .5) / featW;
    truth.y = (ypos + .5) / featH;
    truth.w = aw / featW;
    truth.h = ah / featH;
    delta_region_box(truth, output, featH, featW, 0, ypos, xpos, ah, aw, derInput, 
            type(-.01), spatialSize);
  }
 
}


// object and biases matching kernel -----------------------------------
template<typename type>
__global__ void ObjectKernel(type *derInput, type *output, int channel, int featH, int featW, int numAnchors, 
                          int classes, int coords, type *biases, type *gts, YoloRegion::YoloRegionLayer layer, int vol,
                          type *avg_iou_data, type *avg_cls_data, type *avg_obj_data, type *recall_data)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid >= vol)
  {
  }

  int tmp;
  int spatialSize = featW*featH;

  int bid = tid / (YOLO_MAX_NUM_GT_PER_IMAGE);
  int objId = tid % YOLO_MAX_NUM_GT_PER_IMAGE;

  int truth_entry = (bid * YOLO_MAX_NUM_GT_PER_IMAGE + objId) * (coords+1);

  box<type> truth = float_to_box(gts + truth_entry, 1);
  if(!truth.x) return;

  type best_iou = 0;
  int best_n = 0;

  int hidx = truth.y * featH;
  int widx = truth.x * featW;

  int entryIdx = bid * spatialSize * channel * numAnchors  + widx*featH + hidx;
  output += entryIdx;
  derInput += entryIdx;

  box<type> truth_shift = truth;
  truth_shift.x = 0;
  truth_shift.y = 0;

  type ah, aw, iou;

  for(int n = 0; n < numAnchors; n++)
  {
    aw = biases[2 * n];
    ah = biases[2 * n + 1];
    //int box_index = 1;
    box<type> pred = get_region_box(output + n*channel*spatialSize, spatialSize, featH, featW, hidx, widx, ah, aw);
    if(layer.bias_match)
    {
      pred.w = aw / featW;
      pred.h = ah / featH;
    }

    pred.x = 0;
    pred.y = 0;

    iou = box_iou(pred, truth_shift);
    if(iou > best_iou)
    {
      best_iou = iou;
      best_n = n;
    }
  }

  int box_ofs = best_n * channel * spatialSize;
  int obj_ofs = coords * spatialSize;
  iou = delta_region_box(truth, output + box_ofs, featH, featW, 0, hidx, widx, ah, aw, derInput + box_ofs, 
            type(-layer.coord_scale)*(2 - truth.w*truth.h), spatialSize);
  
  if(coords > 4)
  {
    ////
  }

  int idx = (bid * numAnchors + best_n) * spatialSize + widx*featH + hidx;
  if(iou > 0.5) recall_data[idx] += 1;
  avg_iou_data[idx] += iou;
  avg_obj_data[idx] += output[(best_n*channel + coords)*spatialSize];

  derInput[obj_ofs] = -layer.object_scale * (1 - output[obj_ofs]) * logistic_gradient(output[obj_ofs]);
  if(layer.rescore)
  {
    derInput[obj_ofs]  = -layer.object_scale * (iou - output[obj_ofs]) * logistic_gradient(output[obj_ofs]);
  }

  if(layer.background)
  {
    ////
  }

  int cls_id = gts[truth_entry + coords];

  if(layer.map) {cls_id = layer.map[cls_id]; }

  int cls_ofs = obj_ofs + spatialSize;

  delta_region_class(output + cls_ofs, spatialSize, -(type)layer.class_scale, classes, cls_id, derInput + cls_ofs);

}

// -------------------------------------------------------------------
//                                                        GPU backward
// -------------------------------------------------------------------

template<vl::DataType dataType>
struct YoloRegionBackwardGPU
{
  vl::ErrorCode operator()(YoloRegion &op,
                           Tensor &derInput, Tensor &avg_iou, Tensor &avg_cls, 
                           Tensor &avg_obj, Tensor &avg_anyobj, Tensor &recall, 
                           Tensor const &input)
  {
    typedef typename vl::DataTypeTraits<dataType>::type type ;
    vl::ErrorCode err;
    //printf("YoloRegionBackwardGPU...\n");
    int batchSize = input.getSize();
    int featC = input.getDepth();
    int featH = input.getHeight();
    int featW = input.getWidth();
    int numAnchors = op.numAnchors;
    int coords = op.coords;
    int classes = op.classes;

    int chPerAnchor = coords + 1 + classes;
    int volume = batchSize * numAnchors * featH * featW;
    size_t workspaceSize = batchSize * featC * featH * featW;
    type *tmp;
    cudaError_t cudaErr;

    int objVol = batchSize * YOLO_MAX_NUM_GT_PER_IMAGE;

    // statics for training
    //type avg_iou = 0;
    //type avg_cat = 0;
    //type avg_obj = 0;
    //type avg_anyobj = 0;
    //type recall = 0;
    int count = 0;
/*
    printf("batchSize: %d\n", batchSize);
    printf("featC: %d\n", featC);
    printf("featH: %d\n", featH);
    printf("featW: %d\n", featW);
    printf("numAnchors: %d\n", numAnchors);
    printf("classes: %d\n", classes);
    printf("coord: %d\n", coords);
*/

    Buffer biases_buf, gts_buf;
    type *biases, *gts;

    Buffer biases_gpu_buf, gts_gpu_buf;
    type *biases_gpu, *gts_gpu;

    type * inputData = (type *)input.getMemory() ; /////////   for debug, no const

    type *avg_iou_data = (type *)avg_iou.getMemory();
    type *avg_cls_data = (type *)avg_cls.getMemory();
    type *avg_obj_data = (type *)avg_obj.getMemory();
    type *avg_anyobj_data = (type *)avg_anyobj.getMemory();
    type *recall_data = (type *)recall.getMemory();

    // Compute derInputs.
    type *derInputData = (type *)derInput.getMemory() ;
    type *output = (type *)op.context.getWorkspace(vl::VLDT_GPU, workspaceSize*sizeof(type)) ; // tmp
    if(output == NULL)
    {
      //mexPrintf("fail to get workspace for temp output\n");
      goto done;
    }

    // biases on cpu----------------------------
    err = biases_buf.init(vl::VLDT_CPU, dataType, op.numAnchors * 2);
    if(err != vl::VLE_Success)
    {
      //mexPrintf("fail to init buffer for anchors on cpu\n");
      goto done;
    }
    biases = (type *)biases_buf.getMemory();

    // ground truths on cpu----------------------
    err = gts_buf.init(vl::VLDT_CPU, dataType, batchSize * YOLO_MAX_NUM_GT_PER_IMAGE * (coords+1));
    if(err != vl::VLE_Success)
    {
      //mexPrintf("fail to init buffer for anchors on cpu\n");
      goto done;
    }
    gts = (type *)gts_buf.getMemory();

    // init with zeros
    err = operations<vl::VLDT_CPU, type>::fill(biases, op.numAnchors * 2, 0);
    if(err != vl::VLE_Success)
    {
      //mexPrintf("fail to fill biases with 0\n");
      goto done;
    }
    operations<vl::VLDT_CPU, type>::fill(gts, batchSize * YOLO_MAX_NUM_GT_PER_IMAGE * (coords+1), 0);
    if(err != vl::VLE_Success)
    {
      //mexPrintf("fail to fill gts with 0\n");
      goto done;
    }

    //type *biases_gpu = (type *)op.context.getWorkspace(vl::VLDT_GPU, sizeof(type) * op.numAnchors * 2);
    //type *gts_gpu = (type *)op.context.getWorkspace(vl::VLDT_GPU, sizeof(type) * batchSize * YOLO_MAX_NUM_GT_PER_IMAGE * (coords+1));

    // biases on gpu ----------------------------
    err = biases_gpu_buf.init(vl::VLDT_GPU, dataType, op.numAnchors * 2);
    if(err != vl::VLE_Success)
    {
      //mexPrintf("fail to init buffer for anchors on cpu\n");
      goto done;
    }
    biases_gpu = (type *)biases_gpu_buf.getMemory();

    // ground truths on gpu ---------------------
    gts_gpu_buf.init(vl::VLDT_GPU, dataType, batchSize * YOLO_MAX_NUM_GT_PER_IMAGE * (coords+1));
    if(err != vl::VLE_Success)
    {
      //mexPrintf("fail to init buffer for anchors on cpu\n");
      goto done;
    }
    gts_gpu = (type *)gts_gpu_buf.getMemory();

    //operations<vl::VLDT_GPU, type>::fill(biases_gpu, op.numAnchors * 2, 0);
    //operations<vl::VLDT_GPU, type>::fill(gts_gpu, batchSize * YOLO_MAX_NUM_GT_PER_IMAGE * (coords+1), 0);

    for(int i = 0; i < numAnchors; i++)
    {
      biases[i*2] = op.biases[i*2];
      biases[i*2+1] = op.biases[i*2+1];
      //printf("%f %f ", biases[i*2], biases[i*2]);
    }
    //printf("\n");
    
    for(int i = 0; i < op.gts.size(); i++)
    {
      std::vector<float> gt = op.gts[i];
      tmp = gts + i * YOLO_MAX_NUM_GT_PER_IMAGE * (coords+1);
      for(int j = 0; j < gt.size(); j++)
      {
        tmp[j] = gt[j];
      }
    }
    

/*
    for(int i = 0; i < op.gts.size(); i++)
    {
      std::vector<float> gt = op.gts[i];
      tmp = gts + i * YOLO_MAX_NUM_GT_PER_IMAGE * (coords+1);
      for(int j = 0; j < 20 / (coords+1); j++) //gt.size() 
      {
        for(int k = 0; k < coords+1; k++)
        {
          printf("%f ", tmp[(coords + 1) * j + k]);
        }
        printf("\n");
      }
      printf("\n");
    }
*/    

    //printf("copy biases and ground truths to device\n");
    
    cudaErr = cudaMemcpy(biases_gpu, biases, sizeof(type) * numAnchors * 2, cudaMemcpyHostToDevice);
    if(cudaErr != cudaSuccess)
    {
      //mexPrintf("fail to copy biases to gpu, %d\n", cudaErr);
      err = vl::VLE_Cuda;
      goto done;
    }
    cudaErr = cudaMemcpy(gts_gpu, gts, sizeof(type) * batchSize * YOLO_MAX_NUM_GT_PER_IMAGE * (coords + 1), cudaMemcpyHostToDevice);
    if(cudaErr != cudaSuccess)
    {
      //mexPrintf("fail to copy ground truths to gpu, %d\n", cudaErr);
      err = vl::VLE_Cuda;
      goto done;
    }
    
    err = operations<vl::VLDT_GPU, type>::copy(output, inputData, workspaceSize);
    if(err != VLE_Success)
    {
      //mexPrintf("fail to copy input to output\n");
      goto done;
    }
    

    err = operations<vl::VLDT_GPU, type>::fill(derInputData, workspaceSize, (type)0);
    if(err != VLE_Success)
    {
      //mexPrintf("fail to fill derInput with 0, %d\n", err);
      goto done;
    }
    //cudaDeviceSynchronize();

    // 
    SpatialSigmoidKernel<<<divideAndRoundUp(volume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS>>>
    (output, chPerAnchor, featH, featW, 0, 2, volume); // offset of xc, yc

    SpatialSigmoidKernel<<<divideAndRoundUp(volume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS>>>
    (output, chPerAnchor, featH, featW, coords, 1, volume); // objectness

    SpatialSoftmaxKernel<<<divideAndRoundUp(volume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS>>>
    (output, chPerAnchor, featH, featW, coords+1, classes, volume);

    //YoloRegion::YoloRegionLayer *layer_gpu = (YoloRegion::YoloRegionLayer *)op.context.getWorkspace(vl::VLDT_GPU, sizeof(YoloRegion::YoloRegionLayer));
    //cudaMemcpy(layer_gpu, &op.layer, sizeof(YoloRegion::YoloRegionLayer), cudaMemcpyHostToDevice);
    
    NoobjectKernel<<<divideAndRoundUp(volume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS>>>
    (derInputData, output, chPerAnchor, featH, featW, numAnchors, classes, 
      coords, biases_gpu, gts_gpu, op.layer, volume, avg_anyobj_data);

    ObjectKernel<<<divideAndRoundUp(objVol, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS>>>
    (derInputData, output, chPerAnchor, featH, featW, numAnchors, classes, 
    coords, biases_gpu, gts_gpu, op.layer, objVol, avg_iou_data, avg_cls_data, avg_obj_data, recall_data);

    //operations<vl::VLDT_GPU, type>::copy(derInputData, output, workspaceSize);

    biases_buf.clear(); gts_buf.clear();
    biases_gpu_buf.clear(); gts_gpu_buf.clear();
    return vl::VLE_Success ;

done:
    return op.context.passError(err, __func__) ;
    
  }
} ;

template<vl::DataType dataType>
struct YoloRegionForward<vl::VLDT_GPU, dataType>
: public YoloRegionForwardGPU<dataType>
{ } ;

template<vl::DataType dataType>
struct YoloRegionBackward<vl::VLDT_GPU, dataType>
: public YoloRegionBackwardGPU<dataType>
{ } ;
