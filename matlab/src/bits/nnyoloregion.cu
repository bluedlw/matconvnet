// @file nnconv.cu
// @brief Convolution block
// @author Andrea Vedaldi
// @author Max Jaderberg

/*
Copyright (C) 2014 Andrea Vedaldi and Max Jaderberg.
Copyright (C) 2015-17 Andrea Vedaldi.

All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnyoloregion.hpp"
#include "impl/dispatcher.hpp"
#include "impl/blashelper.hpp"
#include "impl/copy.hpp"
#include <cassert>

using namespace vl ;
using namespace vl::nn ;
using namespace vl::impl ;

template<DeviceType deviceType, DataType dataType> struct YoloRegionForward ;
template<DeviceType deviceType, DataType dataType> struct YoloRegionBackward ;

// -------------------------------------------------------------------
//                                                             Forward
// -------------------------------------------------------------------

template<DataType dataType>
struct YoloRegionForward<vl::VLDT_CPU, dataType>
{
  vl::ErrorCode operator()
  (YoloRegion &op,
   Tensor &output, Tensor const &input)
  {

    assert(output) ;
    assert(input) ;

    vl::ErrorCode error = VLE_Success;
    //typedef typename vl::DataTypeTraits<dataType>::type type ;


  //done:
    //return op.context.passError(error, __func__) ;
    return error;
  }
} ;

// -------------------------------------------------------------------
//                                                            Backward
// -------------------------------------------------------------------

template<DataType dataType>
struct YoloRegionBackward<vl::VLDT_CPU, dataType>
{
  vl::ErrorCode operator()
  (YoloRegion &op,
   Tensor &derInput,
   Tensor const &input)
  {

    vl::ErrorCode error = VLE_Success;
    //typedef typename vl::DataTypeTraits<dataType>::type type ;

  //done:
    //return op.context.passError(error, __func__) ;

    return error;
  }
} ;



// -------------------------------------------------------------------
//                                                             Drivers
// -------------------------------------------------------------------

#if ENABLE_GPU
#include "nnyoloregion_gpu.cu"
#endif

YoloRegion::YoloRegion(Context &context,
    std::vector<std::vector<float>> gts, std::vector<float> biases,
    int bias_match, int classes, int coords, int softmax,
    float jitter, int rescore, float object_scale, float noobject_scale,
    float class_scale, float coord_scale, int absolute, float thresh,
    int random, int train, int iters)
:
context(context),
gts(gts), biases(biases), bias_match(bias_match), classes(classes),
coords(coords), softmax(softmax), jitter(jitter), rescore(rescore),
object_scale(object_scale), noobject_scale(noobject_scale), class_scale(class_scale),
coord_scale(coord_scale), absolute(absolute), thresh(thresh), random(random),
train(train), iters(iters)
{
  memset(&layer, 0, sizeof(YoloRegionLayer));

  network.train = train;
  //network.input = data;
  network.seen[0] = iters;

  //layer.batch = batchSize;
  //layer.c = channel;
  //layer.h = width; //transpose
  //layer.w = height; //transpose
  //layer.outputs = height*width*numAnchors*(classes + coords + 1);
  //layer.inputs = l.outputs;
  //layer.n = numAnchors;
  layer.truths = YOLO_MAX_NUM_GT_PER_IMAGE * (coords + 1);
  layer.bias_match = bias_match;
  layer.classes = classes;
  layer.coords = coords;
  layer.softmax = softmax;
  layer.jitter = jitter;
  layer.rescore = rescore;
  layer.object_scale = object_scale;
  layer.noobject_scale = noobject_scale;
  layer.class_scale = class_scale;
  layer.coord_scale = coord_scale;
  layer.absolute = absolute;
  layer.thresh = thresh;
  layer.random = random;
  layer.mask_scale = 1; //default
  layer.background = 0; //default
  layer.iters = iters;

  ///////////////////////////////////////
  numAnchors = biases.size() / 2;
  //printf("class construct, numAnchors: %d\n", numAnchors);
}

vl::ErrorCode
YoloRegion::forward(vl::Tensor &output,
                    vl::Tensor const &input)
{
  return dispatch<YoloRegionForward>()(*this,output,input) ;
}

vl::ErrorCode
YoloRegion::backward(vl::Tensor &derInput,
                     vl::Tensor const &input)
{
  return dispatch<YoloRegionBackward>()(*this,derInput,input) ;
}

