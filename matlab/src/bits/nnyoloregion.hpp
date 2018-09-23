// @file nnyoloregion.hpp
// @brief Yolo Region block
// @author Andrea Vedaldi

/*
Copyright (C) 2014-17 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnyoloregion__
#define __vl__nnyoloregion__

#include <vector>
#include "data.hpp"
#include <stdio.h>

#define YOLO_MAX_NUM_ANCHORS 128
#define YOLO_MAX_NUM_GT_PER_IMAGE 256
#define BATCH_SIZE 1024

namespace vl { namespace nn {

  class YoloRegion {
  public:
    YoloRegion(vl::Context &context,
        std::vector<std::vector<float>> gts, std::vector<float> biases,
        int bias_match, int classes, int coords, int softmax,
        float jitter, int rescore, float object_scale, float noobject_scale,
        float class_scale, float coord_scale, int absolute, float thresh,
        int random, int train, int iters);

    vl::ErrorCode forward(vl::Tensor &output,
                          vl::Tensor const &data) ;

    vl::ErrorCode backward(vl::Tensor &derData, vl::Tensor &avg_iou, 
                           vl::Tensor &avg_cls, vl::Tensor &avg_obj, 
                           vl::Tensor &avg_anyobj, vl::Tensor &recall,
                           vl::Tensor const &data) ;
    vl::Context& context ;

    std::vector<std::vector<float>> gts;
    std::vector<float> biases;
    int bias_match;
    int classes;
    int coords;
    int softmax;
    float jitter;
    int rescore;
    float object_scale;
    float noobject_scale;
    float class_scale;
    float coord_scale;
    int absolute;
    float thresh;
    int random;
    int train;
    int iters;

    int numAnchors;


    typedef struct{
      float x, y, w, h;
    } box;
    typedef struct{
      int *leaf;
      int n;
      int *parent;
      int *child;
      int *group;
      char **name;

      int groups;
      int *group_size;
      int *group_offset;
    } tree;
    typedef struct
    {
      int batch;
      int c;
      int h;
      int w;
      int n;
      int bias_match;
      int classes;
      int coords;
      int softmax;
      tree *softmax_tree;
      float jitter;
      int rescore;
      float object_scale;
      float noobject_scale;
      float class_scale;
      float coord_scale;
      int absolute;
      float thresh;
      int random;
      int background;
      int *map;
      float mask_scale;
    
      float temperature;

      int outputs;
      int inputs;
      float *output;
      float *delta;
      int truths;
      float biases[YOLO_MAX_NUM_ANCHORS * 2];

      float cost[1];

      int numAnchors;
      //float truth[YOLO_MAX_NUM_GT_PER_IMAGE * (4+1) * BATCH_SIZE]; // x1,y1,x2,y2,clsId
      int iters;
      int coordWarmup;
      int noobjectWarmup;
    }YoloRegionLayer;

    typedef struct
    {
      float *input;
      int numGts;
      float *truth;

      int train;
      int seen[1];
    }YoloNetwork;



    YoloRegionLayer layer;
    YoloNetwork network;
  } ;



} }

#endif /* defined(__vl__nnyoloregion__) */
