// @file vl_nnnormalize.cu
// @brief Normalization block MEX wrapper
// @author Andrea Vedaldi

/*
Copyright (C) 2014-15 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include <vector>
#include "bits/mexutils.h"
#include "bits/nnyoloregion.hpp"
#include "bits/datamex.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <assert.h>

/* option codes */
enum {
  opt_verbose = 0
} ;

/* options */
VLMXOption  options [] = {
  {"Verbose",          0,   opt_verbose           },
  {0,                  0,   0                     }
} ;

/* ---------------------------------------------------------------- */
/*                                                          Context */
/* ---------------------------------------------------------------- */

vl::MexContext context ;

/*
 Resetting the context here resolves a crash when MATLAB quits and
 the ~Context function is implicitly called on unloading the MEX file.
 */
void atExit()
{
  context.clear() ;
}

/* ---------------------------------------------------------------- */
/*                                                       MEX driver */
/* ---------------------------------------------------------------- */

enum {
  IN_DATA = 0, IN_GT, IN_ANCHORS, IN_BIAS_MATCH, IN_CLASSES, IN_COORDS, IN_SOFTMAX,
  IN_JITTER, IN_RESCORE, IN_OBJECT_SCALE, IN_NOOBJECT_SCALE, IN_CLASS_SCALE,
  IN_COORD_SCALE, IN_ABSOLUTE, IN_THRESH, IN_RANDOM, IN_TRAIN, IN_ITERS, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  bool backMode = false ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  mexAtExit(atExit) ;

  //printf("entering mexFunction\n");
  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  backMode = 1;
/*
  if (nin > 2 && vlmxIsString(in[2],-1)) {
    next = 2 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 3) ;
  }
*/

/*
  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose :
        ++ verbosity ;
        break ;
      default: break ;
    }
  }
*/

  //printf("1111111\n");
  vl::MexTensor data(context) ;
  vl::MexTensor derOutput(context) ;

  //printf("22222222\n");

  data.init(in[IN_DATA]) ;
  data.reshape(4) ;


  //printf("33333333\n");
/*
  if (backMode) {
    derOutput.init(in[IN_DEROUTPUT]) ; //2 input, params, derOutput
    derOutput.reshape(4) ;
  }
*/

/*
  if (backMode && ! vl::areCompatible(data, derOutput)) {
    mexErrMsgTxt("DATA and DEROUTPUT do not have compatible formats.") ;
  }
  if (backMode && (data.getShape() != derOutput.getShape())) {
    mexErrMsgTxt("DATA and DEROUTPUT do not have the same size.") ;
  }
*/
/*
  if (!mxIsNumeric(in[IN_PARAM]) ||
       mxGetClassID(in[IN_PARAM]) != mxDOUBLE_CLASS ||
       mxIsComplex(in[IN_PARAM]) ||
       mxGetNumberOfElements(in[IN_PARAM]) != 4)
  {
    mexErrMsgTxt("PARAM is not a plain 4 vector.") ;
  }
*/
/*
  normDepth = (size_t) mxGetPr(in[IN_PARAM])[0]  ;
  normKappa = mxGetPr(in[IN_PARAM])[1]  ;
  normAlpha = mxGetPr(in[IN_PARAM])[2]  ;
  normBeta = mxGetPr(in[IN_PARAM])[3]  ;
  if (normDepth < 1) {
    mexErrMsgTxt("The normalization depth is smaller than 1.") ;
  }
*/

  int batchSize = data.getSize();

  /**************   parse   arguments  **************/
  int bias_match = (int)mxGetScalar(in[IN_BIAS_MATCH]);
  int classes = (int)mxGetScalar(in[IN_CLASSES]);
  int coords = (int)mxGetScalar(in[IN_COORDS]);
  int softmax = (int)mxGetScalar(in[IN_SOFTMAX]);
  float jitter = (float)mxGetScalar(in[IN_JITTER]);
  int rescore = (int)mxGetScalar(in[IN_RESCORE]);
  float object_scale = (float)mxGetScalar(in[IN_OBJECT_SCALE]);
  float noobject_scale = (float)mxGetScalar(in[IN_NOOBJECT_SCALE]);
  float class_scale = (float)mxGetScalar(in[IN_CLASS_SCALE]);
  float coord_scale = (float)mxGetScalar(in[IN_COORD_SCALE]);
  int absolute = (int)mxGetScalar(in[IN_ABSOLUTE]);
  float thresh = (float)mxGetScalar(in[IN_THRESH]);
  int random = (int)mxGetScalar(in[IN_RANDOM]);
  int train = (int)mxGetScalar(in[IN_TRAIN]);
  int iters = (int)mxGetScalar(in[IN_ITERS]);

  //printf("43333333\n");

  mwSize numDims;
  numDims = mxGetNumberOfDimensions(in[IN_ANCHORS]);
  const mwSize *dims = mxGetDimensions(in[IN_ANCHORS]);
  if(numDims != 2 || dims[1] != 2)
  {
      mexErrMsgTxt("anchors should be nx2 array");
  }
  int numAnchors = dims[0];
  double *dp = mxGetPr(in[IN_ANCHORS]);
  std::vector<float> anchors;
  anchors.clear();
  anchors.resize(dims[1]*dims[0]);
  for(int i = 0; i < numAnchors; i++)
  {
      //anchors[2*i] = dp[i+numAnchors];
      //anchors[2*i+1] = dp[i];
      anchors[2*i]   = dp[i];
      anchors[2*i+1] = dp[i+numAnchors];
  }

  //printf("555333333\n");

  std::vector<std::vector<float>> gts;
  gts.clear();
  gts.resize(batchSize);
  for(int i = 0; i < batchSize; i++)
  {
      mxArray *gtArr = mxGetCell(in[IN_GT], i);
      dp = mxGetPr(gtArr);
      numDims = mxGetNumberOfDimensions(gtArr);
      dims = mxGetDimensions(gtArr);
      if(numDims != 2)
      {
          mexErrMsgTxt("ground truths per batch should be dimensions of 2");
      }
      if((dims[0] != 0 && dims[1] != 0) && dims[1] != 5)
      {
          mexErrMsgTxt("one ground truth should be [x0,y0,x1,y1,classId]");
      }
      std::vector<float> tmp;
      tmp.resize(dims[0] * dims[1]); // nx5
      int num = dims[0];
      for(int j = 0; j < num; j++)
      {
          //tmp[j*5] = dp[j+num];
          //tmp[j*5+1] = dp[j+num*0];
          //tmp[j*5+2] = dp[j+num*3];
          //tmp[j*5+3] = dp[j+num*2];
          //tmp[j*5+4] = dp[j+num*4];

          tmp[j*5] = dp[j];
          tmp[j*5+1] = dp[j+num];
          tmp[j*5+2] = dp[j+num*2];
          tmp[j*5+3] = dp[j+num*3];
          tmp[j*5+4] = dp[j+num*4];
      }
      gts[i] = tmp;
  }

  /**************   parse   arguments  **************/

  //printf("create MexTensor\n");
  /* Create output buffers */
  vl::DeviceType deviceType = data.getDeviceType() ;
  vl::DataType dataType = data.getDataType() ;
  vl::MexTensor output(context) ;
  vl::MexTensor derData(context) ;

  ///////   training statics /////////////
  vl::MexTensor avg_iou(context);
  vl::MexTensor avg_cls(context);
  vl::MexTensor avg_obj(context);
  vl::MexTensor avg_anyobj(context);
  vl::MexTensor recall(context);


  if (!backMode) {
    output.init(deviceType, dataType, data.getShape()) ;
  } else {
    derData.init(deviceType, dataType, data.getShape()) ;
  }

  if(backMode)
  {
    TensorShape shape = data.getShape();
    shape.setDepth(numAnchors);
    avg_iou.initWithZeros(deviceType, dataType, shape);
    avg_cls.initWithZeros(deviceType, dataType, shape);
    avg_obj.initWithZeros(deviceType, dataType, shape);
    avg_anyobj.initWithZeros(deviceType, dataType, shape);
    recall.initWithZeros(deviceType, dataType, shape);
  }

  if (verbosity > 0) {
/*
    mexPrintf("vl_nnnormalize: mode %s; %s\n",  (data.getDeviceType()==vl::VLDT_GPU)?"gpu":"cpu", backMode?"backward":"forward") ;
    mexPrintf("vl_nnnormalize: (depth,kappa,alpha,beta): (%d,%g,%g,%g)\n",
              normDepth, normKappa, normAlpha, normBeta) ;
    vl::print("vl_nnnormalize: data: ", data) ;
    if (backMode) {
      vl::print("vl_nnnormalize: derOutput: ", derOutput) ;
      vl::print("vl_nnnormalize: derData: ", derData) ;
    } else {
      vl::print("vl_nnnormalize: output: ", output) ;
    }
*/
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  vl::ErrorCode error ;

  //printf("initialize YoloRegion op\n");
  vl::nn::YoloRegion op(context, gts, anchors, bias_match, classes, coords, softmax, jitter,
                        rescore, object_scale, noobject_scale, class_scale,
                        coord_scale, absolute, thresh, random, train, iters);
  //op.setAnchors();
  //op.setGTs();

  //vl::nn::LRN op(context,normDepth,normKappa,normAlpha,normBeta) ;

  if (!backMode) {
    error = op.forward(output, data) ;
  } else {
    error = op.backward(derData, avg_iou, avg_cls, avg_obj, avg_anyobj, recall, data) ;
  }

  /* -------------------------------------------------------------- */
  /*                                                         Finish */
  /* -------------------------------------------------------------- */

  if (error != vl::VLE_Success) {
    mexErrMsgTxt(context.getLastErrorMessage().c_str()) ;
  }
  if (backMode) {
    out[OUT_RESULT] = derData.relinquish() ;
  } else {
    out[OUT_RESULT] = output.relinquish() ;
    out[1] = avg_iou.relinquish();
    out[2] = avg_cls.relinquish();
    out[3] = avg_obj.relinquish();
    out[4] = avg_anyobj.relinquish();
    out[5] = recall.relinquish();
  }
}
