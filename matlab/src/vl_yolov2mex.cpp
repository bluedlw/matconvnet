
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

#include "mex.h"

#define MAX_ANCHOR_NUM 128
#define MAX_GT_PER_IMAGE 256

typedef enum
{
    LOGISTIC
}ACTIVATION;

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
    float biases[MAX_ANCHOR_NUM * 2];

    float cost[1];
}layer;

typedef struct
{
    float *input;
    int numGts;
    float *truth;

    int train;
    int seen[1];
}network;

/*
#include "region_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>


layer make_region_layer(int batch, int w, int h, int n, int classes, int coords)
{
    layer l = { 0 };
    l.type = REGION;

    l.n = n;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + coords + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.coords = coords;
    l.cost = calloc(1, sizeof(float));
    l.biases = calloc(n * 2, sizeof(float));
    l.bias_updates = calloc(n * 2, sizeof(float));
    l.outputs = h*w*n*(classes + coords + 1);
    l.inputs = l.outputs;
    l.truths = 30 * (l.coords + 1);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    int i;
    for (i = 0; i < n * 2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_region_layer;
    l.backward = backward_region_layer;
#ifdef GPU
    l.forward_gpu = forward_region_layer_gpu;
    l.backward_gpu = backward_region_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "detection\n");
    srand(0);

    return l;
}
*/

/*
void resize_region_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + l->coords + 1);
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu = cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);
#endif
}
*/

float mag_array(float *a, int n)
{
    int i;
    float sum = 0;
    for (i = 0; i < n; ++i){
        sum += a[i] * a[i];
    }
    return sqrt(sum);
}

box float_to_box(float *f, int stride)
{
    box b = { 0 };
    b.x = f[0];
    b.y = f[1 * stride];
    b.w = f[2 * stride];
    b.h = f[3 * stride];
    return b;
}
float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b) / box_union(a, b);
}

float logistic_gradient(float x)
{
    return x * (1-x);
}

box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;
//     b.x = (i + x[index + 0 * stride]) / w;
//     b.y = (j + x[index + 1 * stride]) / h;
//     b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
//     b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    
    b.x = (i + x[index + 1 * stride]) / w;
    b.y = (j + x[index + 0 * stride]) / h;
    b.w = exp(x[index + 3 * stride]) * biases[2 * n] / w;
    b.h = exp(x[index + 2 * stride]) * biases[2 * n + 1] / h;
    return b;
}


float delta_region_box(box truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_region_box(x, biases, n, index, i, j, w, h, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x*w - i);
    float ty = (truth.y*h - j);
    float tw = log(truth.w*w / biases[2 * n]);
    float th = log(truth.h*h / biases[2 * n + 1]);

//     delta[index + 0 * stride] = scale * (tx - x[index + 0 * stride]);
//     delta[index + 1 * stride] = scale * (ty - x[index + 1 * stride]);
//     delta[index + 2 * stride] = scale * (tw - x[index + 2 * stride]);
//     delta[index + 3 * stride] = scale * (th - x[index + 3 * stride]);
    
    delta[index + 1 * stride] = scale * (tx - x[index + 1 * stride]) * logistic_gradient(x[index + 1 * stride]);
    delta[index + 0 * stride] = scale * (ty - x[index + 0 * stride]) * logistic_gradient(x[index + 0 * stride]);
    delta[index + 3 * stride] = scale * (tw - x[index + 3 * stride]);
    delta[index + 2 * stride] = scale * (th - x[index + 2 * stride]);

    return iou;
}


void delta_region_mask(float *truth, float *x, int n, int index, float *delta, int stride, int scale)
{
    int i;
    for (i = 0; i < n; ++i){
        delta[index + i*stride] = scale*(truth[i] - x[index + i*stride]);
    }
}


void delta_region_class(float *output, float *delta, int index, int class_, int classes, tree *hier, float scale, int stride, float *avg_cat, int tag)
{
    int i, n;
    if (hier){
        float pred = 1;
        while (class_ >= 0){
            pred *= output[index + stride*class_];
            int g = hier->group[class_];
            int offset = hier->group_offset[g];
            for (i = 0; i < hier->group_size[g]; ++i){
                delta[index + stride*(offset + i)] = scale * (0 - output[index + stride*(offset + i)]);
            }
            delta[index + stride*class_] = scale * (1 - output[index + stride*class_]);

            class_ = hier->parent[class_];
        }
        *avg_cat += pred;
    }
    else {
        if (delta[index] && tag){
            delta[index + stride*class_] = scale * (1 - output[index + stride*class_]);
            return;
        }
        for (n = 0; n < classes; ++n){
            delta[index + stride*n] = scale * (((n == class_) ? 1 : 0) - output[index + stride*n]);
            if (n == class_) *avg_cat += output[index + stride*n];
        }
    }
}


float logit(float x)
{
    return log(x / (1. - x));
}

float tisnan(float x)
{
    return (x != x);
}

int entry_index(layer l, int batch, int location, int entry)
{
    int n = location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(l.coords + l.classes + 1) + entry*l.w*l.h + loc;
}

void activate_array(float *x, const int n, const ACTIVATION a)
{
    int i;
    for (i = 0; i < n; ++i){
        //x[i] = activate(x[i], a);
        x[i] = 1. / (1. + exp(-x[i]));
    }
}

void softmax(float *input, int n, float temp, int stride, float *output)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for (i = 0; i < n; ++i){
        if (input[i*stride] > largest) largest = input[i*stride];
    }
    for (i = 0; i < n; ++i){
        float e = exp(input[i*stride] / temp - largest / temp);
        sum += e;
        output[i*stride] = e;
    }
    for (i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}

void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    int g, b;
    for (b = 0; b < batch; ++b){
        for (g = 0; g < groups; ++g){
            softmax(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
        }
    }
}

void forward_region_layer(layer l, network net)
{
    int i, j, b, t, n;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));

#ifndef GPU
    for (b = 0; b < l.batch; ++b){
        for (n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2 * l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, l.coords);
            if (!l.background) activate_array(l.output + index, l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, l.coords + 1);
            if (!l.softmax && !l.softmax_tree) activate_array(l.output + index, l.classes*l.w*l.h, LOGISTIC);
        }
    }
    if (l.softmax_tree){
        int i;
        int count = l.coords + 1;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_cpu(net.input + count, group_size, l.batch, l.inputs, l.n*l.w*l.h, 1, l.n*l.w*l.h, l.temperature, l.output + count);
            count += group_size;
        }
    }
    else if (l.softmax){
        int index = entry_index(l, 0, 0, l.coords + !l.background);
        softmax_cpu(net.input + index, l.classes + l.background, l.batch*l.n, l.inputs / l.n, l.w*l.h, 1, l.w*l.h, 1, l.output + index);
    }
#endif

    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if (!net.train) return;
    float avg_iou = 0;
    float recall = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) {
        /*
        if (l.softmax_tree){
            int onlyclass = 0;
            for (t = 0; t < 30; ++t){
                box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);
                if (!truth.x) break;
                int class = net.truth[t*(l.coords + 1) + b*l.truths + l.coords];
                float maxp = 0;
                int maxi = 0;
                if (truth.x > 100000 && truth.y > 100000){
                    for (n = 0; n < l.n*l.w*l.h; ++n){
                        int class_index = entry_index(l, b, n, l.coords + 1);
                        int obj_index = entry_index(l, b, n, l.coords);
                        float scale = l.output[obj_index];
                        l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
                        float p = scale*get_hierarchy_probability(l.output + class_index, l.softmax_tree, class, l.w*l.h);
                        if (p > maxp){
                            maxp = p;
                            maxi = n;
                        }
                    }
                    int class_index = entry_index(l, b, maxi, l.coords + 1);
                    int obj_index = entry_index(l, b, maxi, l.coords);
                    delta_region_class(l.output, l.delta, class_index, class, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat, !l.softmax);
                    if (l.output[obj_index] < .3) l.delta[obj_index] = l.object_scale * (.3 - l.output[obj_index]);
                    else  l.delta[obj_index] = 0;
                    l.delta[obj_index] = 0;
                    ++class_count;
                    onlyclass = 1;
                    break;
                }
            }
            if (onlyclass) continue;
        }*/
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);
                    float best_iou = 0;
                    for (t = 0; t < MAX_GT_PER_IMAGE; ++t){
                        box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);
                        if (!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                        }
                    }
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, l.coords);
                    avg_anyobj += l.output[obj_index];
                    l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index])*logistic_gradient(l.output[obj_index]);
                    if (l.background) l.delta[obj_index] = l.noobject_scale * (1 - l.output[obj_index])*logistic_gradient(l.output[obj_index]);
                    if (best_iou > l.thresh) {
                        l.delta[obj_index] = 0;
                    }

                    if (*(net.seen) < 12800){
                        box truth = { 0 };
                        truth.x = (i + .5) / l.w;
                        truth.y = (j + .5) / l.h;
                        truth.w = l.biases[2 * n] / l.w;
                        truth.h = l.biases[2 * n + 1] / l.h;
                        delta_region_box(truth, l.output, l.biases, n, box_index, i, j, l.w, l.h, l.delta, 
                                .01, l.w*l.h);
                    }
                }
            }
        }
        for (t = 0; t < MAX_GT_PER_IMAGE; ++t){
            box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);

            if (!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            box truth_shift = truth;
            truth_shift.x = 0;
            truth_shift.y = 0;
            for (n = 0; n < l.n; ++n){
                int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);
                if (l.bias_match){
                    pred.w = l.biases[2 * n] / l.w;
                    pred.h = l.biases[2 * n + 1] / l.h;
                }
                pred.x = 0;
                pred.y = 0;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }

            int box_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 0);
            float iou = delta_region_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, 
                    l.delta, l.coord_scale *  (2 - truth.w*truth.h), l.w*l.h);
            if (l.coords > 4){
                int mask_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 4);
                delta_region_mask(net.truth + t*(l.coords + 1) + b*l.truths + 5, l.output, l.coords - 4, 
                        mask_index, l.delta, l.w*l.h, l.mask_scale);
            }
            if (iou > .5) recall += 1;
            avg_iou += iou;

            int obj_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, l.coords);
            avg_obj += l.output[obj_index];
            l.delta[obj_index] = l.object_scale * (1 - l.output[obj_index])*logistic_gradient(l.output[obj_index]);
            if (l.rescore) {
                l.delta[obj_index] = l.object_scale * (iou - l.output[obj_index])*logistic_gradient(l.output[obj_index]);
            }
            if (l.background){
                l.delta[obj_index] = l.object_scale * (0 - l.output[obj_index])*logistic_gradient(l.output[obj_index]);
            }

            int class_ = net.truth[t*(l.coords + 1) + b*l.truths + l.coords];
            if (l.map) class_ = l.map[class_];
            int class_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, l.coords + 1);
            delta_region_class(l.output, l.delta, class_index, class_, l.classes, l.softmax_tree, 
                    l.class_scale, l.w*l.h, &avg_cat, !l.softmax);
            ++count;
            ++class_count;
        }
    }
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n", 
            avg_iou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w*l.h*l.n*l.batch), 
            recall / count, count);
}




void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // args:    0: input array, 
    //          1: gt, 
    //          2: anchors, 
    //          3: bias_match, 
    //          4: classes, 
    //          5: coords
    //          6: softmax, 
    //          7: jitter=0.3, 
    //          8: rescore, 
    //          9: object_scale=5, 
    //          10: noobject_scale=1, 
    //          11: class_scale=1, 
    //          12: coord_scale=1, 
    //          13: absolute=1, 
    //          14: thresh = 0.6, 
    //          15: random, 
    //          16: train/test.
    //          17: iters.
    int numArgs = 18;
    //printf("number of inputs: %d\n", nrhs);
    if (nrhs != numArgs)
    {
        mexErrMsgTxt("number of input args are not correct");
    }

    //////////////////////// input /////////////////////
    if (!mxIsSingle(prhs[0]))
    {
        mexErrMsgTxt("only support single precision input");
    }
    
    layer l;
    network net;
    memset(&l, 0, sizeof(l));
    memset(&net, 0, sizeof(network));
    
    
    int batchSize = 1, channel = 1, height = 1, width = 1;
    float *data = (float *)mxGetData(prhs[0]);
    int inputNDims = mxGetNumberOfDimensions(prhs[0]);
    const size_t *inputDims = mxGetDimensions(prhs[0]);
    height = inputDims[0]; width = inputDims[1];
    switch (inputNDims)
    {
    default:
        mexErrMsgTxt("only support <= 4 dimensions of input");
        break;
    case 4:
        batchSize = inputDims[3];
    case 3:
        channel = inputDims[2];
    case 2:
        height = inputDims[0];
    case 1:
        width = inputDims[1];
    }

    //////////////////////// anchors /////////////////////
    //float anchors[MAX_ANCHOR_NUM*2];
    int numAnchors;
    int anchorNDims = mxGetNumberOfDimensions(prhs[2]);
    const size_t *anchorDims = mxGetDimensions(prhs[2]);
    numAnchors = anchorDims[0];
    if (anchorDims[1] != 2)
        mexErrMsgTxt("anchors should be nx2 array");
    double *dp = mxGetPr(prhs[2]);
    for (int i = 0; i < numAnchors; i++)
    {
        l.biases[2*i] = dp[i+numAnchors];   //w T ......
        l.biases[2*i+1] = dp[i]; //h T .....
        //l.biases[2*i] = dp[i];
        //l.biases[2*i+1] = dp[i+numAnchors];
    }

    ///////////////////////////////////////////////////////////
    int bias_match = (int)mxGetScalar(prhs[3]);
    int classes = (int)mxGetScalar(prhs[4]);
    int coords = (int)mxGetScalar(prhs[5]);
    int softmax_ = (int)mxGetScalar(prhs[6]);
    float jitter = (float)mxGetScalar(prhs[7]);
    int rescore = (int)mxGetScalar(prhs[8]);
    float object_scale = (float)mxGetScalar(prhs[9]);
    float noobject_scale = (float)mxGetScalar(prhs[10]);
    float class_scale = (float)mxGetScalar(prhs[11]);
    float coord_scale = (float)mxGetScalar(prhs[12]);
    int absolute = (int)mxGetScalar(prhs[13]);
    float thresh = (float)mxGetScalar(prhs[14]);
    int random = (int)mxGetScalar(prhs[15]);
    int train = (int)mxGetScalar(prhs[16]);
    int iters = (int)mxGetScalar(prhs[17]);


    l.batch = batchSize;
    l.c = channel;
    l.h = width; //transpose
    l.w = height; //transpose
    l.outputs = height*width*numAnchors*(classes + coords + 1);
    l.inputs = l.outputs;
    l.n = numAnchors;
    l.truths = MAX_GT_PER_IMAGE * (coords + 1);
    l.bias_match = bias_match;
    l.classes = classes;
    l.coords = coords;
    l.softmax = softmax_;
    l.jitter = jitter;
    l.rescore = rescore;
    l.object_scale = object_scale;
    l.noobject_scale = noobject_scale;
    l.class_scale = class_scale;
    l.coord_scale = coord_scale;
    l.absolute = absolute;
    l.thresh = thresh;
    l.random = random;
    l.mask_scale = 1; //default
    l.background = 0; //default

    /////////////////////////////// network//////////////////
    net.train = train;
    net.input = data;
    net.seen[0] = iters;
    
    ///////////////// ground truths ///////////////////////////
    int n = mxGetNumberOfElements(prhs[1]);
    if(n != batchSize)
    {
        mexErrMsgTxt("the number of gt does not match the number of feat");
    }
    
    net.truth = (float *)malloc(sizeof(float)*batchSize*l.truths);
    memset(net.truth, 0, sizeof(float)*batchSize*l.truths);
    for(int i = 0; i < batchSize; i++)
    {
        float *t = net.truth + i * l.truths;
        mxArray *gtArr = mxGetCell(prhs[1], i);
        double *dp = mxGetPr(gtArr);
        int ndims = mxGetNumberOfDimensions(gtArr);
        const size_t *dims = mxGetDimensions(gtArr);
        if(ndims != 2)
        {
            mexErrMsgTxt("gt of one image should dimension of 2");
        }
        if((dims[0] != 0 && dims[1] != 0) & dims[1] != 5)
        {
            mexErrMsgTxt("one gt should be [x0,y0,x1,y1,classId]");
        }
        int num = dims[0];
        for(int j = 0; j < num; j++)
        {
            t[j*5]   = dp[j + num]; // x T
            t[j*5+1] = dp[j + num*0]; // y T
            t[j*5+2] = dp[j + num*3]; // w T
            t[j*5+3] = dp[j + num*2]; // h T
            t[j*5+4] = dp[j + num*4]; // id
            
//             t[j*5]   = dp[j + num*0]; // x
//             t[j*5+1] = dp[j + num]; // y
//             t[j*5+2] = dp[j + num*2]; // w
//             t[j*5+3] = dp[j + num*3]; // h
//             t[j*5+4] = dp[j + num*4]; // id
        }
    }
    
    // print
//     printf("\n");
//     printf("input size: %d %d %d %d\n", l.batch, l.c, l.h, l.w);
//     printf("biases: ");
//     for(int i = 0; i < l.n; i++)
//     {
//         printf("%f %f  ", l.biases[2*i], l.biases[2*i+1]);
//     }
//     printf("\n");
//     
//     printf("ground truth:\n");
//     
//     for(int i = 0; i < l.batch; i++)
//     {
//         float *p = net.truth + i * l.truths;
//         printf("batch[%d]: ", i);
//         for(int j = 0; j < MAX_GT_PER_IMAGE; j++)
//         {
//              
//             if(!p[j*5])
//                 break;
//             printf("%.2f %.2f %.2f %.2f %.0f   ", p[5*j],
//                     p[5*j+1], p[5*j+2], p[5*j+3], p[5*j+4]);
//         }
//         printf("\n");
//     }
//     
//     printf("=====================================");

    mxArray *mxDeltaArr, *mxOutArr;
    size_t outDims[10];
    for(int i = 0; i < inputNDims; i++)
    {
        outDims[i] = inputDims[i];
    }
    mxDeltaArr = mxCreateUninitNumericArray(inputNDims, outDims, mxSINGLE_CLASS, mxREAL);
    if(mxDeltaArr == NULL)
    {
        mexErrMsgTxt("fail to malloc memory for delta(gradient)");
    }
    mxOutArr = mxCreateUninitNumericArray(inputNDims, outDims, mxSINGLE_CLASS, mxREAL);
    if(mxOutArr == NULL)
    {
        mexErrMsgTxt("fail to malloc memory for output");
    }
    
    
    plhs[0] = mxDeltaArr;
    plhs[1] = mxOutArr;
    
    l.delta = (float *)mxGetData(mxDeltaArr);
    l.output = (float *)mxGetData(mxOutArr);
    
    printf(" iters: %d ", iters);
    
    forward_region_layer(l, net);
    
    float norm = l.batch;
    if(iters < 10000)
    {
        //norm = norm * 4;
    }
    
    for(int i = 0; i < l.batch*l.outputs; i++)
    {
        l.delta[i] = -l.delta[i]/norm;
    }
    
    if(net.truth != NULL)
        free(net.truth);
    
}


///// junk
/*
void backward_region_layer(const layer l, network net)
{
    
    //int b;
    //int size = l.coords + l.classes + 1;
    //for (b = 0; b < l.batch*l.n; ++b){
    //int index = (b*size + 4)*l.w*l.h;
    //gradient_array(l.output + index, l.w*l.h, LOGISTIC, l.delta + index);
    //}
    //axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
    
}

void correct_region_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w = 0;
    int new_h = 0;
    if (((float)netw / w) < ((float)neth / h)) {
        new_w = netw;
        new_h = (h * netw) / w;
    }
    else {
        new_h = neth;
        new_w = (w * neth) / h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x = (b.x - (netw - new_w) / 2. / netw) / ((float)new_w / netw);
        b.y = (b.y - (neth - new_h) / 2. / neth) / ((float)new_h / neth);
        b.w *= (float)netw / new_w;
        b.h *= (float)neth / new_h;
        if (!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets)
{
    int i, j, n, z;
    float *predictions = l.output;
    if (l.batch == 2) {
        float *flip = l.output + l.outputs;
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w / 2; ++i) {
                for (n = 0; n < l.n; ++n) {
                    for (z = 0; z < l.classes + l.coords + 1; ++z){
                        int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                        int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                        float swap = flip[i1];
                        flip[i1] = flip[i2];
                        flip[i2] = swap;
                        if (z == 0){
                            flip[i1] = -flip[i1];
                            flip[i2] = -flip[i2];
                        }
                    }
                }
            }
        }
        for (i = 0; i < l.outputs; ++i){
            l.output[i] = (l.output[i] + flip[i]) / 2.;
        }
    }
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for (n = 0; n < l.n; ++n){
            int index = n*l.w*l.h + i;
            for (j = 0; j < l.classes; ++j){
                dets[index].prob[j] = 0;
            }
            int obj_index = entry_index(l, 0, n*l.w*l.h + i, l.coords);
            int box_index = entry_index(l, 0, n*l.w*l.h + i, 0);
            int mask_index = entry_index(l, 0, n*l.w*l.h + i, 4);
            float scale = l.background ? 1 : predictions[obj_index];
            dets[index].bbox = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h, l.w*l.h);
            dets[index].objectness = scale > thresh ? scale : 0;
            if (dets[index].mask){
                for (j = 0; j < l.coords - 4; ++j){
                    dets[index].mask[j] = l.output[mask_index + j*l.w*l.h];
                }
            }

            int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + !l.background);
            if (l.softmax_tree){

                hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0, l.w*l.h);
                if (map){
                    for (j = 0; j < 200; ++j){
                        int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + map[j]);
                        float prob = scale*predictions[class_index];
                        dets[index].prob[j] = (prob > thresh) ? prob : 0;
                    }
                }
                else {
                    int j = hierarchy_top_prediction(predictions + class_index, l.softmax_tree, tree_thresh, l.w*l.h);
                    dets[index].prob[j] = (scale > thresh) ? scale : 0;
                }
            }
            else {
                if (dets[index].objectness){
                    for (j = 0; j < l.classes; ++j){
                        int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + j);
                        float prob = scale*predictions[class_index];
                        dets[index].prob[j] = (prob > thresh) ? prob : 0;
                    }
                }
            }
        }
    }
    correct_region_boxes(dets, l.w*l.h*l.n, w, h, netw, neth, relative);
}

#ifdef GPU

void forward_region_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for (n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_gpu(l.output_gpu + index, 2 * l.w*l.h, LOGISTIC);
            if (l.coords > 4){
                index = entry_index(l, b, n*l.w*l.h, 4);
                activate_array_gpu(l.output_gpu + index, (l.coords - 4)*l.w*l.h, LOGISTIC);
            }
            index = entry_index(l, b, n*l.w*l.h, l.coords);
            if (!l.background) activate_array_gpu(l.output_gpu + index, l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, l.coords + 1);
            if (!l.softmax && !l.softmax_tree) activate_array_gpu(l.output_gpu + index, l.classes*l.w*l.h, LOGISTIC);
        }
    }
    if (l.softmax_tree){
        int index = entry_index(l, 0, 0, l.coords + 1);
        softmax_tree(net.input_gpu + index, l.w*l.h, l.batch*l.n, l.inputs / l.n, 1, l.output_gpu + index, *l.softmax_tree);
    }
    else if (l.softmax) {
        int index = entry_index(l, 0, 0, l.coords + !l.background);
        softmax_gpu(net.input_gpu + index, l.classes + l.background, l.batch*l.n, l.inputs / l.n, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu + index);
    }
    if (!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }

    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_region_layer(l, net);
    //cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
    if (!net.train) return;
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

void backward_region_layer_gpu(const layer l, network net)
{
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for (n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            gradient_array_gpu(l.output_gpu + index, 2 * l.w*l.h, LOGISTIC, l.delta_gpu + index);
            if (l.coords > 4){
                index = entry_index(l, b, n*l.w*l.h, 4);
                gradient_array_gpu(l.output_gpu + index, (l.coords - 4)*l.w*l.h, LOGISTIC, l.delta_gpu + index);
            }
            index = entry_index(l, b, n*l.w*l.h, l.coords);
            if (!l.background) gradient_array_gpu(l.output_gpu + index, l.w*l.h, LOGISTIC, l.delta_gpu + index);
        }
    }
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

void zero_objectness(layer l)
{
    int i, n;
    for (i = 0; i < l.w*l.h; ++i){
        for (n = 0; n < l.n; ++n){
            int obj_index = entry_index(l, 0, n*l.w*l.h + i, l.coords);
            l.output[obj_index] = 0;
        }
    }
}
*/