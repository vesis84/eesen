#!/usr/bin/env python

# Copyright 2015       Yajie Miao    (Carnegie Mellon University)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

import sys

def parse_arguments(arg_elements):
    args = {}
    arg_num = len(arg_elements) / 2
    for i in xrange(arg_num):
        key = arg_elements[2*i].replace("--","").replace("-", "_");
        args[key] = arg_elements[2*i+1]
    return args


if __name__ == '__main__':

    """
    Python script to generate the network topology. Parameters:
    ------------------
    --input-feat-dim : int
        Dimension of the input features
        Required.
    --lstm-layer-num : int
        Number of LSTM layers
        Required.
    --lstm-cell-dim : int
        Number of memory cells in LSTM. For the bi-directional case, this is the number of cells
        in either the forward or the backward sub-layer.
        Required.
    --target-num : int
        Number of labels as the targets
        Required.
    --param-range : float
        Range to randomly draw the initial values of model parameters. For example, setting it to
        0.1 means model parameters are drawn uniformly from [-0.1, 0.1]
        Optional. By default it is set to 0.1.
    --lstm-type : string
        Type of LSTMs. Optional. Either "bi" (bi-directional) or "uni" (uni-directional). By default,
        "bi" (bi-directional).
    --fgate-bias-init : float
        Initial value of the forget-gate bias. Not specifying this option means the forget-gate bias
        will be initialized randomly, in the same way as the other parameters.
    --input-dim : int
        Reduce the input feature to a given dimensionality before passing to the LSTM.
        Optional.
    --projection-dim : int
        Project the feature vector down to a given dimensionality between LSTM layers.
        Optional.

    """


    # parse arguments
    arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
    arguments = parse_arguments(arg_elements)

    # these 4 arguments are mandatory
    input_feat_dim=int(arguments['input_feat_dim'])
    lstm_layer_num=int(arguments['lstm_layer_num'])
    lstm_cell_dim=int(arguments['lstm_cell_dim'])
    target_num=int(arguments['target_num'])

    # by default, the range of the parameters is set to 0.1; however, you can change it by specifying "--param-range"
    # this means for initialization, model parameters are drawn uniformly from the interval [-0.1, 0.1]
    param_range='0.1'
    if arguments.has_key('param_range'):
        param_range = arguments['param_range']

    actual_cell_dim = 2*lstm_cell_dim
    model_type = '<BiLstmParallel>'   # by default
    if arguments.has_key('lstm_type') and arguments['lstm_type'] == 'uni':
        actual_cell_dim = lstm_cell_dim
        model_type = '<LstmParallel>'

    # process options for LSTM initialization,
    lstm_opts = ' <ParamRange> ' + param_range
    # add the option to set the initial value of the forget-gate bias
    if arguments.has_key('fgate_bias_init'):
        lstm_opts += ' <FgateBias> ' + arguments['fgate_bias_init']
    if arguments.has_key('learn_rate_coef'):
        lstm_opts += ' <LearnRateCoef> ' + arguments['learn_rate_coef']
    if arguments.has_key('bias_learn_rate_coef'):
        lstm_opts += ' <BiasLearnRateCoef> ' + arguments['bias_learn_rate_coef']
    if arguments.has_key('phole_learn_rate_coef'):
        lstm_opts += ' <PholeLearnRateCoef> ' + arguments['phole_learn_rate_coef']
    if arguments.has_key('grad_max_norm'):
        lstm_opts += ' <GradMaxNorm> ' + arguments['grad_max_norm']
    if arguments.has_key('grad_clip'):
        lstm_opts += ' <GradClip> ' + arguments['grad_clip']
    if arguments.has_key('diff_clip'):
        lstm_opts += ' <DiffClip> ' + arguments['diff_clip']
    if arguments.has_key('cell_clip'):
        lstm_opts += ' <CellClip> ' + arguments['cell_clip']
    if arguments.has_key('drop_factor'):
        lstm_opts += ' <DropFactor> ' + arguments['drop_factor']

    # add the option to specify projection layers
    if arguments.has_key('projection_dim'):
        proj_dim = arguments['projection_dim']
    else:
        proj_dim = 0

    # add the option to reduce the dimensionality of the input features
    if arguments.has_key('input_dim'):
        input_dim = arguments['input_dim']
    else:
        input_dim = 0

    # pre-amble
    print '<Nnet>'

    # optional dimensionality reduction layer on input,
    if input_dim > 0:
        print '<AffineTransform> <InputDim> ' + str(input_feat_dim) + ' <OutputDim> ' + str(input_dim) + ' <ParamRange> ' + param_range
        input_feat_dim = input_dim

    # the first layer takes input features
    print model_type + ' <InputDim> ' + str(input_feat_dim) + ' <CellDim> ' + str(actual_cell_dim) + lstm_opts
    # the following bidirectional LSTM layers
    for n in range(1, lstm_layer_num):
        if proj_dim > 0:
            print '<AffineTransform> <InputDim> ' + str(actual_cell_dim) + ' <OutputDim> ' + str(proj_dim) + ' <ParamRange> ' + param_range
            print model_type + ' <InputDim> ' +        str(proj_dim) + ' <CellDim> ' + str(actual_cell_dim) + lstm_opts
        else:
            print model_type + ' <InputDim> ' + str(actual_cell_dim) + ' <CellDim> ' + str(actual_cell_dim) + lstm_opts

    # process options for the last affine transform,
    last_affine_opts=''
    if arguments.has_key('softmax_bias_learn_rate_coef'):
      last_affine_opts += ' <BiasLearnRateCoef> ' + arguments['softmax_bias_learn_rate_coef']

    # the final affine-transform and softmax layer
    print '<AffineTransform> <InputDim> ' + str(actual_cell_dim) + ' <OutputDim> ' + str(target_num) + ' <ParamRange> ' + param_range + last_affine_opts
    print '<Softmax> <InputDim> ' + str(target_num) + ' <OutputDim> ' + str(target_num)
    print '</Nnet>'
