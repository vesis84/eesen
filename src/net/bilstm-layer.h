// net/bilstm-layer.h

// Copyright 2015   Yajie Miao, Hang Su
// Copyright 2016   Brno University of Technology (author: Karel Vesely)

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef EESEN_BILSTM_LAYER_H_
#define EESEN_BILSTM_LAYER_H_

#include "net/layer.h"
#include "net/trainable-layer.h"
#include "net/utils-functions.h"
#include "gpucompute/cuda-math.h"

namespace eesen {

class BiLstm : public TrainableLayer {
public:
    BiLstm(int32 input_dim, int32 output_dim):
        TrainableLayer(input_dim, output_dim),
        cell_dim_(output_dim / 2),
        learn_rate_coef_(1.0),
        bias_learn_rate_coef_(1.0),
        phole_learn_rate_coef_(1.0),
        grad_max_norm_(0.0),
        grad_clip_(0.0),
        diff_clip_(1.0),
        cell_clip_(50.0),
        drop_factor_(0.0)
    { }

    ~BiLstm()
    { }

    Layer* Copy() const { return new BiLstm(*this); }
    LayerType GetType() const { return l_BiLstm; }
    LayerType GetTypeNonParal() const { return l_BiLstm; }

    void SetDropFactor(float drop_factor) {
      drop_factor_ = drop_factor;
    }

    void InitData(std::istream &is) {
      // define options
      float param_range = 0.1;
      float fgate_bias_init = 0.0;   // the initial value for the bias of the forget gates
      // parse config
      std::string token;
      while (is >> std::ws, !is.eof()) {
        ReadToken(is, false, &token);
        if (token == "<ParamRange>")  ReadBasicType(is, false, &param_range);
        else if (token == "<FgateBias>") ReadBasicType(is, false, &fgate_bias_init);
        else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef_);
        else if (token == "<BiasLearnRateCoef>") ReadBasicType(is, false, &bias_learn_rate_coef_);
        else if (token == "<PholeLearnRateCoef>") ReadBasicType(is, false, &phole_learn_rate_coef_);
        else if (token == "<GradMaxNorm>") ReadBasicType(is, false, &grad_max_norm_);
        else if (token == "<GradClip>") ReadBasicType(is, false, &grad_clip_);
        else if (token == "<DiffClip>") ReadBasicType(is, false, &diff_clip_);
        else if (token == "<CellClip>") ReadBasicType(is, false, &cell_clip_);
        else if (token == "<DropFactor>") ReadBasicType(is, false, &drop_factor_);
        else KALDI_ERR << "Unknown token " << token << ", a typo in config? "
                       << "(ParamRange|FgateBias|LearnRateCoef|BiasLearnRateCoef|PholeLearnRateCoef|"
                       << "GradClip|GradMaxNorm|CellClip|DropFactor)";
      }

      // initialize weights and biases for the forward sub-layer
      wei_gifo_x_fw_.Resize(4 * cell_dim_, input_dim_); wei_gifo_x_fw_.InitRandUniform(param_range);
      // the weights connecting momory cell outputs with the units/gates
      wei_gifo_m_fw_.Resize(4 * cell_dim_, cell_dim_);  wei_gifo_m_fw_.InitRandUniform(param_range);
      // the bias for the units/gates
      bias_fw_.Resize(4 * cell_dim_); bias_fw_.InitRandUniform(param_range);
      if (fgate_bias_init != 0.0) {   // reset the bias of the forget gates
        bias_fw_.Range(2 * cell_dim_, cell_dim_).Set(fgate_bias_init);
      }
      // peephole connections for i, f, and o, with diagonal matrices (vectors)
      phole_i_c_fw_.Resize(cell_dim_); phole_i_c_fw_.InitRandUniform(param_range);
      phole_f_c_fw_.Resize(cell_dim_); phole_f_c_fw_.InitRandUniform(param_range);
      phole_o_c_fw_.Resize(cell_dim_); phole_o_c_fw_.InitRandUniform(param_range);

      // initialize weights and biases for the backward sub-layer
      wei_gifo_x_bw_.Resize(4 * cell_dim_, input_dim_); wei_gifo_x_bw_.InitRandUniform(param_range);
      wei_gifo_m_bw_.Resize(4 * cell_dim_, cell_dim_);  wei_gifo_m_bw_.InitRandUniform(param_range);
      bias_bw_.Resize(4 * cell_dim_); bias_bw_.InitRandUniform(param_range);
      if (fgate_bias_init != 0.0) {   // reset the bias of the forget gates
        bias_bw_.Range(2 * cell_dim_, cell_dim_).Set(fgate_bias_init);
      }

      phole_i_c_bw_.Resize(cell_dim_); phole_i_c_bw_.InitRandUniform(param_range);
      phole_f_c_bw_.Resize(cell_dim_); phole_f_c_bw_.InitRandUniform(param_range);
      phole_o_c_bw_.Resize(cell_dim_); phole_o_c_bw_.InitRandUniform(param_range);
    }

    void ReadData(std::istream &is, bool binary) {
      while ('<' == Peek(is, binary)) {
        std::string token;
        int first_char = PeekToken(is, binary);
        switch (first_char) {
          case 'L': ExpectToken(is, binary, "<LearnRateCoef>");
            ReadBasicType(is, binary, &learn_rate_coef_);
            break;
          case 'B': ExpectToken(is, binary, "<BiasLearnRateCoef>");
            ReadBasicType(is, binary, &bias_learn_rate_coef_);
            break;
          case 'P': ExpectToken(is, binary, "<PeepholeLearnRateCoef>");
            ReadBasicType(is, binary, &phole_learn_rate_coef_);
            break;
          case 'G': ReadToken(is, false, &token);
            /**/ if (token == "<GradClip>") ReadBasicType(is, binary, &grad_clip_);
            else if (token == "<GradMaxNorm>") ReadBasicType(is, binary, &grad_max_norm_);
            else KALDI_ERR << "Unknown token: " << token;
            break;
          // <MaxGrad> as alias of <GradClip> (backward compatibility),
          case 'M': ExpectToken(is, binary, "<MaxGrad>");
            ReadBasicType(is, binary, &grad_clip_);
            break;
          case 'C': ExpectToken(is, binary, "<CellClip>");
            ReadBasicType(is, binary, &cell_clip_);
            break;
          case 'D': ReadToken(is, false, &token);
            /**/ if (token == "<DiffClip>") ReadBasicType(is, binary, &diff_clip_);
            else if (token == "<DropFactor>") ReadBasicType(is, binary, &drop_factor_);
            else KALDI_ERR << "Unknown token: " << token;
            break;
          default:
            ReadToken(is, false, &token);
            KALDI_ERR << "Unknown token: " << token;
        }
      }

      // read parameters of forward layer
      wei_gifo_x_fw_.Read(is, binary);
      wei_gifo_m_fw_.Read(is, binary);
      bias_fw_.Read(is, binary);
      phole_i_c_fw_.Read(is, binary);
      phole_f_c_fw_.Read(is, binary);
      phole_o_c_fw_.Read(is, binary);
      // initialize the buffer for gradients updates
      wei_gifo_x_fw_corr_ = wei_gifo_x_fw_; wei_gifo_x_fw_corr_.SetZero();
      wei_gifo_m_fw_corr_ = wei_gifo_m_fw_; wei_gifo_m_fw_corr_.SetZero();
      bias_fw_corr_ = bias_fw_; bias_fw_corr_.SetZero();
      phole_i_c_fw_corr_ = phole_i_c_fw_; phole_i_c_fw_corr_.SetZero();
      phole_f_c_fw_corr_ = phole_f_c_fw_; phole_f_c_fw_corr_.SetZero();
      phole_o_c_fw_corr_ = phole_o_c_fw_; phole_o_c_fw_corr_.SetZero();

      // read parameters of backward layer
      wei_gifo_x_bw_.Read(is, binary);
      wei_gifo_m_bw_.Read(is, binary);
      bias_bw_.Read(is, binary);
      phole_i_c_bw_.Read(is, binary);
      phole_f_c_bw_.Read(is, binary);
      phole_o_c_bw_.Read(is, binary);
      // initialize the buffer for gradients updates
      wei_gifo_x_bw_corr_ = wei_gifo_x_bw_; wei_gifo_x_bw_corr_.SetZero();
      wei_gifo_m_bw_corr_ = wei_gifo_m_bw_; wei_gifo_m_bw_corr_.SetZero();
      bias_bw_corr_ = bias_bw_; bias_bw_corr_.SetZero();
      phole_i_c_bw_corr_ = phole_i_c_bw_; phole_i_c_bw_corr_.SetZero();
      phole_f_c_bw_corr_ = phole_f_c_bw_; phole_f_c_bw_corr_.SetZero();
      phole_o_c_bw_corr_ = phole_o_c_bw_; phole_o_c_bw_corr_.SetZero();
    }

    void WriteData(std::ostream &os, bool binary) const {
      WriteToken(os, binary, "<LearnRateCoef>");
      WriteBasicType(os, binary, learn_rate_coef_);
      WriteToken(os, binary, "<BiasLearnRateCoef>");
      WriteBasicType(os, binary, bias_learn_rate_coef_);
      WriteToken(os, binary, "<PeepholeLearnRateCoef>");
      WriteBasicType(os, binary, phole_learn_rate_coef_);
      WriteToken(os, binary, "<GradMaxNorm>");
      WriteBasicType(os, binary, grad_max_norm_);
      WriteToken(os, binary, "<GradClip>");
      WriteBasicType(os, binary, grad_clip_);
      WriteToken(os, binary, "<DiffClip>");
      WriteBasicType(os, binary, diff_clip_);
      WriteToken(os, binary, "<CellClip>");
      WriteBasicType(os, binary, cell_clip_);
      WriteToken(os, binary, "<DropFactor>");
      WriteBasicType(os, binary, drop_factor_);

      // write parameters of the forward layer
      wei_gifo_x_fw_.Write(os, binary);
      wei_gifo_m_fw_.Write(os, binary);
      bias_fw_.Write(os, binary);
      phole_i_c_fw_.Write(os, binary);
      phole_f_c_fw_.Write(os, binary);
      phole_o_c_fw_.Write(os, binary);

      // write parameters of the backward layer
      wei_gifo_x_bw_.Write(os, binary);
      wei_gifo_m_bw_.Write(os, binary);
      bias_bw_.Write(os, binary);
      phole_i_c_bw_.Write(os, binary);
      phole_f_c_bw_.Write(os, binary);
      phole_o_c_bw_.Write(os, binary);
    }

    // print statistics of the parameters
    std::string Info() const {
        return std::string("    ") +
            "\n  wei_gifo_x_fw_  "   + MomentStatistics(wei_gifo_x_fw_) +
            "\n  wei_gifo_m_fw_  "   + MomentStatistics(wei_gifo_m_fw_) +
            "\n  bias_fw_  "         + MomentStatistics(bias_fw_) +
            "\n  phole_i_c_fw_  "      + MomentStatistics(phole_i_c_fw_) +
            "\n  phole_f_c_fw_  "      + MomentStatistics(phole_f_c_fw_) +
            "\n  phole_o_c_fw_  "      + MomentStatistics(phole_o_c_fw_) +
            "\n  ---" +
            "\n  wei_gifo_x_bw_  "   + MomentStatistics(wei_gifo_x_bw_) +
            "\n  wei_gifo_m_bw_  "   + MomentStatistics(wei_gifo_m_bw_) +
            "\n  bias_bw_  "         + MomentStatistics(bias_bw_) +
            "\n  phole_i_c_bw_  "      + MomentStatistics(phole_i_c_bw_) +
            "\n  phole_f_c_bw_  "      + MomentStatistics(phole_f_c_bw_) +
            "\n  phole_o_c_bw_  "      + MomentStatistics(phole_o_c_bw_);
    }

    // print statistics of the gradients buffer
    std::string InfoGradient() const {
        CuSubMatrix<BaseFloat> YG_FW(propagate_buf_fw_.ColRange(0, cell_dim_));
        CuSubMatrix<BaseFloat> YI_FW(propagate_buf_fw_.ColRange(1 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> YF_FW(propagate_buf_fw_.ColRange(2 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> YO_FW(propagate_buf_fw_.ColRange(3 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> YC_FW(propagate_buf_fw_.ColRange(4 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> YH_FW(propagate_buf_fw_.ColRange(5 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> YM_FW(propagate_buf_fw_.ColRange(6 * cell_dim_, cell_dim_));

        CuSubMatrix<BaseFloat> YG_BW(propagate_buf_bw_.ColRange(0, cell_dim_));
        CuSubMatrix<BaseFloat> YI_BW(propagate_buf_bw_.ColRange(1 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> YF_BW(propagate_buf_bw_.ColRange(2 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> YO_BW(propagate_buf_bw_.ColRange(3 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> YC_BW(propagate_buf_bw_.ColRange(4 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> YH_BW(propagate_buf_bw_.ColRange(5 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> YM_BW(propagate_buf_bw_.ColRange(6 * cell_dim_, cell_dim_));

        CuSubMatrix<BaseFloat> DG_FW(backpropagate_buf_fw_.ColRange(0, cell_dim_));
        CuSubMatrix<BaseFloat> DI_FW(backpropagate_buf_fw_.ColRange(1 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> DF_FW(backpropagate_buf_fw_.ColRange(2 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> DO_FW(backpropagate_buf_fw_.ColRange(3 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> DC_FW(backpropagate_buf_fw_.ColRange(4 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> DH_FW(backpropagate_buf_fw_.ColRange(5 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> DM_FW(backpropagate_buf_fw_.ColRange(6 * cell_dim_, cell_dim_));

        CuSubMatrix<BaseFloat> DG_BW(backpropagate_buf_bw_.ColRange(0, cell_dim_));
        CuSubMatrix<BaseFloat> DI_BW(backpropagate_buf_bw_.ColRange(1 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> DF_BW(backpropagate_buf_bw_.ColRange(2 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> DO_BW(backpropagate_buf_bw_.ColRange(3 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> DC_BW(backpropagate_buf_bw_.ColRange(4 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> DH_BW(backpropagate_buf_bw_.ColRange(5 * cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> DM_BW(backpropagate_buf_bw_.ColRange(6 * cell_dim_, cell_dim_));

        BaseFloat grad_l2_norm = std::sqrt(
          std::pow(wei_gifo_x_fw_corr_.FrobeniusNorm(), 2.0) +
          std::pow(wei_gifo_m_fw_corr_.FrobeniusNorm(), 2.0) +
          std::pow(bias_fw_corr_.Norm(2), 2.0) +
          std::pow(phole_i_c_fw_corr_.Norm(2), 2.0) +
          std::pow(phole_f_c_fw_corr_.Norm(2), 2.0) +
          std::pow(phole_o_c_fw_corr_.Norm(2), 2.0) +

          std::pow(wei_gifo_x_bw_corr_.FrobeniusNorm(), 2.0) +
          std::pow(wei_gifo_m_bw_corr_.FrobeniusNorm(), 2.0) +
          std::pow(bias_bw_corr_.Norm(2), 2.0) +
          std::pow(phole_i_c_bw_corr_.Norm(2), 2.0) +
          std::pow(phole_f_c_bw_corr_.Norm(2), 2.0) +
          std::pow(phole_o_c_bw_corr_.Norm(2), 2.0)
        );

        return std::string("") +
          "( learn_rate_coef_ " + ToString(learn_rate_coef_) +
          ", bias_learn_rate_coef_ " + ToString(bias_learn_rate_coef_) +
          ", phole_learn_rate_coef_ " + ToString(phole_learn_rate_coef_) +
          ", grad_max_norm_ " + ToString(grad_max_norm_) +
          ", grad_clip_ " + ToString(grad_clip_) +
          ", diff_clip_ " + ToString(diff_clip_) +
          ", cell_clip_ " + ToString(cell_clip_) + " )" +
          "\n  ### Gradients " +
          "\n  wei_gifo_x_fw_corr_ "   + MomentStatistics(wei_gifo_x_fw_corr_) +
          "\n  wei_gifo_m_fw_corr_ "   + MomentStatistics(wei_gifo_m_fw_corr_) +
          "\n  bias_fw_corr_       "   + MomentStatistics(bias_fw_corr_) +
          "\n  phole_i_c_fw_corr_  "     + MomentStatistics(phole_i_c_fw_corr_) +
          "\n  phole_f_c_fw_corr_  "     + MomentStatistics(phole_f_c_fw_corr_) +
          "\n  phole_o_c_fw_corr_  "     + MomentStatistics(phole_o_c_fw_corr_) +
          "\n  ---" +
          "\n  wei_gifo_x_bw_corr_ "   + MomentStatistics(wei_gifo_x_bw_corr_) +
          "\n  wei_gifo_m_bw_corr_ "   + MomentStatistics(wei_gifo_m_bw_corr_) +
          "\n  bias_bw_corr_       "   + MomentStatistics(bias_bw_corr_) +
          "\n  phole_i_c_bw_corr_  "     + MomentStatistics(phole_i_c_bw_corr_) +
          "\n  phole_f_c_bw_corr_  "     + MomentStatistics(phole_f_c_bw_corr_) +
          "\n  phole_o_c_bw_corr_  "     + MomentStatistics(phole_o_c_bw_corr_) +
          "\n  --- grad L2-norm: " + ToString(grad_l2_norm) +
          "\n" +
          "\n  ### Activations (mostly after non-linearities)" +
          "\n  YI_FW(0..1)^  " + MomentStatistics(YI_FW) +
          "\n  YF_FW(0..1)^  " + MomentStatistics(YF_FW) +
          "\n  YO_FW(0..1)^  " + MomentStatistics(YO_FW) +
          "\n  YG_FW(-1..1)  " + MomentStatistics(YG_FW) +
          "\n  YC_FW(-R..R)* " + MomentStatistics(YC_FW) +
          "\n  YH_FW(-1..1)  " + MomentStatistics(YH_FW) +
          "\n  YM_FW(-1..1)  " + MomentStatistics(YM_FW) +
          "\n  ---" +
          "\n  YI_BW(0..1)^  " + MomentStatistics(YI_BW) +
          "\n  YF_BW(0..1)^  " + MomentStatistics(YF_BW) +
          "\n  YO_BW(0..1)^  " + MomentStatistics(YO_BW) +
          "\n  YG_BW(-1..1)  " + MomentStatistics(YG_BW) +
          "\n  YC_BW(-R..R)* " + MomentStatistics(YC_BW) +
          "\n  YH_BW(-1..1)  " + MomentStatistics(YH_BW) +
          "\n  YM_BW(-1..1)  " + MomentStatistics(YM_BW) +
          "\n" +
          "\n  ### Derivatives (w.r.t. inputs of non-linearities)" +
          "\n  DI_FW^ " + MomentStatistics(DI_FW) +
          "\n  DF_FW^ " + MomentStatistics(DF_FW) +
          "\n  DO_FW^ " + MomentStatistics(DO_FW) +
          "\n  DG_FW  " + MomentStatistics(DG_FW) +
          "\n  DC_FW* " + MomentStatistics(DC_FW) +
          "\n  DH_FW  " + MomentStatistics(DH_FW) +
          "\n  DM_FW  " + MomentStatistics(DM_FW) +
          "\n  ---" +
          "\n  DI_BW^ " + MomentStatistics(DI_BW) +
          "\n  DF_BW^ " + MomentStatistics(DF_BW) +
          "\n  DO_BW^ " + MomentStatistics(DO_BW) +
          "\n  DG_BW  " + MomentStatistics(DG_BW) +
          "\n  DC_BW* " + MomentStatistics(DC_BW) +
          "\n  DH_BW  " + MomentStatistics(DH_BW) +
          "\n  DM_BW  " + MomentStatistics(DM_BW);
    }

    // the feedforward pass
    void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
        int32 T = in.NumRows();  // total number of frames
        // resize & clear propagation buffers for the forward sub-layer. [0] - the initial states with all the values to be 0
        // [1, T] - correspond to the inputs  [T+1] - not used; for alignment with the backward layer
        propagate_buf_fw_.Resize(T + 2, 7 * cell_dim_, kSetZero);
        // resize & clear propagation buffers for the backward sub-layer
        propagate_buf_bw_.Resize(T + 2, 7 * cell_dim_, kSetZero);

        KALDI_ASSERT(propagate_buf_fw_.Sum() == 0.0);
        KALDI_ASSERT(propagate_buf_bw_.Sum() == 0.0);

        // the forward layer
        if (1) {
          CuSubMatrix<BaseFloat> YG(propagate_buf_fw_.ColRange(0, cell_dim_));
          CuSubMatrix<BaseFloat> YI(propagate_buf_fw_.ColRange(1 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YF(propagate_buf_fw_.ColRange(2 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YO(propagate_buf_fw_.ColRange(3 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YC(propagate_buf_fw_.ColRange(4 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YH(propagate_buf_fw_.ColRange(5 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YM(propagate_buf_fw_.ColRange(6 * cell_dim_, cell_dim_));

          CuSubMatrix<BaseFloat> YGIFO(propagate_buf_fw_.ColRange(0, 4 * cell_dim_));
          // no recurrence involved in the inputs
          YGIFO.RowRange(1,T).AddMatMat(1.0, in, kNoTrans, wei_gifo_x_fw_, kTrans, 0.0);
          YGIFO.RowRange(1,T).AddVecToRows(1.0, bias_fw_);

          for (int t = 1; t <= T; t++) {
            // variables representing invidivual units/gates. we additionally use the Matrix forms
            // because we want to take advantage of the Mat.Sigmoid and Mat.Tanh function.
            CuSubVector<BaseFloat> y_g(YG.Row(t));  CuSubMatrix<BaseFloat> YG_t(YG.RowRange(t,1));
            CuSubVector<BaseFloat> y_i(YI.Row(t));  CuSubMatrix<BaseFloat> YI_t(YI.RowRange(t,1));
            CuSubVector<BaseFloat> y_f(YF.Row(t));  CuSubMatrix<BaseFloat> YF_t(YF.RowRange(t,1));
            CuSubVector<BaseFloat> y_o(YO.Row(t));  CuSubMatrix<BaseFloat> YO_t(YO.RowRange(t,1));
            CuSubVector<BaseFloat> y_c(YC.Row(t));  CuSubMatrix<BaseFloat> YC_t(YC.RowRange(t,1));
            CuSubVector<BaseFloat> y_h(YH.Row(t));  CuSubMatrix<BaseFloat> YH_t(YH.RowRange(t,1));
            CuSubVector<BaseFloat> y_m(YM.Row(t));  CuSubMatrix<BaseFloat> YM_t(YM.RowRange(t,1));
            CuSubVector<BaseFloat> y_gifo(YGIFO.Row(t));
            // add the recurrence of the previous memory cell to various gates/units
            y_gifo.AddMatVec(1.0, wei_gifo_m_fw_, kNoTrans, YM.Row(t-1), 1.0);
            // input gate
            y_i.AddVecVec(1.0, phole_i_c_fw_, YC.Row(t-1), 1.0);
            // forget gate
            y_f.AddVecVec(1.0, phole_f_c_fw_, YC.Row(t-1), 1.0);
            // apply sigmoid/tanh functionis to squash the outputs
            YI_t.Sigmoid(YI_t);
            YF_t.Sigmoid(YF_t);
            YG_t.Tanh(YG_t);

            // memory cell
            y_c.AddVecVec(1.0, y_i, y_g, 0.0);
            y_c.AddVecVec(1.0, y_f, YC.Row(t-1), 1.0);
            // clipping the activation in memory cell 'c',
            if (cell_clip_ > 0.0) {
              YC_t.ApplyFloor(-cell_clip_);
              YC_t.ApplyCeiling(cell_clip_);
            }
            // h - the tanh-squashed version of c
            YH_t.Tanh(YC_t);

            // output gate
            y_o.AddVecVec(1.0, phole_o_c_fw_, y_c, 1.0);
            YO_t.Sigmoid(YO_t);

            // finally the outputs
            y_m.AddVecVec(1.0, y_o, y_h, 0.0);
          }  // end of loop t
        }  // end of the forward layer

        // the backward layer; follows the same procedures, but iterates from t=T to t=1
        if (1) {
          CuSubMatrix<BaseFloat> YG(propagate_buf_bw_.ColRange(0, cell_dim_));
          CuSubMatrix<BaseFloat> YI(propagate_buf_bw_.ColRange(1 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YF(propagate_buf_bw_.ColRange(2 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YO(propagate_buf_bw_.ColRange(3 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YC(propagate_buf_bw_.ColRange(4 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YH(propagate_buf_bw_.ColRange(5 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> YM(propagate_buf_bw_.ColRange(6 * cell_dim_, cell_dim_));

          CuSubMatrix<BaseFloat> YGIFO(propagate_buf_bw_.ColRange(0, 4 * cell_dim_));
          YGIFO.RowRange(1,T).AddMatMat(1.0, in, kNoTrans, wei_gifo_x_bw_, kTrans, 0.0);
          YGIFO.RowRange(1,T).AddVecToRows(1.0, bias_bw_);

          for (int t = T; t >= 1; t--) {
            CuSubVector<BaseFloat> y_g(YG.Row(t));  CuSubMatrix<BaseFloat> YG_t(YG.RowRange(t,1));
            CuSubVector<BaseFloat> y_i(YI.Row(t));  CuSubMatrix<BaseFloat> YI_t(YI.RowRange(t,1));
            CuSubVector<BaseFloat> y_f(YF.Row(t));  CuSubMatrix<BaseFloat> YF_t(YF.RowRange(t,1));
            CuSubVector<BaseFloat> y_o(YO.Row(t));  CuSubMatrix<BaseFloat> YO_t(YO.RowRange(t,1));
            CuSubVector<BaseFloat> y_c(YC.Row(t));  CuSubMatrix<BaseFloat> YC_t(YC.RowRange(t,1));
            CuSubVector<BaseFloat> y_h(YH.Row(t));  CuSubMatrix<BaseFloat> YH_t(YH.RowRange(t,1));
            CuSubVector<BaseFloat> y_m(YM.Row(t));  CuSubMatrix<BaseFloat> YM_t(YM.RowRange(t,1));
            CuSubVector<BaseFloat> y_gifo(YGIFO.Row(t));

            y_gifo.AddMatVec(1.0, wei_gifo_m_bw_, kNoTrans, YM.Row(t+1), 1.0);
            // input gate
            y_i.AddVecVec(1.0, phole_i_c_bw_, YC.Row(t+1), 1.0);
            // forget gate
            y_f.AddVecVec(1.0, phole_f_c_bw_, YC.Row(t+1), 1.0);
            // apply sigmoid/tanh function
            YI_t.Sigmoid(YI_t);
            YF_t.Sigmoid(YF_t);
            YG_t.Tanh(YG_t);

            // memory cell
            y_c.AddVecVec(1.0, y_i, y_g, 0.0);
            y_c.AddVecVec(1.0, y_f, YC.Row(t+1), 1.0);
            // clipping the activation in memory cell 'c',
            if (cell_clip_ > 0.0) {
              YC_t.ApplyFloor(-cell_clip_);
              YC_t.ApplyCeiling(cell_clip_);
            }
            // h, the tanh-squashed version of c
            YH_t.Tanh(YC_t);

            // output gate
            y_o.AddVecVec(1.0, phole_o_c_bw_, y_c, 1.0);
            YO_t.Sigmoid(YO_t);

            // the final output
            y_m.AddVecVec(1.0, y_o, y_h, 0.0);
          }
        }
        // final outputs now become the concatenation of the foward and backward activations
        CuMatrix<BaseFloat> YM_RB;
        YM_RB.Resize(T+2, 2 * cell_dim_, kSetZero);
        YM_RB.ColRange(0, cell_dim_).CopyFromMat(propagate_buf_fw_.ColRange(6 * cell_dim_, cell_dim_));
        YM_RB.ColRange(cell_dim_, cell_dim_).CopyFromMat(propagate_buf_bw_.ColRange(6* cell_dim_, cell_dim_));

        out->CopyFromMat(YM_RB.RowRange(1,T));
    }

    // the back-propagation pass
    void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                          const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
        int32 T = in.NumRows();
        // initialize the back-propagation buffer
        backpropagate_buf_fw_.Resize(T + 2, 7 * cell_dim_, kSetZero);
        backpropagate_buf_bw_.Resize(T + 2, 7 * cell_dim_, kSetZero);

        if (1) {
          // get the activations of the gates/units from the feedforward buffer; these variabiles will be used
          // in gradients computation
          const CuSubMatrix<BaseFloat> YG(propagate_buf_fw_.ColRange(0, cell_dim_));
          const CuSubMatrix<BaseFloat> YI(propagate_buf_fw_.ColRange(1 * cell_dim_, cell_dim_));
          const CuSubMatrix<BaseFloat> YF(propagate_buf_fw_.ColRange(2 * cell_dim_, cell_dim_));
          const CuSubMatrix<BaseFloat> YO(propagate_buf_fw_.ColRange(3 * cell_dim_, cell_dim_));
          const CuSubMatrix<BaseFloat> YC(propagate_buf_fw_.ColRange(4 * cell_dim_, cell_dim_));
          const CuSubMatrix<BaseFloat> YH(propagate_buf_fw_.ColRange(5 * cell_dim_, cell_dim_));
          const CuSubMatrix<BaseFloat> YM(propagate_buf_fw_.ColRange(6 * cell_dim_, cell_dim_));

          // errors back-propagated to individual gates/units
          CuSubMatrix<BaseFloat> DG(backpropagate_buf_fw_.ColRange(0, cell_dim_));
          CuSubMatrix<BaseFloat> DI(backpropagate_buf_fw_.ColRange(1 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> DF(backpropagate_buf_fw_.ColRange(2 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> DO(backpropagate_buf_fw_.ColRange(3 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> DC(backpropagate_buf_fw_.ColRange(4 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> DH(backpropagate_buf_fw_.ColRange(5 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> DM(backpropagate_buf_fw_.ColRange(6 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> DGIFO(backpropagate_buf_fw_.ColRange(0, 4 * cell_dim_));

          // assume that the fist half of out_diff is about the forward layer
          DM.RowRange(1,T).CopyFromMat(out_diff.ColRange(0, cell_dim_));

          for (int t = T; t >= 1; t--) {
            // variables representing activations of invidivual units/gates
            const CuSubVector<BaseFloat> y_g(YG.Row(t));  const CuSubMatrix<BaseFloat> YG_t(YG.RowRange(t,1));
            const CuSubVector<BaseFloat> y_i(YI.Row(t));  const CuSubMatrix<BaseFloat> YI_t(YI.RowRange(t,1));
            const CuSubVector<BaseFloat> y_f(YF.Row(t));  const CuSubMatrix<BaseFloat> YF_t(YF.RowRange(t,1));
            const CuSubVector<BaseFloat> y_o(YO.Row(t));  const CuSubMatrix<BaseFloat> YO_t(YO.RowRange(t,1));
            const CuSubVector<BaseFloat> y_c(YC.Row(t));  const CuSubMatrix<BaseFloat> YC_t(YC.RowRange(t,1));
            const CuSubVector<BaseFloat> y_h(YH.Row(t));  const CuSubMatrix<BaseFloat> YH_t(YH.RowRange(t,1));
            const CuSubVector<BaseFloat> y_m(YM.Row(t));  const CuSubMatrix<BaseFloat> YM_t(YM.RowRange(t,1));
            // variables representing errors of invidivual units/gates
            CuSubVector<BaseFloat> d_all(backpropagate_buf_fw_.Row(t));
            CuSubVector<BaseFloat> d_g(DG.Row(t));  CuSubMatrix<BaseFloat> DG_t(DG.RowRange(t,1));
            CuSubVector<BaseFloat> d_i(DI.Row(t));  CuSubMatrix<BaseFloat> DI_t(DI.RowRange(t,1));
            CuSubVector<BaseFloat> d_f(DF.Row(t));  CuSubMatrix<BaseFloat> DF_t(DF.RowRange(t,1));
            CuSubVector<BaseFloat> d_o(DO.Row(t));  CuSubMatrix<BaseFloat> DO_t(DO.RowRange(t,1));
            CuSubVector<BaseFloat> d_c(DC.Row(t));  CuSubMatrix<BaseFloat> DC_t(DC.RowRange(t,1));
            CuSubVector<BaseFloat> d_h(DH.Row(t));  CuSubMatrix<BaseFloat> DH_t(DH.RowRange(t,1));
            CuSubVector<BaseFloat> d_m(DM.Row(t));  CuSubMatrix<BaseFloat> DM_t(DM.RowRange(t,1));

            // d_m comes from two parts: errors from the upper layer and errors from the following frame (t+1)
            d_m.AddMatVec(1.0, wei_gifo_m_fw_, kTrans, DGIFO.Row(t+1), 1.0);

            // d_h
            d_h.AddVecVec(1.0, y_o, d_m, 0.0);
            DH_t.DiffTanh(YH_t, DH_t);

            // d_o - output gate
            d_o.AddVecVec(1.0, y_h, d_m, 0.0);
            DO_t.DiffSigmoid(YO_t, DO_t);

            // d_c - memory cell
            d_c.AddVec(1.0, d_h, 0.0);
            d_c.AddVecVec(1.0, phole_o_c_fw_, d_o, 1.0);
            d_c.AddVecVec(1.0, YF.Row(t+1), DC.Row(t+1), 1.0);
            d_c.AddVecVec(1.0, phole_f_c_fw_, DF.Row(t+1), 1.0);
            d_c.AddVecVec(1.0, phole_i_c_fw_, DI.Row(t+1), 1.0);

            // d_f - forge gate
            d_f.AddVecVec(1.0, YC.Row(t-1), d_c, 0.0);
            DF_t.DiffSigmoid(YF_t, DF_t);

            // d_i - input gate
            d_i.AddVecVec(1.0, y_g, d_c, 0.0);
            DI_t.DiffSigmoid(YI_t, DI_t);

            // d_g
            d_g.AddVecVec(1.0, y_i, d_c, 0.0);
            DG_t.DiffTanh(YG_t, DG_t);

            // clipping of per-frame derivatives (as done in Google),
            if (diff_clip_ > 0.0) {
              d_all.ApplyFloor(-diff_clip_);
              d_all.ApplyCeiling(diff_clip_);
            }
          } // end of t

          // errors back-propagated to the inputs
          in_diff->AddMatMat(1.0, DGIFO.RowRange(1,T), kNoTrans, wei_gifo_x_fw_, kNoTrans, 0.0);
          // updates to the model parameters
          const BaseFloat mmt = opts_.momentum;
          wei_gifo_x_fw_corr_.AddMatMat(1.0, DGIFO.RowRange(1,T), kTrans, in, kNoTrans, mmt);
          wei_gifo_m_fw_corr_.AddMatMat(1.0, DGIFO.RowRange(1,T), kTrans, YM.RowRange(0,T), kNoTrans, mmt);
          bias_fw_corr_.AddRowSumMat(1.0, DGIFO.RowRange(1,T), mmt);
          phole_i_c_fw_corr_.AddDiagMatMat(1.0, DI.RowRange(1,T), kTrans, YC.RowRange(0,T), kNoTrans, mmt);
          phole_f_c_fw_corr_.AddDiagMatMat(1.0, DF.RowRange(1,T), kTrans, YC.RowRange(0,T), kNoTrans, mmt);
          phole_o_c_fw_corr_.AddDiagMatMat(1.0, DO.RowRange(1,T), kTrans, YC.RowRange(1,T), kNoTrans, mmt);
        } // end of the forward layer

        // back-propagation in the backward layer
        if (1) {
          // get the activations of the gates/units from the feedforward buffer
          const CuSubMatrix<BaseFloat> YG(propagate_buf_bw_.ColRange(0, cell_dim_));
          const CuSubMatrix<BaseFloat> YI(propagate_buf_bw_.ColRange(1 * cell_dim_, cell_dim_));
          const CuSubMatrix<BaseFloat> YF(propagate_buf_bw_.ColRange(2 * cell_dim_, cell_dim_));
          const CuSubMatrix<BaseFloat> YO(propagate_buf_bw_.ColRange(3 * cell_dim_, cell_dim_));
          const CuSubMatrix<BaseFloat> YC(propagate_buf_bw_.ColRange(4 * cell_dim_, cell_dim_));
          const CuSubMatrix<BaseFloat> YH(propagate_buf_bw_.ColRange(5 * cell_dim_, cell_dim_));
          const CuSubMatrix<BaseFloat> YM(propagate_buf_bw_.ColRange(6 * cell_dim_, cell_dim_));

          // errors back-propagated to individual gates/units
          CuSubMatrix<BaseFloat> DG(backpropagate_buf_bw_.ColRange(0, cell_dim_));
          CuSubMatrix<BaseFloat> DI(backpropagate_buf_bw_.ColRange(1 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> DF(backpropagate_buf_bw_.ColRange(2 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> DO(backpropagate_buf_bw_.ColRange(3 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> DC(backpropagate_buf_bw_.ColRange(4 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> DH(backpropagate_buf_bw_.ColRange(5 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> DM(backpropagate_buf_bw_.ColRange(6 * cell_dim_, cell_dim_));
          CuSubMatrix<BaseFloat> DGIFO(backpropagate_buf_bw_.ColRange(0, 4 * cell_dim_));

          // the second half of the error vector corresponds to the backward layer
          DM.RowRange(1,T).CopyFromMat(out_diff.ColRange(cell_dim_, cell_dim_));

          for (int t = 1; t <= T; t++) {
            // variables representing activations of invidivual units/gates
            const CuSubVector<BaseFloat> y_g(YG.Row(t));  const CuSubMatrix<BaseFloat> YG_t(YG.RowRange(t,1));
            const CuSubVector<BaseFloat> y_i(YI.Row(t));  const CuSubMatrix<BaseFloat> YI_t(YI.RowRange(t,1));
            const CuSubVector<BaseFloat> y_f(YF.Row(t));  const CuSubMatrix<BaseFloat> YF_t(YF.RowRange(t,1));
            const CuSubVector<BaseFloat> y_o(YO.Row(t));  const CuSubMatrix<BaseFloat> YO_t(YO.RowRange(t,1));
            const CuSubVector<BaseFloat> y_c(YC.Row(t));  const CuSubMatrix<BaseFloat> YC_t(YC.RowRange(t,1));
            const CuSubVector<BaseFloat> y_h(YH.Row(t));  const CuSubMatrix<BaseFloat> YH_t(YH.RowRange(t,1));
            const CuSubVector<BaseFloat> y_m(YM.Row(t));  const CuSubMatrix<BaseFloat> YM_t(YM.RowRange(t,1));
            // variables representing errors of invidivual units/gates
            CuSubVector<BaseFloat> d_all(backpropagate_buf_bw_.Row(t));
            CuSubVector<BaseFloat> d_g(DG.Row(t));  CuSubMatrix<BaseFloat> DG_t(DG.RowRange(t,1));
            CuSubVector<BaseFloat> d_i(DI.Row(t));  CuSubMatrix<BaseFloat> DI_t(DI.RowRange(t,1));
            CuSubVector<BaseFloat> d_f(DF.Row(t));  CuSubMatrix<BaseFloat> DF_t(DF.RowRange(t,1));
            CuSubVector<BaseFloat> d_o(DO.Row(t));  CuSubMatrix<BaseFloat> DO_t(DO.RowRange(t,1));
            CuSubVector<BaseFloat> d_c(DC.Row(t));  CuSubMatrix<BaseFloat> DC_t(DC.RowRange(t,1));
            CuSubVector<BaseFloat> d_h(DH.Row(t));  CuSubMatrix<BaseFloat> DH_t(DH.RowRange(t,1));
            CuSubVector<BaseFloat> d_m(DM.Row(t));  CuSubMatrix<BaseFloat> DM_t(DM.RowRange(t,1));

            // d_m comes from two parts: errors from the upper layer and errors from the previous frame (t-1)
            d_m.AddMatVec(1.0, wei_gifo_m_bw_, kTrans, DGIFO.Row(t-1), 1.0);

            // d_h
            d_h.AddVecVec(1.0, y_o, d_m, 0.0);
            DH_t.DiffTanh(YH_t, DH_t);

            // d_o - output gate
            d_o.AddVecVec(1.0, y_h, d_m, 0.0);
            DO_t.DiffSigmoid(YO_t, DO_t);

            // d_c - memory cell
            d_c.AddVec(1.0, d_h, 0.0);
            d_c.AddVecVec(1.0, phole_o_c_bw_, d_o, 1.0);
            d_c.AddVecVec(1.0, YF.Row(t-1), DC.Row(t-1), 1.0);
            d_c.AddVecVec(1.0, phole_f_c_bw_, DF.Row(t-1), 1.0);
            d_c.AddVecVec(1.0, phole_i_c_bw_, DI.Row(t-1), 1.0);

            // d_f - forget gate
            d_f.AddVecVec(1.0, YC.Row(t+1), d_c, 0.0);
            DF_t.DiffSigmoid(YF_t, DF_t);

            // d_i - input gate
            d_i.AddVecVec(1.0, y_g, d_c, 0.0);
            DI_t.DiffSigmoid(YI_t, DI_t);

            // d_g
            d_g.AddVecVec(1.0, y_i, d_c, 0.0);
            DG_t.DiffTanh(YG_t, DG_t);

            // clipping of per-frame derivatives (as done in Google),
            if (diff_clip_ > 0.0) {
              d_all.ApplyFloor(-diff_clip_);
              d_all.ApplyCeiling(diff_clip_);
            }
          }  // end of t

          // errors back-propagated to the inputs
          in_diff->AddMatMat(1.0, DGIFO.RowRange(1,T), kNoTrans, wei_gifo_x_bw_, kNoTrans, 1.0);
          // updates to the parameters
          const BaseFloat mmt = opts_.momentum;
          wei_gifo_x_bw_corr_.AddMatMat(1.0, DGIFO.RowRange(1,T), kTrans, in, kNoTrans, mmt);
          wei_gifo_m_bw_corr_.AddMatMat(1.0, DGIFO.RowRange(1,T), kTrans, YM.RowRange(0,T), kNoTrans, mmt);
          bias_bw_corr_.AddRowSumMat(1.0, DGIFO.RowRange(1,T), mmt);
          phole_i_c_bw_corr_.AddDiagMatMat(1.0, DI.RowRange(1,T), kTrans, YC.RowRange(0,T), kNoTrans, mmt);
          phole_f_c_bw_corr_.AddDiagMatMat(1.0, DF.RowRange(1,T), kTrans, YC.RowRange(0,T), kNoTrans, mmt);
          phole_o_c_bw_corr_.AddDiagMatMat(1.0, DO.RowRange(1,T), kTrans, YC.RowRange(1,T), kNoTrans, mmt);
        } // end of the backward layer
    }

    void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
      // clip gradients
      if (grad_clip_ > 0) {
        wei_gifo_x_fw_corr_.ApplyFloor(-grad_clip_); wei_gifo_x_fw_corr_.ApplyCeiling(grad_clip_);
        wei_gifo_m_fw_corr_.ApplyFloor(-grad_clip_); wei_gifo_m_fw_corr_.ApplyCeiling(grad_clip_);
        bias_fw_corr_.ApplyFloor(-grad_clip_); bias_fw_corr_.ApplyCeiling(grad_clip_);
        phole_i_c_fw_corr_.ApplyFloor(-grad_clip_); phole_i_c_fw_corr_.ApplyCeiling(grad_clip_);
        phole_f_c_fw_corr_.ApplyFloor(-grad_clip_); phole_f_c_fw_corr_.ApplyCeiling(grad_clip_);
        phole_o_c_fw_corr_.ApplyFloor(-grad_clip_); phole_o_c_fw_corr_.ApplyCeiling(grad_clip_);

        wei_gifo_x_bw_corr_.ApplyFloor(-grad_clip_); wei_gifo_x_bw_corr_.ApplyCeiling(grad_clip_);
        wei_gifo_m_bw_corr_.ApplyFloor(-grad_clip_); wei_gifo_m_bw_corr_.ApplyCeiling(grad_clip_);
        bias_bw_corr_.ApplyFloor(-grad_clip_); bias_bw_corr_.ApplyCeiling(grad_clip_);
        phole_i_c_bw_corr_.ApplyFloor(-grad_clip_); phole_i_c_bw_corr_.ApplyCeiling(grad_clip_);
        phole_f_c_bw_corr_.ApplyFloor(-grad_clip_); phole_f_c_bw_corr_.ApplyCeiling(grad_clip_);
        phole_o_c_bw_corr_.ApplyFloor(-grad_clip_); phole_o_c_bw_corr_.ApplyCeiling(grad_clip_);
      }

      // rescale all gradients,
      if (grad_max_norm_ > 0) {
        BaseFloat grad_l2_norm = std::sqrt(
          std::pow(wei_gifo_x_fw_corr_.FrobeniusNorm(), 2.0) +
          std::pow(wei_gifo_m_fw_corr_.FrobeniusNorm(), 2.0) +
          std::pow(bias_fw_corr_.Norm(2), 2.0) +
          std::pow(phole_i_c_fw_corr_.Norm(2), 2.0) +
          std::pow(phole_f_c_fw_corr_.Norm(2), 2.0) +
          std::pow(phole_o_c_fw_corr_.Norm(2), 2.0) +

          std::pow(wei_gifo_x_bw_corr_.FrobeniusNorm(), 2.0) +
          std::pow(wei_gifo_m_bw_corr_.FrobeniusNorm(), 2.0) +
          std::pow(bias_bw_corr_.Norm(2), 2.0) +
          std::pow(phole_i_c_bw_corr_.Norm(2), 2.0) +
          std::pow(phole_f_c_bw_corr_.Norm(2), 2.0) +
          std::pow(phole_o_c_bw_corr_.Norm(2), 2.0)
        );

        BaseFloat scale = grad_max_norm_ / grad_l2_norm;  // same scale for all buffers,

        if (grad_l2_norm > grad_max_norm_) {
          wei_gifo_x_fw_corr_.Scale(scale);
          wei_gifo_m_fw_corr_.Scale(scale);
          bias_fw_corr_.Scale(scale);
          phole_i_c_fw_corr_.Scale(scale);
          phole_f_c_fw_corr_.Scale(scale);
          phole_o_c_fw_corr_.Scale(scale);

          wei_gifo_x_bw_corr_.Scale(scale);
          wei_gifo_m_bw_corr_.Scale(scale);
          bias_bw_corr_.Scale(scale);
          phole_i_c_bw_corr_.Scale(scale);
          phole_f_c_bw_corr_.Scale(scale);
          phole_o_c_bw_corr_.Scale(scale);
        }
      }

      // update parameters
      const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;

      wei_gifo_x_fw_.AddMat(-lr, wei_gifo_x_fw_corr_);
      wei_gifo_m_fw_.AddMat(-lr, wei_gifo_m_fw_corr_);
      bias_fw_.AddVec(-lr * bias_learn_rate_coef_, bias_fw_corr_, 1.0);
      phole_i_c_fw_.AddVec(-lr * phole_learn_rate_coef_, phole_i_c_fw_corr_, 1.0);
      phole_f_c_fw_.AddVec(-lr * phole_learn_rate_coef_, phole_f_c_fw_corr_, 1.0);
      phole_o_c_fw_.AddVec(-lr * phole_learn_rate_coef_, phole_o_c_fw_corr_, 1.0);

      wei_gifo_x_bw_.AddMat(-lr, wei_gifo_x_bw_corr_);
      wei_gifo_m_bw_.AddMat(-lr, wei_gifo_m_bw_corr_);
      bias_bw_.AddVec(-lr * bias_learn_rate_coef_, bias_bw_corr_, 1.0);
      phole_i_c_bw_.AddVec(-lr * phole_learn_rate_coef_, phole_i_c_bw_corr_, 1.0);
      phole_f_c_bw_.AddVec(-lr * phole_learn_rate_coef_, phole_f_c_bw_corr_, 1.0);
      phole_o_c_bw_.AddVec(-lr * phole_learn_rate_coef_, phole_o_c_bw_corr_, 1.0);
    }

    void Scale(BaseFloat scale) {
      wei_gifo_x_fw_.Scale(scale);
      wei_gifo_m_fw_.Scale(scale);
      bias_fw_.Scale(scale);
      phole_i_c_fw_.Scale(scale);
      phole_f_c_fw_.Scale(scale);
      phole_o_c_fw_.Scale(scale);

      wei_gifo_x_bw_.Scale(scale);
      wei_gifo_m_bw_.Scale(scale);
      bias_bw_.Scale(scale);
      phole_i_c_bw_.Scale(scale);
      phole_f_c_bw_.Scale(scale);
      phole_o_c_bw_.Scale(scale);
    }

    void Add(BaseFloat scale, const TrainableLayer & layer_other) {
      const BiLstm *other = dynamic_cast<const BiLstm*>(&layer_other);
      wei_gifo_x_fw_.AddMat(scale, other->wei_gifo_x_fw_);
      wei_gifo_m_fw_.AddMat(scale, other->wei_gifo_m_fw_);
      bias_fw_.AddVec(scale, other->bias_fw_);
      phole_i_c_fw_.AddVec(scale, other->phole_i_c_fw_);
      phole_f_c_fw_.AddVec(scale, other->phole_f_c_fw_);
      phole_o_c_fw_.AddVec(scale, other->phole_o_c_fw_);

      wei_gifo_x_bw_.AddMat(scale, other->wei_gifo_x_bw_);
      wei_gifo_m_bw_.AddMat(scale, other->wei_gifo_m_bw_);
      bias_bw_.AddVec(scale, other->bias_bw_);
      phole_i_c_bw_.AddVec(scale, other->phole_i_c_bw_);
      phole_f_c_bw_.AddVec(scale, other->phole_f_c_bw_);
      phole_o_c_bw_.AddVec(scale, other->phole_o_c_bw_);
    }

    int32 NumParams() const {
      return 2 * ( wei_gifo_x_fw_.NumRows() * wei_gifo_x_fw_.NumCols() +
                   wei_gifo_m_fw_.NumRows() * wei_gifo_m_fw_.NumCols() +
                   bias_fw_.Dim() +
                   phole_i_c_fw_.Dim() +
                   phole_f_c_fw_.Dim() +
                   phole_o_c_fw_.Dim() );
    }

    void GetParams(Vector<BaseFloat>* wei_copy) const {
      wei_copy->Resize(NumParams());
      int32 offset = 0, size;
      // copy parameters of the forward sub-layer
      size = wei_gifo_x_fw_.NumRows() * wei_gifo_x_fw_.NumCols();
      wei_copy->Range(offset, size).CopyRowsFromMat(wei_gifo_x_fw_); offset += size;
      size = wei_gifo_m_fw_.NumRows() * wei_gifo_m_fw_.NumCols();
      wei_copy->Range(offset, size).CopyRowsFromMat(wei_gifo_m_fw_); offset += size;
      size = bias_fw_.Dim();
      wei_copy->Range(offset, size).CopyFromVec(bias_fw_); offset += size;
      size = phole_i_c_fw_.Dim();
      wei_copy->Range(offset, size).CopyFromVec(phole_i_c_fw_); offset += size;
      size = phole_f_c_fw_.Dim();
      wei_copy->Range(offset, size).CopyFromVec(phole_f_c_fw_); offset += size;
      size = phole_o_c_fw_.Dim();
      wei_copy->Range(offset, size).CopyFromVec(phole_o_c_fw_); offset += size;

      // copy parameters of the backward sub-layer
      size = wei_gifo_x_bw_.NumRows() * wei_gifo_x_bw_.NumCols();
      wei_copy->Range(offset, size).CopyRowsFromMat(wei_gifo_x_bw_); offset += size;
      size = wei_gifo_m_bw_.NumRows() * wei_gifo_m_bw_.NumCols();
      wei_copy->Range(offset, size).CopyRowsFromMat(wei_gifo_m_bw_); offset += size;
      size = bias_bw_.Dim();
      wei_copy->Range(offset, size).CopyFromVec(bias_bw_); offset += size;
      size = phole_i_c_bw_.Dim();
      wei_copy->Range(offset, size).CopyFromVec(phole_i_c_bw_); offset += size;
      size = phole_f_c_bw_.Dim();
      wei_copy->Range(offset, size).CopyFromVec(phole_f_c_bw_); offset += size;
      size = phole_o_c_bw_.Dim();
      wei_copy->Range(offset, size).CopyFromVec(phole_o_c_bw_); offset += size;
    }

//private:
protected:
    int32 cell_dim_;
    BaseFloat learn_rate_coef_;
    BaseFloat bias_learn_rate_coef_;
    BaseFloat phole_learn_rate_coef_;
    BaseFloat grad_max_norm_;
    BaseFloat grad_clip_;
    BaseFloat diff_clip_;
    BaseFloat cell_clip_;
    BaseFloat drop_factor_;

    CuMatrix<BaseFloat> drop_mask_;

    // parameters of the forward layer
    CuMatrix<BaseFloat> wei_gifo_x_fw_;
    CuMatrix<BaseFloat> wei_gifo_m_fw_;
    CuVector<BaseFloat> bias_fw_;
    CuVector<BaseFloat> phole_i_c_fw_;
    CuVector<BaseFloat> phole_f_c_fw_;
    CuVector<BaseFloat> phole_o_c_fw_;
    // the corresponding parameter updates
    CuMatrix<BaseFloat> wei_gifo_x_fw_corr_;
    CuMatrix<BaseFloat> wei_gifo_m_fw_corr_;
    CuVector<BaseFloat> bias_fw_corr_;
    CuVector<BaseFloat> phole_i_c_fw_corr_;
    CuVector<BaseFloat> phole_f_c_fw_corr_;
    CuVector<BaseFloat> phole_o_c_fw_corr_;

    // parameters of the backward layer
    CuMatrix<BaseFloat> wei_gifo_x_bw_;
    CuMatrix<BaseFloat> wei_gifo_m_bw_;
    CuVector<BaseFloat> bias_bw_;
    CuVector<BaseFloat> phole_i_c_bw_;
    CuVector<BaseFloat> phole_f_c_bw_;
    CuVector<BaseFloat> phole_o_c_bw_;
    // the corresponding parameter updates
    CuMatrix<BaseFloat> wei_gifo_x_bw_corr_;
    CuMatrix<BaseFloat> wei_gifo_m_bw_corr_;
    CuVector<BaseFloat> bias_bw_corr_;
    CuVector<BaseFloat> phole_i_c_bw_corr_;
    CuVector<BaseFloat> phole_f_c_bw_corr_;
    CuVector<BaseFloat> phole_o_c_bw_corr_;

    // propagation buffer
    CuMatrix<BaseFloat> propagate_buf_fw_;
    CuMatrix<BaseFloat> propagate_buf_bw_;

    // back-propagation buffer
    CuMatrix<BaseFloat> backpropagate_buf_fw_;
    CuMatrix<BaseFloat> backpropagate_buf_bw_;

};

} // namespace eesen

#endif
