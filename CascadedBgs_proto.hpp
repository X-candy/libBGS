#ifndef CASCADEDBGS_H_
#define CASCADEDBGS_H_

#include "mog_diag.hpp"
#include "CascadedBgsParams.h"


template <typename eT>
class CascadedBgs
{

public:

	cv::Mat rawMask;
	double detection_percentage;
	string file_name;

	CascadedBgs() {};

	CascadedBgs(const cv::Mat &frame,
			const CascadedBgsParams& cascadedBgsParams);
	virtual ~CascadedBgs();

	void model_estimation(const s32 &len);
	void initialise_new_means();
	void initialise_new_dcovs();
	virtual void detectRaw(const cv::Mat &frame);

private:

	int m_nframeNumber, m_mod_B_frameNumber;
	field<cv::Mat> m_vecImgarray;
	CascadedBgsParams *m_pCascadedBgsParams;
	s16 m_nXmb, m_nYmb;
	u32 no_of_blks;
	field<mog_diag<eT> > m_Gm_model_initial;
	field<mog_diag<eT> > m_Gm_model_final;
	field<mog_diag<eT> > m_Global_image_model;

	field< Col<eT> > m_Full_update_BG_model;
	field< Col<eT> > m_mu;
	field< Col<eT> > m_varience;
	field< Col<eT> > m_frame_fv;
	Mat<eT> m_dct_mtx, m_dct_mtx_trans;
	Mat<eT> m_fg_wt_mtx;
	Mat<eT> m_similarity_mtx;
	Mat<u32> m_Fg_wts;
	Col<eT>  m_DCT_coeffs;
	Col<eT>  m_Global_gmm_id;
	field<Mat<eT> > m_vecMask;
	mat m_Fg_persistence;
	field< running_stat_vec<eT> >m_vec_Fg_model;
	cv::Mat m_bin_image1, m_bin_image2;
	cv::Mat m_Mean_bg;
	cv::Mat m_tmp_img;
	cv::Mat m_Threshold_img;
	double m_Seq_threshold;


	u16 channels;
	u16 gauss_flg, toggle_flg;
	u16 frm_idx;
	bool model_B_train;
	u16 width, height;


public:

	void fv_extraction(field<Col<eT> > &f_vec, int len1, const int &channel,
			const Cube<eT> &plane, const u32 &fv_type);

	void frame_fv_extraction(const field< Mat<eT> > &plane_vec, const u32 &fv_type, const u32 &dest_type);

	int detect_foreground(const cv::Mat &frame);
	void create_foreground_mask();
	inline void cosinedist(const Col<eT>  &f_vec, const Col<eT>  &mu, eT &cosdistval);

	void choose_dominant_gaussian(s32 flag);

	void create_dct_table(int N);
	void idct_reconstruct(field<Mat <eT> > &res_mtx, u32 lcnt);
	void trace_blocks(const IplImage *frame, const int i, const int j);
	void gmm_reconstruct(field<Mat <eT> > &out_mtx,const u32  & x, const u32  &y);

	void show_mean_background_img(cv::Mat &get_mean_img);
	void img_mask_concatenate(const cv::Mat &input_frame, cv::Mat &concatenate_img);
	void numtostr(int num, char *str);

};

#endif /* CASCADEDBGS_H_ */
