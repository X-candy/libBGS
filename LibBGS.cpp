#include "LibBGS.h"
#include "CascadedBgsParams.h"

CLibBGS::CLibBGS()
{
	m_nCur_param_vec = initialiseParameters();
	
	m_nTraining_frames = 50;
	const s16 N = m_nCur_param_vec(0);
	const s16 ovlstep = m_nCur_param_vec(1);

	bgscascade_thresholds T_vals;
	T_vals.cosinedist_T = m_nCur_param_vec(2);

	m_pCascadedBgsParams = new CascadedBgsParams(m_nTraining_frames, N, ovlstep, T_vals);
	// CascadedBgsParams(m_nTraining_frames, N, ovlstep, T_vals);
	m_nFrameCount = 0;
}


CLibBGS::~CLibBGS()
{
}


int CLibBGS::Initial(int _nImgWitdh,int _nImgHeigth,int _nChannel)
{
	//assert(_nChannel == 3);
	m_InputImg = cv::Mat::zeros(_nImgHeigth, _nImgWitdh, CV_8UC(_nChannel));
	m_pObj = new CascadedBgs<double>(m_InputImg, *m_pCascadedBgsParams);
	//TODO:ReInitial

	return 0;
}

rowvec CLibBGS::initialiseParameters()
{
	rowvec tmp;
	tmp.set_size(3);

	/*set block size N*/
	tmp(0) = 8;

	/*set pixel advancement (smaller advancement translates to higher overlap) */
	tmp(1) = 2;

	/*threshold for Cosine distance based classifier (2nd classifier)*/
	tmp(2) = 0.001;

	return tmp;
}

int CLibBGS::FillBuffer(cv::Mat _Img, cv::Mat& _Fg)
{
	if (m_nFrameCount ==0)
		Initial(_Img.cols, _Img.rows, _Img.channels());

	m_pObj->detectRaw(_Img);
	m_pObj->rawMask.copyTo(_Fg);

	m_nFrameCount++;
	return 0;
}

int CLibBGS::GetBg(cv::Mat& _bg)
{
	m_pObj->show_mean_background_img(_bg);
	return 1;
}

int CLibBGS::Destroy()
{
	return 1;
}

