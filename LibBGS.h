#pragma once
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>


#include <stdio.h>
#include <tchar.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <iostream>
#include <Windows.h>
#include <time.h>
#include <algorithm>
#include <iomanip>

#include "CascadedBgsParams.h"
#include "inc.hpp"

#include "CascadedBgs_meat.hpp"

#include "../h/ILibBGS.h"
using namespace std;

class CLibBGS :public CILibBGS
{
public:
	CLibBGS();
	~CLibBGS();

	int Initial(int _nImgWitdh, int _nImgHeigth, int _nChannel);
	int FillBuffer(cv::Mat _Img, cv::Mat& _Fg);
	int GetBg(cv::Mat& _bg);
	int Destroy();

private:
	rowvec initialiseParameters();

	rowvec m_nCur_param_vec;
	int m_nTraining_frames;
	CascadedBgsParams* m_pCascadedBgsParams;
	CascadedBgs<double>* m_pObj;

	cv::Mat m_InputImg;
	long m_nFrameCount;
};


LIBBGS_API CILibBGS* CreateBackGround()
{
	CILibBGS* pBackGround = (CILibBGS*)new CLibBGS();

	return pBackGround;
}

LIBBGS_API int DestoryBackGround(CILibBGS* _pIBackGround)
{
	CLibBGS* pBackGround = (CLibBGS*)_pIBackGround;
	pBackGround->Destroy();
	return 0;
}