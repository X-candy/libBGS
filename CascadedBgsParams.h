#ifndef CASCADEDBGSPARAMS_H_
#define CASCADEDBGSPARAMS_H_
#pragma once
#include "inc.hpp"

typedef struct BGS_THRESHOLD
{

	 double cosinedist_T;

} bgscascade_thresholds;



class CascadedBgsParams
{
public:
	const s32 len;
	const s16 N;
	const s16 sub_mat_elem;
	const s16 ovlstep;
	const u32 n_gaus;
	const u32 n_gaus_final;
	const u32 n_iter;
	const double trust;
	const bool normalise;
	const bool print_progress;
	const double rho;
	const double alpha;
	const double cosinedist_T;
	const double likelihood_ratio_T;
	const double tmprl_cosinedist_T;
	const u32 fv_type;

public:
	CascadedBgsParams(const s32 in_len,const s16 in_N, const s16 in_ovlstep, const bgscascade_thresholds  &T_vals);

	~CascadedBgsParams();
};


#endif /* CASCADEDBGSPARAMS_H_ */
