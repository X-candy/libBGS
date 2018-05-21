#include "CascadedBgsParams.h"


CascadedBgsParams::CascadedBgsParams(const s32 in_len, const s16 in_N,
		const s16 in_ovlstep,
		const bgscascade_thresholds &in_T_vals)
	:
	len(in_len),
	N(in_N),
	sub_mat_elem(4),
	ovlstep(in_ovlstep),
	n_gaus(2),
	n_gaus_final(1),
	n_iter(5),
	trust(0.9),
	normalise(true),
	print_progress(false),
	rho(0.02),
	alpha(0.05),
	cosinedist_T(in_T_vals.cosinedist_T),
	tmprl_cosinedist_T(in_T_vals.cosinedist_T * 0.5),
	likelihood_ratio_T(0.9),
	fv_type(0)
{
}

CascadedBgsParams::~CascadedBgsParams()
{
	
}
