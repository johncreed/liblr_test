#ifndef _EXP_CODE_H
#define _EXP_CODE_H

#include "linear.h"

#ifdef __cplusplus
extern "C" {
#endif

struct problem_folds{
	int *perm;
	int *fold_start;
	int nr_fold;
	problem *subprobs;
};

struct problem_folds* split_data(const problem *prob, int nr_fold);
void find_parameter(const problem *prob, parameter *param, int nr_fold, double min_C, double max_C, double *best_C, double *best_rate);
void find_parameter_fix_p(const problem *prob, const problem_folds *prob_folds, const parameter *param, int nr_fold, double min_C, double max_C, double *best_C, double *best_rate);
double calc_error(const problem *prob ,const parameter *param, double *target);

double calc_min_C(const problem *prob, const parameter *param);
double calc_max_P(const problem *prob, const parameter *param);
double calc_min_P(const problem *prob, const parameter *param);

void add_iter(int num);
void reset_iter_sum();
void print_iter_sum(double cur_C, double cur_P);

#ifdef __cplusplus
}
#endif

#endif
