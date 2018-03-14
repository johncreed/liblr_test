#ifndef _EXP_CODE_H
#define _EXP_CODE_H

#include "linear.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>
#include "linear.h"
#include "tron.h"

#ifdef __cplusplus
extern "C" {
#endif

struct problem_folds{
	int *perm;
	int *fold_start;
	int nr_fold;
	problem *subprobs;
	~problem_folds();
};

struct problem_folds* split_data(const problem *prob, int nr_fold);
void find_parameter(const problem *prob, const parameter *param, int nr_fold);
void find_parameter_v2(const problem *prob, const parameter *param, int nr_fold);
void find_parameter_fix_c(const problem *prob, const problem_folds *prob_folds, const parameter *param, int nr_fold, double min_P, double max_P, double *best_P, double *best_rate);
void find_parameter_fix_p(const problem *prob, const problem_folds *prob_folds, const parameter *param, int nr_fold, double min_C, double max_C, double *best_C, double *best_rate);
void find_parameter_fix_p_v2(const problem *prob, const problem_folds *prob_folds, const parameter *param, int nr_fold, double min_C, double max_C, double *best_C, double *best_rate);
double calc_error(const problem *prob ,const parameter *param, double *target);

double calc_min_C(const problem *prob, const parameter *param);
double calc_max_P(const problem *prob, const parameter *param);
double calc_min_P(const problem *prob, const parameter *param);

void add_iter(int num);
void reset_iter_sum();
void reset_iter_sum_v2();
void print_iter_sum(char c, double val);
void print_iter_sum_v2(double p, double C);

double get_l2r_l2_svr_fun_grad_norm(double *w, const problem *prob, const parameter *param);
double get_l2r_l2_svr_loss_norm(double *w, const problem *prob, const double p);

void find_parameter_classification(const problem *prob, const parameter *param, int nr_fold, double start_C, double max_C, double *best_C, double *best_rate);

double get_l2r_lr_loss_norm(double *w, const problem *prob);
double get_l2r_l2l_svc_loss_norm(double *w, const problem *prob);
#ifdef __cplusplus
}
#endif

#endif
