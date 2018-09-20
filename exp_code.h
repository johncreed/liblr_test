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
  double **init_sols;
	problem *subprobs;
	~problem_folds();
};

struct problem_folds* split_data(const problem *prob, int nr_fold);
double calc_min_C(const problem *prob, const parameter *param);
double calc_max_P(const problem *prob, const parameter *param);
double calc_min_P(const problem *prob, const parameter *param);

void find_parameter_classification(const problem *prob, const problem_folds * prob_folds, const parameter *param, int nr_fold, double start_C, double max_C, double *best_C, double *best_rate);

void find_parameter_linear_step_C_P(const problem *prob, const parameter *param, int nr_fold);
void find_parameter_P(const problem *prob, const problem_folds *prob_folds, const parameter *param, int nr_fold, double min_P, double max_P, double *best_P, double *best_rate);

void find_parameter_linear_step_P_C(const problem *prob,const parameter *param, int nr_fold);
void find_parameter_C(const problem *prob, const problem_folds *prob_folds, const parameter *param, int nr_fold, double min_C, double max_C, double *best_C, double *best_rate);

void find_parameter_linear_step_P_C_noWarm(const problem *prob,const parameter *param, int nr_fold);
void find_parameter_C_noWarm(const problem *prob, const problem_folds *prob_folds, const parameter *param, int nr_fold, double min_C, double max_C, double *best_C, double *best_rate);


void cross_validation_with_splits(const problem *prob, const problem_folds *prob_folds,const parameter *param, int nr_fold, double &score, bool &w_diff);
void linear_step_fix_range(const problem *prob,const parameter *param, int nr_fold);
void log_step_fix_range(const problem *prob,const parameter *param, int nr_fold);

double calc_error(const problem *prob ,const parameter *param, double *target);

void add_iter_sum_fix_one_param(int num);
void reset_iter_sum_fix_one_param();
void print_iter_sum_fix_one_param(char c, double val);

void add_iter(int num);
void reset_iter_sum();
void print_iter_sum(double p, double C);

void reset_iter_sum_whole_process();
void update_iter_sum_whole_process();
void print_iter_sum_whole_process();

double get_l2r_l2_svr_fun_grad_norm(double *w, const problem *prob, const parameter *param);
double get_l2r_l2_svr_loss_norm(double *w, const problem *prob, const double p);


double get_l2r_lr_loss_norm(double *w, const problem *prob);
double get_l2r_l2l_svc_loss_norm(double *w, const problem *prob);

void reset_new_break();
void add_new_break();
int get_new_break();

// Show the go P for each fix parameter C.

#ifdef __cplusplus
}
#endif

#endif
