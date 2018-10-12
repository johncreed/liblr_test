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

void reset_init_sols(problem_folds *prob_folds);

struct problem_folds* split_data(const problem *prob, int nr_fold);
double calc_min_C(const problem *prob, const parameter *param);
double calc_max_P(const problem *prob, const parameter *param);
double calc_min_P(const problem *prob, const parameter *param);

void cross_validation_with_splits(const problem *prob, const problem_folds *prob_folds,const parameter *param, int nr_fold, double &score, bool &w_diff);
void linear_step_fix_range(const problem *prob,const parameter *param, int nr_fold);
void linear_step_fix_range_nowarm(const problem *prob,const parameter *param, int nr_fold);
void log_step_fix_range(const problem *prob,const parameter *param, int nr_fold);
void log_step_fix_range_nowarm(const problem *prob,const parameter *param, int nr_fold);
void C_P_new(const problem *prob,const parameter *param,int nr_fold);
void P_C_new(const problem *prob,const parameter *param,int nr_fold);
void P_C_old(const problem *prob,const parameter *param,int nr_fold);
void P_C_nowarm(const problem *prob,const parameter *param,int nr_fold);
void FP_C_nowarm(const problem *prob,const parameter *param,int nr_fold);
void full_nowarm_fix_p(const problem *prob,const parameter *param,int nr_fold);

void cls_old(const problem *prob,const parameter *param,int nr_fold); 
void cls_new(const problem *prob,const parameter *param,int nr_fold); 

double calc_error(const problem *prob ,const parameter *param, double *target);

void add_iter(int num);
void reset_iter_sum();
void print_iter_sum(double p, double C);

void reset_new_break();
void add_new_break();
int get_new_break();

// Show the go P for each fix parameter C.

#ifdef __cplusplus
}
#endif

#endif
