#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>
#include "linear.h"
#include "tron.h"
#include "exp_code.h"

typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}
static void print_null(const char *s) {}

static void (*liblinear_print_string) (const char *) = &print_string_stdout;

#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*liblinear_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif


#ifdef __cplusplus
extern "C" {
#endif
// define in blas/dnrm2_ file
extern double dnrm2_(int *, double *, int *);

#ifdef __cplusplus
}
#endif

problem_folds* split_data(const problem *prob, int nr_fold){
	problem_folds *prob_folds = Malloc(problem_folds, 1);
	
	int l = prob->l;
	prob_folds->nr_fold = nr_fold;
	prob_folds->fold_start = Malloc(int, nr_fold+1);
	prob_folds->perm = Malloc(int, l);
	prob_folds->subprobs = Malloc(problem, nr_fold);

	int *fold_start = prob_folds->fold_start;
	int *perm = prob_folds->perm;
	problem *subprobs = prob_folds->subprobs;

	for(int i = 0; i < l; i++) perm[i] = i;
	for(int i = 0; i < l; i++)
	{
		int j = i+rand()%(l-i);
		swap(perm[i],perm[j]);
	}
	for(int i=0;i<=nr_fold;i++)
		fold_start[i]=i*l/nr_fold;

	for(int i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];

		subprobs[i].bias = prob->bias;
		subprobs[i].n = prob->n;
		subprobs[i].l = l-(end-begin);
		subprobs[i].x = Malloc(struct feature_node*,subprobs[i].l);
		subprobs[i].y = Malloc(double,subprobs[i].l);

		int k = 0;
		for(int j=0;j<begin;j++)
		{
			subprobs[i].x[k] = prob->x[perm[j]];
			subprobs[i].y[k] = prob->y[perm[j]];
			++k;
		}
		for(int j=end;j<l;j++)
		{
			subprobs[i].x[k] = prob->x[perm[j]];
			subprobs[i].y[k] = prob->y[perm[j]];
			++k;
		}
	}

		return prob_folds;
}



void find_parameter(const problem *prob, parameter *param, int nr_fold, double min_C, double max_C, double *best_C, double *best_rate)
{
	double min_P = calc_min_P(prob, param);
	double  max_P = calc_max_P(prob, param);
	max_C = (int) 1 << 30;
	double ratio = 2.0;
	param->p = max_P / ratio;

// split data
if (nr_fold > prob->l)
{
	nr_fold = prob->l;
	fprintf(stderr,"WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
}
struct problem_folds *prob_folds = split_data(prob, nr_fold);

//fix p run C
double current_rate = *best_rate;
double best_P = max_P;
while(param->p > min_P){
	min_C = calc_min_C(prob, param);
	reset_iter_sum();
	find_parameter_fix_p(prob, prob_folds, param, nr_fold, min_C, max_C, best_C, best_rate);
	print_iter_sum(max_C, param->p);
	if(*best_rate != current_rate)
		best_P = param->p;
	current_rate = *best_rate;
	param->p /= ratio;
 }
}

void find_parameter_fix_p(const problem *prob, const problem_folds *prob_folds, const parameter *param, int nr_fold, double min_C, double max_C, double *best_C, double *best_rate)
{
	int *fold_start = prob_folds->fold_start;
	int l = prob->l;
	int *perm = prob_folds->perm;
	double *target = Malloc(double, l);
	struct problem *subprob = prob_folds->subprobs;

	// variables for warm start
	double ratio = 2;
	double **prev_w = Malloc(double*, nr_fold);
	for(int i = 0; i < nr_fold; i++)
		prev_w[i] = NULL;
	int num_unchanged_w = 0;
	struct parameter param1 = *param;
	void (*default_print_string) (const char *) = liblinear_print_string;


	*best_rate = 0;
	if(min_C <= 0)
		min_C = calc_start_C(prob,param);
	param1.C = min_C;

	while(param1.C <= max_C)
	{
		//Output disabled for running CV at a particular C
		set_print_string_function(&print_null);

		for(int i=0; i<nr_fold; i++)
		{
			int begin = fold_start[i];
			int end = fold_start[i+1];

			param1.init_sol = prev_w[i];
			struct model *submodel = train(&subprob[i],&param1);

			int total_w_size;
			if(submodel->nr_class == 2)
				total_w_size = subprob[i].n;
			else
				total_w_size = subprob[i].n * submodel->nr_class;

			if(prev_w[i] == NULL)
			{
				prev_w[i] = Malloc(double, total_w_size);
			}
			else if(num_unchanged_w >= 0)
			{
				double norm_w_diff = 0;
				for(int j=0; j<total_w_size; j++)
					norm_w_diff += (submodel->w[j] - prev_w[i][j])*(submodel->w[j] - prev_w[i][j]);
				norm_w_diff = sqrt(norm_w_diff);

				if(norm_w_diff > 1e-15)
					num_unchanged_w = -1;
			}

			for(int j=0; j<total_w_size; j++)
				prev_w[i][j] = submodel->w[j];

			for(int j=begin; j<end; j++)
				target[perm[j]] = predict(submodel,prob->x[perm[j]]);

			free_and_destroy_model(&submodel);
		}
		set_print_string_function(default_print_string);

		double current_rate = calc_error(prob, param, target);
		double order;
		if(param->solver_type == L2R_LR || param->solver_type == L2R_L2LOSS_SVC)
		{
			order = 1.0;
		}
		else if(param->solver_type == L2R_L2LOSS_SVR)
		{
			order = -1.0;
		}
		else
		{
			printf("Not supported");
			exit(EXIT_FAILURE);
		}
		if( (order * current_rate) > (order * *best_rate) )
		{
			*best_C = param1.C;
			*best_rate = current_rate;
		}
		info("log2c=%7.2f\trate=%g\n",log(param1.C)/log(2.0),100.0*current_rate);
		num_unchanged_w++;
		if(num_unchanged_w == 3)
			break;
		param1.C = param1.C*ratio;
	}

	if(param1.C > max_C && max_C > min_C)
		info("warning: maximum C reached.\n");
	free(target);
	for(int i=0; i<nr_fold; i++)
		free(prev_w[i]);
	free(prev_w);
}

double calc_error(const problem *prob ,const parameter *param, double *target)
{
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;

	if(param->solver_type == L2R_L2LOSS_SVR ||
	   param->solver_type == L2R_L1LOSS_SVR_DUAL ||
	   param->solver_type == L2R_L2LOSS_SVR_DUAL)
	{
		for(int i=0;i<prob->l;i++)
		{
			double y = prob->y[i];
			double v = target[i];
			total_error += (v-y)*(v-y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		//printf("Cross Validation Mean squared error = %g\n",total_error/prob->l);
		//printf("Cross Validation Squared correlation coefficient = %g\n",
		//		((prob->l*sumvy-sumv*sumy)*(prob->l*sumvy-sumv*sumy))/
		//		((prob->l*sumvv-sumv*sumv)*(prob->l*sumyy-sumy*sumy))
		//	  );
		return total_error / prob->l;
	}
	else
	{
		for(int i=0;i<prob->l;i++)
			if(target[i] == prob->y[i])
				++total_correct;
		//printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob->l);
		return (double) total_correct / prob->l;
	}
}


double calc_min_C(const problem *prob, const parameter *param)
{
	int i;
	double xTx,max_xTx;
	double phi, loss, yi_abs;
	double delta = 0.1;
	phi = loss = max_xTx = 0;
	for(i=0; i<prob->l; i++)
	{
		xTx = 0;
		feature_node *xi=prob->x[i];
		yi_abs = (prob->y[i] >= 0)? prob->y[i] : -1.0 * prob->y[i];
		while(xi->index != -1)
		{
			double val = xi->value;
			xTx += val*val;
			xi++;
		}
		if(xTx > max_xTx)
			max_xTx = xTx;
		phi += max( yi_abs - param->p, 0.0 );
		loss += max( yi_abs - param->p, 0.0);
	}

	if(loss == 0.0){
		fprintf( stderr, "param->p is too large!!! Please try param->p / 2.0!!!\n");
		exit(1);
	}

	double min_C = 1.0;
	if(param->solver_type == L2R_LR)
		min_C = 1.0 / (prob->l * max_xTx);
	else if(param->solver_type == L2R_L2LOSS_SVC)
		min_C = 1.0 / (2 * prob->l * max_xTx);
	else if(param->solver_type == L2R_L2LOSS_SVR)
		min_C = delta * delta * loss / (8.0 * phi * max_xTx);

	return pow( 2, floor(log(min_C) / log(2.0)) );
}

double calc_max_P(const problem *prob, const parameter *param)
{
	double yi_abs, max_yi_abs;
	double l = prob->l;
	max_yi_abs = 0;
	for(int i = 0; i < l; i++)
	{
		yi_abs = (prob->y[i] >= 0)? prob->y[i] : -1.0 * prob->y[i];
		max_yi_abs = max( max_yi_abs, yi_abs);
	}

	double max_P = max_yi_abs;

	return pow( 2, floor(log(max_P) / log(2.0)));
}

double calc_min_P(const problem *prob, const parameter *param)
{
	int n = prob->n, l = prob->l;
	
	double norm_2Xy;
	int inc = 1;
	double *w0 = new double[n];
	double *g = new double[n];
	double *C = new double[l];
	for (int i=0; i<n; i++)
		w0[i] = 0;
	for(int j=0; j < l; j++)
		C[j] = 1;
	function * fun_obj=new l2r_l2_svr_fun(prob, C, 0.0);
	fun_obj->grad(w0, g);
	norm_2Xy = dnrm2_(&n, g, &inc);
	free(C);
	free(fun_obj);
	free(w0);
	free(g);
	
	double max_x = 0;
	for(int i = 0; i < prob->l; i++){
		feature_node * xi = prob->x[i];
		while( xi->index != -1){
			double val = (xi->value >= 0)? xi->value : -xi->value;
			max_x = max( max_x, val);
			xi++;
		}
	}
	
	double delta = param->eps / (param->eps + 2);
	double min_P = delta * norm_2Xy / ( 2.0 * max_x* sqrt(n) * l );
	return pow( 2, ceil( log(min_P) / log(2.0) ));
}


int total_iter_sum;

void add_iter(int num)
{
	total_iter_sum += num;
}

void reset_iter_sum()
{
	total_iter_sum = 0;
}

void print_iter_sum(double cur_C, double cur_P)
{
	printf("cur_C: %g cur_P: %g iter_sum: %d\n", log(cur_C)/log(2.0), log(cur_P)/log(2.0), total_iter_sum);
}

