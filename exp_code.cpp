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

const double numSteps = 20.0;
double stepSz;
const double max_PReduce = 5.0;

problem_folds::~problem_folds()
{
	free(perm);
	free(fold_start);
	for(int i = 0; i < nr_fold; i++)
	{
		free( (subprobs+i)->x);
		free( (subprobs+i)->y);
	}
	free(subprobs);
}

problem_folds* split_data(const problem *prob, int nr_fold){
	//Check nr_fold is valid
	if (nr_fold > prob->l)
	{
		nr_fold = prob->l;
		fprintf(stderr,"WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
	}
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

void find_parameter_linear_step(const problem *prob,const parameter *param, int nr_fold)
{
	//Set range of parameter
	double ratio = 2.0;
	double min_P = 0.0;
	double max_P = calc_max_P(prob, param);
	double max_C = pow(2.0, 50);
	double min_C = INF;
	struct parameter param1 = *param;
	param1.p = min_P;
	while( param1.p < max_P ){
		min_C = min( calc_min_C( prob, &param1), min_C);
		param1.p += stepSz;
	}
	printf("min_P %g max_P %g min_C %g max_C %g\n", log(min_P)/log(2.0), log(max_P)/log(2.0), log(min_C)/log(2.0), log(max_C)/log(2.0));

	// split data
	if (nr_fold > prob->l)
	{
		nr_fold = prob->l;
		fprintf(stderr,"WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
	}
	struct problem_folds *prob_folds = split_data(prob, nr_fold);

	//fix p run C
	double current_rate = INF, best_rate = INF;
	double best_P, best_C;
	param1.p = max_P;
	reset_iter_sum_whole_process();
	while( true ){
		//If next step is too close to zero, set it to zero.
		if( param1.p < stepSz / 2.0)
			param1.p = 0.0;

		reset_iter_sum_fix_one_param();
		find_parameter_fix_p(prob, prob_folds, &param1, nr_fold, min_C, max_C, &best_C, &current_rate);
		if(best_rate > current_rate){
			best_P = param1.p;
			best_rate = current_rate;
		}
		
		if(param1.p == 0.0)
			printf("Cumulative logP : INF ");
		else
			printf("Cumulative logP : %g ", log2(param1.p) );
		print_iter_sum_whole_process();

		// param1.p is zero. It is the last iteration
		if( param1.p == 0.0 )
			break;
		else
			param1.p -= stepSz;
	}
	
	// Print the best result
	printf("======================================\n");
	print_iter_sum_whole_process();
	if( best_P == 0.0 )
		printf("Best logP = INF Best logC = %g Best MSE = %g \n", log2(best_C), best_rate );
	else
		printf("Best logP = %g Best logC = %g Best MSE = %g \n", log2(best_P), log2(best_C), best_rate );
}
/**
void find_parameter(const problem *prob,const parameter *param, int nr_fold)
{
	//Set range of parameter
	double ratio = 2.0;
	double min_P = calc_min_P(prob, param);
	double  max_P = calc_max_P(prob, param) / ratio;
	double max_C = pow(2.0, 50);
	double min_C = INF;
	struct parameter param1 = *param;
	param1.p = min_P;
	while( param1.p <= max_P ){
		min_C = min( calc_min_C( prob, &param1), min_C);
		param1.p *= ratio;
	}
	printf("min_P %g max_P %g min_C %g max_C %g\n", log(min_P)/log(2.0), log(max_P)/log(2.0), log(min_C)/log(2.0), log(max_C)/log(2.0));
	// split data
	if (nr_fold > prob->l)
	{
		nr_fold = prob->l;
		fprintf(stderr,"WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
	}
	struct problem_folds *prob_folds = split_data(prob, nr_fold);

	//fix p run C
	double current_rate = INF, best_rate = INF;
	double best_P, best_C;
	param1.p = max_P;
	while(param1.p > min_P){
		reset_iter_sum_fix_one_param();
		find_parameter_fix_p(prob, prob_folds, &param1, nr_fold, min_C, max_C, &best_C, &current_rate);
		print_iter_sum_fix_one_param('P', param1.p);
		if(best_rate > current_rate){
			best_P = param1.p;
			best_rate = current_rate;
		}
		param1.p /= ratio;
	}
	// Do the last run with tight eps
	param1.eps  = param1.eps / (param1.eps + 2.0);
	param1.p = min_P;
	find_parameter_fix_p(prob, prob_folds, &param1, nr_fold, min_C, max_C, &best_C, &current_rate);
	if(best_rate > current_rate){
		best_P = param1.p;
		best_rate = current_rate;
	}

	printf("Best logC = %g Best logP = %g Best MSE = %g \n", log(best_C)/log(2.0), log(best_P)/log(2.0), best_rate );
}
**/

void find_parameter_fix_p(const problem *prob, const problem_folds *prob_folds, const parameter *param, int nr_fold, double min_C, double max_C, double *best_C, double *best_rate)
{
	printf("======================================\n");
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


	*best_rate = INF;
	
	//test breaking condition
	bool first_old_break = true;
	bool first_new_break = true;

	param1.C = min_C;
	while(param1.C <= max_C)
	{
		//Output disabled for running CV at a particular C
		set_print_string_function(&print_null);

		reset_iter_sum();
		reset_new_break();
		for(int i=0; i<nr_fold; i++)
		{
			int begin = fold_start[i];
			int end = fold_start[i+1];

			param1.init_sol = prev_w[i];
			double eps = param1.eps;
			param1.eps = (1.0 - 0.1) * param1.eps;
			struct model *submodel = train(&subprob[i],&param1);
			param1.eps = eps;

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
		print_iter_sum( param1.p, param1.C);
		set_print_string_function(default_print_string);

		double current_rate = calc_error(prob, param, target);
		if( current_rate < *best_rate )
		{
			*best_C = param1.C;
			*best_rate = current_rate;
		}
		if(param1.p == 0.0)
			info("log2P= INF log2C= %7.2f\tMSE= %g\n",log2(param1.C), current_rate);
		else
			info("log2P= %7.2f log2C= %7.2f\tMSE= %g\n", log2(param1.p), log2(param1.C), current_rate);
		num_unchanged_w++;

		//Check old break condition
		if(num_unchanged_w == 3 && first_old_break == true){
			if( param1.p == 0.0 )
				printf("Old Break P: INF C: %g MSE= %g \n",  log2(param1.C), current_rate ) ;
			else
				printf("Old Break P: %g C: %g MSE= %g \n", log2(param1.p), log2(param1.C), current_rate ) ;
			first_old_break = false;
			printf("Old Break Iteration: ");
			print_iter_sum_fix_one_param('P', param1.p);
		}
		
		if( get_new_break() == nr_fold && first_new_break == true){
			if( param1.p == 0.0 )
				printf("New Break P: INF C: %g MSE= %g \n", log2(param1.C), current_rate );
			else
				printf("New Break P: %g C: %g MSE= %g \n", log2(param1.p), log2(param1.C), current_rate );
			first_new_break = false;
			printf("New Break Iteration: ");
			print_iter_sum_fix_one_param('P', param1.p);
			update_iter_sum_whole_process();
		}

		// If both old break and new break satisfy, then we terminate the search process for parameter C.
		if( first_old_break == false && first_new_break == false){
			break;
		}
		
		
		// Update next problem parameter
		param1.C = param1.C * ratio;
	}

	if(param1.C > max_C && max_C > min_C)
		info("warning: maximum C reached.\n");
	free(target);
	for(int i=0; i<nr_fold; i++)
		free(prev_w[i]);
	free(prev_w);
	param1.init_sol = NULL;
}


void find_parameter_linear_step_noWarm(const problem *prob,const parameter *param, int nr_fold)
{
	printf("No Warm Start Log\n");
	//Set range of parameter
	double ratio = 2.0;
	double min_P = 0.0;
	double max_P = calc_max_P(prob, param);
	double max_C = pow(2.0, 50);
	double min_C = INF;
	struct parameter param1 = *param;
	param1.p = min_P;
	while( param1.p < max_P ){
		min_C = min( calc_min_C( prob, &param1), min_C);
		param1.p += stepSz;
	}
	printf("min_P %g max_P %g min_C %g max_C %g\n", log(min_P)/log(2.0), log(max_P)/log(2.0), log(min_C)/log(2.0), log(max_C)/log(2.0));

	// split data
	if (nr_fold > prob->l)
	{
		nr_fold = prob->l;
		fprintf(stderr,"WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
	}
	struct problem_folds *prob_folds = split_data(prob, nr_fold);

	//fix p run C
	double current_rate = INF, best_rate = INF;
	double best_P, best_C;
	param1.p = max_P;
	reset_iter_sum_whole_process();
	while( true ){
		//If next step is too close to zero, set it to zero.
		if( param1.p < stepSz / 2.0)
			param1.p = 0.0;

		reset_iter_sum_fix_one_param();
		find_parameter_fix_p_noWarm(prob, prob_folds, &param1, nr_fold, min_C, max_C, &best_C, &current_rate);
		if(best_rate > current_rate){
			best_P = param1.p;
			best_rate = current_rate;
		}
		
		if(param1.p == 0.0)
			printf("Cumulative logP : INF ");
		else
			printf("Cumulative logP : %g ", log2(param1.p) );
		print_iter_sum_whole_process();

		// param1.p is zero. It is the last iteration
		if( param1.p == 0.0 )
			break;
		else
			param1.p -= stepSz;
	}
	
	// Print the best result
	printf("======================================\n");
	print_iter_sum_whole_process();
	if( best_P == 0.0 )
		printf("Best logP = INF Best logC = %g Best MSE = %g \n", log2(best_C), best_rate );
	else
		printf("Best logP = %g Best logC = %g Best MSE = %g \n", log2(best_P), log2(best_C), best_rate );
}


void find_parameter_fix_p_noWarm(const problem *prob, const problem_folds *prob_folds, const parameter *param, int nr_fold, double min_C, double max_C, double *best_C, double *best_rate)
{
	printf("======================================\n");
	int *fold_start = prob_folds->fold_start;
	int l = prob->l;
	int *perm = prob_folds->perm;
	double *target = Malloc(double, l);
	struct problem *subprob = prob_folds->subprobs;

	// variables for warm start
	double ratio = 2;
	struct parameter param1 = *param;
	void (*default_print_string) (const char *) = liblinear_print_string;

	*best_rate = INF;
	param1.C = min_C;
	while(param1.C <= max_C)
	{
		//Output disabled for running CV at a particular C
		set_print_string_function(&print_null);

		reset_iter_sum();
		for(int i=0; i<nr_fold; i++)
		{
			int begin = fold_start[i];
			int end = fold_start[i+1];

			param1.init_sol = NULL;
			double eps = param1.eps;
			param1.eps = (1.0 - 0.1) * param1.eps;
			struct model *submodel = train(&subprob[i],&param1);

			for(int j=begin; j<end; j++)
				target[perm[j]] = predict(submodel,prob->x[perm[j]]);

			free_and_destroy_model(&submodel);
		}
		print_iter_sum( param1.p, param1.C);
		set_print_string_function(default_print_string);

		double current_rate = calc_error(prob, param, target);
		if( current_rate < *best_rate )
		{
			*best_C = param1.C;
			*best_rate = current_rate;
		}
		if(param1.p == 0.0)
			info("log2P= INF log2C= %7.2f\tMSE= %g\n",log2(param1.C), current_rate);
		else
			info("log2P= %7.2f log2C= %7.2f\tMSE= %g\n", log2(param1.p), log2(param1.C), current_rate);

		// Update next problem parameter
		param1.C = param1.C * ratio;
	}

	if(param1.C > max_C && max_C > min_C)
		info("Finish noWarm with max_C : %g \n", log2(max_C));
	free(target);
	param1.init_sol = NULL;
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
		loss += max( yi_abs - param->p, 0.0) * max(yi_abs - param->p, 0.0);
	}

	if(loss == 0.0){
		fprintf( stderr, "param->p is too large!!! Please try param->p / 2.0!!!\n");
		int lala;
		scanf("%d", &lala);
		exit(1);
	}

	double min_C = 1.0;
	if(param->solver_type == L2R_LR)
		min_C = 1.0 / (prob->l * max_xTx);
	else if(param->solver_type == L2R_L2LOSS_SVC)
		min_C = 1.0 / (2 * prob->l * max_xTx);
	else if(param->solver_type == L2R_L2LOSS_SVR)
		min_C = delta * delta * loss / (8.0 * phi * phi * max_xTx);

	return pow( 2, floor(log(min_C) / log(2.0)) );
	//return min_C;
}

double calc_max_P(const problem *prob, const parameter *param)
{
	double yi_abs, max_yi_abs = 0.0;
	double l = prob->l;
	for(int i = 0; i < l; i++)
	{
		yi_abs = (prob->y[i] >= 0)? prob->y[i] : -1.0 * prob->y[i];
		max_yi_abs = max( max_yi_abs, yi_abs);
	}

	double max_P = max_yi_abs;
	stepSz = max_P / numSteps;
	printf("Initialize numSteps: %d stepSz: %g\n", int(numSteps), stepSz);
	//max_P -= stepSz * max_PReduce;
	
	return max_P;
}

double get_l2r_l2_svr_fun_grad_norm(double *w, const problem *prob, const parameter *param){
	int n = prob->n, l = prob->l;
	double norm_grad;
	int inc = 1;
	double *g = new double[n];
	double *C = new double[l];
	double *w0;
	for(int j=0; j < l; j++)
		C[j] = param->C;
	function * fun_obj=new l2r_l2_svr_fun(prob, C, param->p);
	if( w != NULL ){
		fun_obj->fun(w);
		fun_obj->grad(w, g);
	}
	else{
		double *w0 = new double[n];
		for(int j = 0; j < n; j++)
			w0[j] = 0;
		fun_obj->fun(w0);
		fun_obj->grad(w0, g);
		free(w0);
	}
	norm_grad = dnrm2_(&n, g, &inc);
	free(C);
	free(fun_obj);
	free(g);
	return norm_grad;
}

double get_l2r_l2_svr_loss_norm(double *w, const problem *prob, const double p){
	int n = prob->n, l = prob->l;
	double norm_grad;
	int inc = 1;
	double *g = new double[n];
	double *C = new double[l];
	double *w0;
	for(int j=0; j < l; j++)
		C[j] = 1;
	function * fun_obj=new l2r_l2_svr_fun(prob, C, p);
	if( w != NULL ){
		fun_obj->fun(w);
		fun_obj->grad(w, g);
		for(int j = 0; j < n; j++){
			g[j] -= w[j];
		}
	}
	else{
		double *w0 = new double[n];
		for(int j = 0; j < n; j++)
			w0[j] = 0;
		fun_obj->fun(w0);
		fun_obj->grad(w0, g);
		free(w0);
	}
	norm_grad = dnrm2_(&n, g, &inc);
	free(C);
	free(fun_obj);
	free(g);
	return norm_grad;
}

/**
double calc_min_P(const problem *prob, const parameter *param)
{
	int n = prob->n, l = prob->l;
	
	//calc norm grad w0
	double norm_2Xy = get_l2r_l2_svr_loss_norm(NULL, prob, 0);
	
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
	printf("eps %lf norm_2Xy %lf max_x %lf n %d l %d\n", param->eps, norm_2Xy, max_x, n, l);
	double min_P = delta * norm_2Xy / ( 2.0 * max_x* (double) sqrt(n) * (double) l );

	return min_P;
	//return pow( 2, floor( log(min_P) / log(2.0) ));
}
**/

//
void find_parameter_classification(const problem *prob, const problem_folds *prob_folds, const parameter *param, int nr_fold, double start_C, double max_C, double *best_C, double *best_rate)
{
	// variables for CV
	int i;
	int *fold_start = prob_folds->fold_start;
	int l = prob->l;
	int *perm = prob_folds->perm;
	double *target = Malloc(double, l);
	struct problem *subprob = prob_folds->subprobs;

	// variables for warm start
	double ratio = 2;
	double **prev_w = Malloc(double*, nr_fold);
	for(i = 0; i < nr_fold; i++)
		prev_w[i] = NULL;
	int num_unchanged_w = 0;
	struct parameter param1 = *param;
	void (*default_print_string) (const char *) = liblinear_print_string;


	*best_rate = 0;
	if(start_C <= 0)
		start_C = calc_start_C(prob,param);
	param1.C = start_C;

	bool first_old_break = true;
	bool first_new_break = true;
	
	double ratio_eps[nr_fold];
	for(int i = 0 ; i < nr_fold; i++){
		int pos = 0;
		int neg = 0;
		for(int j=0; j<subprob[i].l;j++)
			if(subprob[i].y[j] > 0)
				pos++;
		neg = subprob[i].l - pos;
		ratio_eps[i] = (double)max(min(pos,neg), 1)/subprob[i].l;
	}
	
	while(param1.C <= max_C)
	{
		printf("====================C %g========================\n", log2(param1.C));
		//Output disabled for running CV at a particular C
		set_print_string_function(&print_null);
		
		reset_new_break();
		for(i=0; i<nr_fold; i++)
		{
			int j;
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
				for(j=0; j<total_w_size; j++)
					prev_w[i][j] = submodel->w[j];
			}
			else if(num_unchanged_w >= 0)
			{
				double norm_w_diff = 0;
				for(j=0; j<total_w_size; j++)
				{
					norm_w_diff += (submodel->w[j] - prev_w[i][j])*(submodel->w[j] - prev_w[i][j]);
					prev_w[i][j] = submodel->w[j];
				}
				norm_w_diff = sqrt(norm_w_diff);

				if(norm_w_diff > 1e-15)
					num_unchanged_w = -1;
			}
			else
			{
				for(j=0; j<total_w_size; j++)
					prev_w[i][j] = submodel->w[j];
			}

			for(j=begin; j<end; j++)
				target[perm[j]] = predict(submodel,prob->x[perm[j]]);

			free_and_destroy_model(&submodel);
		}
		set_print_string_function(default_print_string);

		int total_correct = 0;
		for(i=0; i<prob->l; i++)
			if(target[i] == prob->y[i])
				++total_correct;
		double current_rate = (double)total_correct/prob->l;
		if(current_rate > *best_rate)
		{
			*best_C = param1.C;
			*best_rate = current_rate;
		}

		printf("log2c=%7.2f\trate=%g\n",log2(param1.C),100.0*current_rate);
		num_unchanged_w++;
		
		//Check break condition
		if(num_unchanged_w == 3 && first_old_break == true){
			fprintf( stdout ,"Old Break C: %g rate= %g \n", log2(param1.C), 100 * *best_rate ) ;
			first_old_break = false;
		}
		
		if( get_new_break() == nr_fold && first_new_break == true){
			fprintf( stdout ,"New Break C: %g rate= %g \n", log2(param1.C), 100 * *best_rate );
			first_new_break = false;
		}
		if(first_old_break == false && first_new_break == false){
			break;
		}	
		param1.C = param1.C*ratio;
	}

	if(param1.C > max_C && max_C > start_C)
		info("warning: maximum C reached.\n");
	free(fold_start);
	free(perm);
	free(target);
	for(i=0; i<nr_fold; i++)
	{
		free(subprob[i].x);
		free(subprob[i].y);
		free(prev_w[i]);
	}
	free(prev_w);
	free(subprob);
}


int new_break_check = 0;

void reset_new_break(){
	new_break_check = 0;
}

void add_new_break(){
	new_break_check++;
}

int get_new_break(){
	return new_break_check;
}


// add_iter_sum_fix_one_param count all cg_iter when fix one parameter and search the other param in the defined range.
int total_iter_sum_fix_one_param;

void add_iter_sum_fix_one_param(int num){
	total_iter_sum_fix_one_param += num;
}

void reset_iter_sum_fix_one_param()
{
	total_iter_sum_fix_one_param = 0;
}

void print_iter_sum_fix_one_param(char c, double val)
{
	if( val == 0.0)
		printf("cur_log%c: INF iter_sum: %d\n", c, total_iter_sum_fix_one_param);
	else
		printf("cur_log%c: %g iter_sum: %d\n", c, log2(val), total_iter_sum_fix_one_param);
}

// add_iter count the cg_iter for parameter (P,C) search total cg_iter
int total_iter_sum;

void add_iter(int num)
{
	total_iter_sum += num;
}

void reset_iter_sum(){
	total_iter_sum = 0;
}

void print_iter_sum(double p, double C){
	if( p == 0.0 )
		printf("iter_sum: %d P: INF C: %g\n", total_iter_sum,  log2(C));
	else
		printf("iter_sum: %d P: %g C: %g\n", total_iter_sum, log2(p), log2(C));
}

// Iteration sum for whole process.

int iter_sum_whole_process;

void reset_iter_sum_whole_process(){
	iter_sum_whole_process = 0;
}

void update_iter_sum_whole_process(){
	// Update for each fix one parameter process.
	iter_sum_whole_process += total_iter_sum_fix_one_param;
}

void print_iter_sum_whole_process(){
	printf("Iteration sum of whole process (new break) : %d\n", iter_sum_whole_process);
}

double get_l2r_lr_loss_norm(double *w, const problem *prob){
	//printf("Warning does not use weighted label\n");
	int n = prob->n, l = prob->l;
	double norm_grad;
	int inc = 1;
	double *g = new double[n];
	double *C = new double[l];
	double *w0;
	for(int j=0; j < l; j++)
		C[j] = 1;
	function * fun_obj=new l2r_lr_fun(prob, C);
	if( w != NULL ){
		fun_obj->fun(w);
		fun_obj->grad(w, g);
		for(int j = 0; j < n; j++){
			g[j] -= w[j];
		}
	}
	else{
		double *w0 = new double[n];
		for(int j = 0; j < n; j++)
			w0[j] = 0;
		fun_obj->fun(w0);
		fun_obj->grad(w0, g);
		free(w0);
	}
	norm_grad = dnrm2_(&n, g, &inc);
	free(C);
	free(fun_obj);
	free(g);
	return norm_grad;
}


// Show go P for each parameter C. Note that we have to search from big P to small P.



void find_parameter_linear_step_fixC_goP(const problem *prob, const parameter *param, int nr_fold)
{
	// Informatino about this method

	printf("This search direction cannot adopt the C stop method.\n");
	//Set range of parameter
	double ratio = 2.0;
	double min_P = 0.0;
	double max_P = calc_max_P(prob, param);
	double max_C = pow(2.0, 50);
	double min_C = INF;
	struct parameter param1 = *param;
	param1.p = min_P;
	while( param1.p < max_P ){
		min_C = min( calc_min_C( prob, &param1), min_C);
		param1.p += stepSz;
	}
	printf("min_P %g max_P %g min_C %g max_C %g\n", log(min_P)/log(2.0), log(max_P)/log(2.0), log(min_C)/log(2.0), log(max_C)/log(2.0));
	
	// split data
	if (nr_fold > prob->l)
	{
		nr_fold = prob->l;
		fprintf(stderr,"WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
	}
	struct problem_folds *prob_folds = split_data(prob, nr_fold);

	// Go P fix C.
	double current_rate = INF, best_rate = INF;
	double best_P, best_C;
	param1.C = min_C;
	reset_iter_sum_whole_process();
	while(param1.C < max_C){
		reset_iter_sum_fix_one_param();
		find_parameter_fix_c(prob, prob_folds, &param1, nr_fold, min_P, max_P, &best_P, &current_rate);
		update_iter_sum_whole_process();
		printf("New Break Iteration: ");
		print_iter_sum_fix_one_param('C', param1.C);
		printf("Cumulative logC : %g ", log2(param1.C));
		print_iter_sum_whole_process();
		if(best_rate > current_rate){
			best_C = param1.C;
			best_rate = current_rate;
		}
		param1.C *= ratio;
	}

	// Print Final Result
	printf("======================================\n");
	if(best_P == 0.0)
		printf("Best logP = INF Best logC = %g Best MSE = %g \n", log2(best_C), best_rate );
	else
		printf("Best logP = %g Best logC = %g Best MSE = %g \n", log2(best_P), log2(best_C), best_rate );
}


void find_parameter_fix_c(const problem *prob, const problem_folds *prob_folds, const parameter *param, int nr_fold, double min_P, double max_P, double *best_P, double *best_rate)
{
	printf("======================================\n");
	int *fold_start = prob_folds->fold_start;
	int l = prob->l;
	int *perm = prob_folds->perm;
	double *target = Malloc(double, l);
	struct problem *subprob = prob_folds->subprobs;

	// variables for warm start
	double **prev_w = Malloc(double*, nr_fold);
	for(int i = 0; i < nr_fold; i++)
		prev_w[i] = NULL;
	struct parameter param1 = *param;
	void (*default_print_string) (const char *) = liblinear_print_string;

	//smaller best rate is better for regression
	*best_rate = INF;

	param1.p = max_P;
	while(true)
	{
		// If step is too small, run the last step with param1.p = 0.0 and then break the while loop.
		if( param1.p <= stepSz / 2.0 )
			param1.p = 0.0;

		//Output disabled for running CV at a particular C
		set_print_string_function(&print_null);

		reset_iter_sum();
		for(int i=0; i<nr_fold; i++){
			int begin = fold_start[i];
			int end = fold_start[i+1];

			param1.init_sol = prev_w[i];
			double eps = param1.eps;
			param1.eps = (1.0 - 0.1) * param1.eps;
			struct model *submodel = train(&subprob[i],&param1);
			param1.eps = eps;

			int total_w_size;
			if(submodel->nr_class == 2)
				total_w_size = subprob[i].n;
			else
				total_w_size = subprob[i].n * submodel->nr_class;

			if(prev_w[i] == NULL)
			{
				prev_w[i] = Malloc(double, total_w_size);
			}

			for(int j=0; j<total_w_size; j++)
				prev_w[i][j] = submodel->w[j];

			for(int j=begin; j<end; j++)
				target[perm[j]] = predict(submodel,prob->x[perm[j]]);

			free_and_destroy_model(&submodel);
		}
		print_iter_sum( param1.p, param1.C);
		set_print_string_function(default_print_string);

		double current_rate = calc_error(prob, param, target);
		if( current_rate < *best_rate){
			*best_P = param1.p;
			*best_rate = current_rate;
		}
		if(param1.p == 0.0)
			info("log2P= INF log2C= %7.2f\tMSE= %g\n",log2(param1.C), current_rate);
		else
			info("log2P= %7.2f log2C= %7.2f\tMSE= %g\n", log2(param1.p), log2(param1.C), current_rate);
	
		// Check if the it is the last step with param1.p = 0.0
		if( param1.p == 0.0 )
			break;
		else
			param1.p = param1.p - stepSz;
	}

	free(target);
	for(int i=0; i<nr_fold; i++)
		free(prev_w[i]);
	free(prev_w);
	param1.init_sol = NULL;
}




#ifdef __cplusplus
}
#endif
