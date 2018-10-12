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


#ifdef __cplusplus
extern "C" {
#endif
// define in blas/dnrm2_ file
extern double dnrm2_(int *, double *, int *);

const int numSteps = 20.0;
double stepSz;
const double max_PReduce = 5.0;
const double delta1 = 1e-5;

problem_folds::~problem_folds()
{
  free(perm);
  free(fold_start);
  for(int i = 0; i < nr_fold; i++)
  {
    free( init_sols[i] );
    free( (subprobs+i)->x);
    free( (subprobs+i)->y);
  }
  free(init_sols);
  free(subprobs);
}

void reset_init_sols(problem_folds *prob_folds)
{
  int nr_fold = prob_folds->nr_fold;
  double **init_sols = prob_folds->init_sols;
  for(int i = 0; i < nr_fold; i++)
  {
    delete [] init_sols[i];
    init_sols[i] = NULL;
  }
}

bool score_type(const struct parameter &param){
	if(param.solver_type == L2R_L2LOSS_SVR ||
	   param.solver_type == L2R_L1LOSS_SVR_DUAL ||
	   param.solver_type == L2R_L2LOSS_SVR_DUAL)
	{
    return true;
	}
	else
	{
	  return false;
  }
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
  prob_folds->perm = Malloc(int, l);
  prob_folds->nr_fold = nr_fold;
  prob_folds->fold_start = Malloc(int, nr_fold+1);
  prob_folds->init_sols = Malloc(double*, nr_fold);
  prob_folds->subprobs = Malloc(problem, nr_fold);

  int *perm = prob_folds->perm;
  int *fold_start = prob_folds->fold_start;
  double **init_sols = prob_folds->init_sols;
  problem *subprobs = prob_folds->subprobs;
  
  srand(1);
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
    init_sols[i] = NULL;
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

void cross_validation_with_splits(const problem *prob, const problem_folds *prob_folds,const parameter *param, int nr_fold, double &score, bool &w_diff)
{
  //Fold data
  int *perm = prob_folds->perm;
  int *fold_start = prob_folds->fold_start;
  double **init_sols = prob_folds->init_sols;
  struct problem *subprob = prob_folds->subprobs;

  double *target = Malloc(double, prob->l);
  struct parameter param1 = *param;

  set_print_string_function(&print_null);
  w_diff = false;
  for(int i=0; i<nr_fold; i++)
  {
    param1.init_sol = init_sols[i];
    struct model *submodel = train(&subprob[i],&param1);

    int total_w_size;
    if(submodel->nr_class == 2)
      total_w_size = subprob[i].n;
    else
      total_w_size = subprob[i].n * submodel->nr_class;

    if(init_sols[i] == NULL)
    {
      init_sols[i] = Malloc(double, total_w_size);
    }
    else
    {
      double norm_w_diff = 0;
      for(int j=0; j<total_w_size; j++)
        norm_w_diff += (submodel->w[j] - init_sols[i][j])*(submodel->w[j] - init_sols[i][j]);
      norm_w_diff = sqrt(norm_w_diff);

      if(norm_w_diff != 0.0){
        w_diff  = true;
      }
    }

    for(int j=0; j<total_w_size; j++)
      init_sols[i][j] = submodel->w[j];

    //Predict value on i-th fold
    int begin = fold_start[i];
    int end = fold_start[i+1];
    for(int j=begin; j<end; j++)
      target[perm[j]] = predict(submodel,prob->x[perm[j]]);

    free_and_destroy_model(&submodel);
  }
  
  score = calc_error(prob, param, target);
  free(target);
}


void linear_step_fix_range_nowarm(const problem *prob,const parameter *param, int nr_fold)
{
  //Set range of parameter
  double min_P = 0.0;
  double max_P = calc_max_P(prob, param);
  //double min_C = INF;
  double max_C = pow(2.0, 50);

  //Split data
  struct problem_folds *prob_folds = split_data(prob, nr_fold);

  //Best score
  double best_score = INF;
  double best_P=-1, best_C=-1;

  //Run
  struct parameter param1 = *param;
  stepSz = max_P / double(numSteps);
  printf("Initialize numSteps: %d stepSz: %10.5f\n", numSteps, stepSz);
  for(int i = numSteps - 1; i >= 0; i--)
  {
    param1.p = stepSz * i;
    param1.C = calc_min_C(prob, &param1);
    while( param1.C < max_C )
    {
      double score = -1;
      bool w_diff = false;
      reset_init_sols(prob_folds);
      reset_iter_sum();
      cross_validation_with_splits(prob, prob_folds, &param1, nr_fold, score, w_diff);
      print_iter_sum(param1.p, param1.C);
      if(param1.p == 0.0)
        printf("log2P: INF log2C: %10.5f MSE: %10.5f\n",log2(param1.C), score);
      else
        printf("log2P: %10.5f log2C: %10.5f MSE: %10.5f\n", log2(param1.p), log2(param1.C), score);
      if(best_score > score){
        best_C = param1.C;
        best_P = param1.p;
        best_score = score;
      }
      param1.C *= 2.0;
    }
  }
  
  // Print the best result
  printf("======================================\n");
  if( best_P == 0.0 )
    printf("Best log2P: INF Best log2C: %10.5f Best MSE: %10.5f \n", log2(best_C), best_score );
  else
    printf("Best log2P: %10.5f Best log2C: %10.5f Best MSE: %10.5f \n", log2(best_P), log2(best_C), best_score );
}

void linear_step_fix_range(const problem *prob,const parameter *param, int nr_fold)
{
  //Set range of parameter
  double min_P = 0.0;
  double max_P = calc_max_P(prob, param);
  //double min_C = INF;
  double max_C = pow(2.0, 50);

  //Split data
  struct problem_folds *prob_folds = split_data(prob, nr_fold);

  //Best score
  double best_score = INF;
  double best_P=-1, best_C=-1;

  //Run
  struct parameter param1 = *param;
  stepSz = max_P / double(numSteps);
  printf("Initialize numSteps: %d stepSz: %10.5f\n", numSteps, stepSz);
  for(int i = numSteps - 1; i >= 0; i--)
  {
    param1.p = stepSz * i;
    param1.C = calc_min_C(prob, &param1);
    reset_init_sols(prob_folds);
    while( param1.C < max_C )
    {
      double score = -1;
      bool w_diff = false;
      reset_iter_sum();
      cross_validation_with_splits(prob, prob_folds, &param1, nr_fold, score, w_diff);
      print_iter_sum(param1.p, param1.C);
      if(param1.p == 0.0)
        printf("log2P: INF log2C: %10.5f MSE: %10.5f\n",log2(param1.C), score);
      else
        printf("log2P: %10.5f log2C: %10.5f MSE: %10.5f\n", log2(param1.p), log2(param1.C), score);
      if(best_score > score){
        best_C = param1.C;
        best_P = param1.p;
        best_score = score;
      }
      param1.C *= 2.0;
    }
  }
  
  // Print the best result
  printf("======================================\n");
  if( best_P == 0.0 )
    printf("Best log2P: INF Best log2C: %10.5f Best MSE: %10.5f \n", log2(best_C), best_score );
  else
    printf("Best log2P: %10.5f Best log2C: %10.5f Best MSE: %10.5f \n", log2(best_P), log2(best_C), best_score );
}


void log_step_fix_range_nowarm(const problem *prob,const parameter *param, int nr_fold)
{
  //Set range of parameter
  double min_P = pow(2.0, -30);
  double max_P = calc_max_P(prob, param);
  //double min_C = INF;
  double max_C = pow(2.0, 50);

  //Split data
  struct problem_folds *prob_folds = split_data(prob, nr_fold);

  //Best score
  double best_score = INF;
  double best_P=-1, best_C=-1;

  //Run
  struct parameter param1 = *param;
  double ratio = 2.0;
  param1.p = max_P;
  while( param1.p > min_P )
  {
    param1.p /= ratio;
    param1.C = calc_min_C(prob, &param1);
    while( param1.C < max_C )
    {
      double score = -1;
      bool w_diff = false;
      reset_init_sols(prob_folds);
      reset_iter_sum();
      cross_validation_with_splits(prob, prob_folds, &param1, nr_fold, score, w_diff);
      print_iter_sum(param1.p, param1.C);
      if(param1.p == 0.0)
        printf("log2P: INF log2C: %10.5f MSE: %10.5f\n",log2(param1.C), score);
      else
        printf("log2P: %10.5f log2C: %10.5f MSE: %10.5f\n", log2(param1.p), log2(param1.C), score);
      if(best_score > score){
        best_C = param1.C;
        best_P = param1.p;
        best_score = score;
      }
      param1.C *= ratio;
    }
  }
  
  // Print the best result
  printf("======================================\n");
  if( best_P == 0.0 )
    printf("Best log2P: INF Best log2C: %10.5f Best MSE: %10.5f \n", log2(best_C), best_score );
  else
    printf("Best log2P: %10.5f Best log2C: %10.5f Best MSE: %10.5f \n", log2(best_P), log2(best_C), best_score );
}

void log_step_fix_range(const problem *prob,const parameter *param, int nr_fold)
{
  //Set range of parameter
  double min_P = pow(2.0, -30);
  double max_P = calc_max_P(prob, param);
  //double min_C = INF;
  double max_C = pow(2.0, 50);

  //Split data
  struct problem_folds *prob_folds = split_data(prob, nr_fold);

  //Best score
  double best_score = INF;
  double best_P=-1, best_C=-1;

  //Run
  struct parameter param1 = *param;
  double ratio = 2.0;
  param1.p = max_P;
  while( param1.p > min_P )
  {
    param1.p /= ratio;
    param1.C = calc_min_C(prob, &param1);
    reset_init_sols(prob_folds);
    while( param1.C < max_C )
    {
      double score = -1;
      bool w_diff = false;
      reset_iter_sum();
      cross_validation_with_splits(prob, prob_folds, &param1, nr_fold, score, w_diff);
      print_iter_sum(param1.p, param1.C);
      if(param1.p == 0.0)
        printf("log2P: INF log2C: %10.5f MSE: %10.5f\n",log2(param1.C), score);
      else
        printf("log2P: %10.5f log2C: %10.5f MSE: %10.5f\n", log2(param1.p), log2(param1.C), score);
      if(best_score > score){
        best_C = param1.C;
        best_P = param1.p;
        best_score = score;
      }
      param1.C *= ratio;
    }
  }
  
  // Print the best result
  printf("======================================\n");
  if( best_P == 0.0 )
    printf("Best log2P: INF Best log2C: %10.5f Best MSE: %10.5f \n", log2(best_C), best_score );
  else
    printf("Best log2P: %10.5f Best log2C: %10.5f Best MSE: %10.5f \n", log2(best_P), log2(best_C), best_score );
}

void C_P_new(const problem *prob,const parameter *param, int nr_fold)
{
  //Set range of parameter
  double min_P = 0.0;
  double max_P = calc_max_P(prob, param);
  double min_C = INF;
  double max_C = pow(2.0, 50);

  //Split data
  struct problem_folds *prob_folds = split_data(prob, nr_fold);

  //Best score
  double best_score = INF;
  double best_P=-1, best_C=-1;

  //Run
  struct parameter param1 = *param;
  stepSz = max_P / double(numSteps);
  double ratio = 2.0;
  printf("Initialize numSteps: %d stepSz: %10.5f\n", numSteps, stepSz);
  for(int i = numSteps - 1; i >= 0; i--)
  {
    param1.p = stepSz * i; 
    min_C = min(min_C, calc_min_C(prob, &param1));
  }
  param1.C = min_C;
  long long pass_set = 0;
  long long break_cond = 1 << (numSteps);
  while( param1.C < max_C )
  {
    printf("=======\n");
    reset_init_sols(prob_folds);
    for(int i = numSteps - 1; i >= 0; i--)
    {
      if( pass_set & (1 << i) )
        continue;
      
      param1.p = stepSz * i;
      double score = -1;
      bool w_diff = false;
      
      reset_new_break();
      reset_iter_sum();
      param1.eps = (1.0 - delta1) * param->eps;
      cross_validation_with_splits(prob, prob_folds, &param1, nr_fold, score, w_diff);
      param1.eps = param->eps;
      print_iter_sum(param1.p, param1.C);

      if(param1.p == 0.0)
        printf("log2P: INF log2C: %10.5f MSE: %10.5f\n",log2(param1.C), score);
      else
        printf("log2P: %10.5f log2C: %10.5f MSE: %10.5f\n", log2(param1.p), log2(param1.C), score);
      if(best_score > score)
      {
        best_C = param1.C;
        best_P = param1.p;
        best_score = score;
      }
      if(get_new_break () == nr_fold)
        pass_set += 1 << i;
    }
    if( pass_set == break_cond - 1)
      break;
    param1.C *= ratio;
  }
  
  // Print the best result
  printf("======================================\n");
  if( best_P == 0.0 )
    printf("Best log2P: INF Best log2C: %10.5f Best MSE: %10.5f \n", log2(best_C), best_score );
  else
    printf("Best log2P: %10.5f Best log2C: %10.5f Best MSE: %10.5f \n", log2(best_P), log2(best_C), best_score );
}

void P_C_new(const problem *prob,const parameter *param, int nr_fold)
{
  //Set range of parameter
  double min_P = 0.0;
  double max_P = calc_max_P(prob, param);
  //double min_C = INF;
  double max_C = pow(2.0, 50);

  //Split data
  struct problem_folds *prob_folds = split_data(prob, nr_fold);

  //Best score
  double best_score = INF;
  double best_P=-1, best_C=-1;

  //Run
  struct parameter param1 = *param;
  stepSz = max_P / double(numSteps);
  printf("Initialize numSteps: %d stepSz: %10.5f\n", numSteps, stepSz);
  for(int i = numSteps - 1; i >= 0; i--)
  {
    param1.p = stepSz * i;
    param1.C = calc_min_C(prob, &param1);
    reset_init_sols(prob_folds);
    while( param1.C < max_C )
    {
      double score = -1;
      bool w_diff = false;

      reset_new_break();
      reset_iter_sum();
      param1.eps = (1 - delta1) * param->eps;
      cross_validation_with_splits(prob, prob_folds, &param1, nr_fold, score, w_diff);
      print_iter_sum( param1.p, param1.C );
      param1.eps = param->eps;

      if(param1.p == 0.0)
        printf("log2P: INF log2C: %10.5f MSE: %10.5f\n",log2(param1.C), score);
      else
        printf("log2P: %10.5f log2C: %10.5f MSE: %10.5f\n", log2(param1.p), log2(param1.C), score);
      if(best_score > score){
        best_C = param1.C;
        best_P = param1.p;
        best_score = score;
      }
      if( get_new_break() == nr_fold )
        break;
      param1.C *= 2.0;
    }
  }
  
  // Print the best result
  printf("======================================\n");
  if( best_P == 0.0 )
    printf("Best log2P: INF Best log2C: %10.5f Best MSE: %10.5f \n", log2(best_C), best_score );
  else
    printf("Best log2P: %10.5f Best log2C: %10.5f Best MSE: %10.5f \n", log2(best_P), log2(best_C), best_score );
}


void P_C_old(const problem *prob,const parameter *param, int nr_fold)
{
  //Set range of parameter
  double min_P = 0.0;
  double max_P = calc_max_P(prob, param);
  //double min_C = INF;
  double max_C = pow(2.0, 50);

  //Split data
  struct problem_folds *prob_folds = split_data(prob, nr_fold);

  //Best score
  double best_score = INF;
  double best_P=-1, best_C=-1;

  //Run
  struct parameter param1 = *param;
  stepSz = max_P / double(numSteps);
  printf("Initialize numSteps: %d stepSz: %10.5f\n", numSteps, stepSz);
  for(int i = numSteps - 1; i >= 0; i--)
  {
    param1.p = stepSz * i;
    param1.C = calc_min_C(prob, &param1);
    int w_diff_cnt = -1;
    reset_init_sols(prob_folds);
    while( param1.C < max_C )
    {
      double score = -1;
      bool w_diff = false;

      reset_iter_sum();
      cross_validation_with_splits(prob, prob_folds, &param1, nr_fold, score, w_diff);
      print_iter_sum( param1.p, param1.C );

      if(param1.p == 0.0)
        printf("log2P: INF log2C: %10.5f MSE: %10.5f\n",log2(param1.C), score);
      else
        printf("log2P: %10.5f log2C: %10.5f MSE: %10.5f\n", log2(param1.p), log2(param1.C), score);
      if(best_score > score){
        best_C = param1.C;
        best_P = param1.p;
        best_score = score;
      }
      
      if(w_diff) w_diff_cnt = -1;
      w_diff_cnt++;
      if(w_diff_cnt == 3)
        break;
      
      param1.C *= 2.0;
    }
  }
  
  // Print the best result
  printf("======================================\n");
  if( best_P == 0.0 )
    printf("Best log2P: INF Best log2C: %10.5f Best MSE: %10.5f \n", log2(best_C), best_score );
  else
    printf("Best log2P: %10.5f Best log2C: %10.5f Best MSE: %10.5f \n", log2(best_P), log2(best_C), best_score );
}

void full_nowarm_fix_p(const problem *prob,const parameter *param, int nr_fold)
{
  //Set range of parameter
  double min_P = 0.0;
  double max_P = calc_max_P(prob, param);
  //double min_C = INF;
  double max_C = pow(2.0, 50);

  //Split data
  struct problem_folds *prob_folds = split_data(prob, nr_fold);

  //Best score
  double best_score = INF;
  double best_P=-1, best_C=-1;

  //Run
  struct parameter param1 = *param;
  stepSz = max_P / double(numSteps);
  printf("Initialize numSteps: %d stepSz: %10.5f\n", numSteps, stepSz);

  param1.p = stepSz * param->p;
  printf("param1.p %f step %f\n", param1.p, param->p);
  param1.C = calc_min_C(prob, &param1);

  while( param1.C < max_C )
  {
    double score = -1;
    bool w_diff = false;
    
    reset_new_break();
    reset_iter_sum();
    reset_init_sols(prob_folds);
    cross_validation_with_splits(prob, prob_folds, &param1, nr_fold, score, w_diff);
    print_iter_sum( param1.p, param1.C );
    
    if(param1.p == 0.0)
      printf("log2P: INF log2C: %10.5f MSE: %10.5f\n",log2(param1.C), score);
    else
      printf("log2P: %10.5f log2C: %10.5f MSE: %10.5f\n", log2(param1.p), log2(param1.C), score);
    if(best_score > score){
      best_C = param1.C;
      best_P = param1.p;
      best_score = score;
    }
    param1.C *= 2.0;
  }
  
  // Print the best result
  printf("======================================\n");
  if( best_P == 0.0 )
    printf("Best log2P: INF Best log2C: %10.5f Best MSE: %10.5f \n", log2(best_C), best_score );
  else
    printf("Best log2P: %10.5f Best log2C: %10.5f Best MSE: %10.5f \n", log2(best_P), log2(best_C), best_score );
}

void FP_C_nowarm(const problem *prob,const parameter *param, int nr_fold)
{
  //Set range of parameter
  double min_P = 0.0;
  double max_P = calc_max_P(prob, param);
  //double min_C = INF;
  double max_C = pow(2.0, 50);

  //Split data
  struct problem_folds *prob_folds = split_data(prob, nr_fold);

  //Best score
  double best_score = INF;
  double best_P=-1, best_C=-1;

  //Run
  struct parameter param1 = *param;
  stepSz = max_P / double(numSteps);
  printf("Initialize numSteps: %d stepSz: %10.5f\n", numSteps, stepSz);

  param1.p = stepSz * param->p;
  printf("param1.p %f step %f\n", param1.p, param->p);
  param1.C = calc_min_C(prob, &param1);

  while( param1.C < max_C )
  {
    double score = -1;
    bool w_diff = false;
    
    reset_new_break();
    reset_iter_sum();
    reset_init_sols(prob_folds);
    param1.eps = (1 - delta1) * param->eps;
    cross_validation_with_splits(prob, prob_folds, &param1, nr_fold, score, w_diff);
    param1.eps = param->eps;
    print_iter_sum( param1.p, param1.C );
    
    if(param1.p == 0.0)
      printf("log2P: INF log2C: %10.5f MSE: %10.5f\n",log2(param1.C), score);
    else
      printf("log2P: %10.5f log2C: %10.5f MSE: %10.5f\n", log2(param1.p), log2(param1.C), score);
    if(best_score > score){
      best_C = param1.C;
      best_P = param1.p;
      best_score = score;
    }
    if( get_new_break() == nr_fold )
      break;
    param1.C *= 2.0;
  }
  
  // Print the best result
  printf("======================================\n");
  if( best_P == 0.0 )
    printf("Best log2P: INF Best log2C: %10.5f Best MSE: %10.5f \n", log2(best_C), best_score );
  else
    printf("Best log2P: %10.5f Best log2C: %10.5f Best MSE: %10.5f \n", log2(best_P), log2(best_C), best_score );
}

void P_C_nowarm(const problem *prob,const parameter *param, int nr_fold)
{
  //Set range of parameter
  double min_P = 0.0;
  double max_P = calc_max_P(prob, param);
  //double min_C = INF;
  double max_C = pow(2.0, 50);

  //Split data
  struct problem_folds *prob_folds = split_data(prob, nr_fold);

  //Best score
  double best_score = INF;
  double best_P=-1, best_C=-1;

  //Run
  struct parameter param1 = *param;
  stepSz = max_P / double(numSteps);
  printf("Initialize numSteps: %d stepSz: %10.5f\n", numSteps, stepSz);
  for(int i = numSteps - 1; i >= 0; i--)
  {
    param1.p = stepSz * i;
    param1.C = calc_min_C(prob, &param1);
    while( param1.C < max_C )
    {
      double score = -1;
      bool w_diff = false;
      
      reset_new_break();
      reset_iter_sum();
      reset_init_sols(prob_folds);
      param1.eps = (1 - delta1) * param->eps;
      cross_validation_with_splits(prob, prob_folds, &param1, nr_fold, score, w_diff);
      param1.eps = param->eps;
      print_iter_sum( param1.p, param1.C );
      
      if(param1.p == 0.0)
        printf("log2P: INF log2C: %10.5f MSE: %10.5f\n",log2(param1.C), score);
      else
        printf("log2P: %10.5f log2C: %10.5f MSE: %10.5f\n", log2(param1.p), log2(param1.C), score);
      if(best_score > score){
        best_C = param1.C;
        best_P = param1.p;
        best_score = score;
      }
      if( get_new_break() == nr_fold )
        break;
      param1.C *= 2.0;
    }
  }
  
  // Print the best result
  printf("======================================\n");
  if( best_P == 0.0 )
    printf("Best log2P: INF Best log2C: %10.5f Best MSE: %10.5f \n", log2(best_C), best_score );
  else
    printf("Best log2P: %10.5f Best log2C: %10.5f Best MSE: %10.5f \n", log2(best_P), log2(best_C), best_score );
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
    return total_error / prob->l;
  }
  else
  {
    for(int i=0;i<prob->l;i++)
      if(target[i] == prob->y[i])
        ++total_correct;
    return (double) total_correct / prob->l;
  }
}


double calc_min_C(const problem *prob, const parameter *param)
{
  int i;
  double xTx,max_xTx;
  double phi, loss, yi_abs;
  double delta2 = 0.1;
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
    phi += max( yi_abs, 0.0 );
    loss += max( yi_abs - param->p, 0.0) * max(yi_abs - param->p, 0.0);
  }

  if(loss == 0.0){
    fprintf( stderr, "param->p is too large!!!\n");
    exit(1);
  }

  double min_C = 1.0;
  if(param->solver_type == L2R_LR)
    min_C = 1.0 / (prob->l * max_xTx);
  else if(param->solver_type == L2R_L2LOSS_SVC)
    min_C = 1.0 / (2 * prob->l * max_xTx);
  else if(param->solver_type == L2R_L2LOSS_SVR)
    min_C = delta2 * delta2 * loss / (8.0 * phi * phi * max_xTx);

  return pow( 2, floor(log(min_C) / log(2.0)) );
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
  return max_P;
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
    printf("iter_sum: %d log2P: INF log2C: %10.5f\n", total_iter_sum,  log2(C));
  else
    printf("iter_sum: %d log2P: %10.5f log2C: %10.5f\n", total_iter_sum, log2(p), log2(C));
}

void cls_new(const problem *prob,const parameter *param, int nr_fold)
{
  //Set range of parameter
  double min_C = calc_min_C(prob, param);
  double max_C = pow(2.0, 50);

  //Split data
  struct problem_folds *prob_folds = split_data(prob, nr_fold);

  //Best score
  double best_score = 0;
  double best_C=-1;

  //Run
  struct parameter param1 = *param;
  param1.eps = (1 - delta1) * param->eps;
  param1.C = min_C;
  int w_diff_cnt = -1;
  while( param1.C < max_C )
  {
    double score = -1;
    bool w_diff = false;

    reset_new_break();
    reset_iter_sum();
    cross_validation_with_splits(prob, prob_folds, &param1, nr_fold, score, w_diff);
    print_iter_sum( 0, param1.C );

    printf("log2C: %10.5f Acc: %10.5f\n",log2(param1.C), score);
    if(best_score < score){
      best_C = param1.C;
      best_score = score;
    }
    
    if( get_new_break() == nr_fold )
      break;
    
    param1.C *= 2.0;
  }
  
  // Print the best result
  printf("======================================\n");
  printf("Best log2C: %10.5f Best Acc: %10.5f\n",log2(param1.C), best_score);
}

void cls_old(const problem *prob,const parameter *param, int nr_fold)
{
  //Set range of parameter
  double min_C = calc_min_C(prob, param);
  double max_C = pow(2.0, 50);

  //Split data
  struct problem_folds *prob_folds = split_data(prob, nr_fold);

  //Best score
  double best_score = 0;
  double best_C=-1;

  //Run
  struct parameter param1 = *param;
  param1.C = min_C;
  int w_diff_cnt = -1;
  while( param1.C < max_C )
  {
    double score = -1;
    bool w_diff = false;

    reset_iter_sum();
    cross_validation_with_splits(prob, prob_folds, &param1, nr_fold, score, w_diff);
    print_iter_sum( 0, param1.C );

    printf("log2C: %10.5f Acc: %10.5f\n",log2(param1.C), score);
    if(best_score < score){
      best_C = param1.C;
      best_score = score;
    }
    
    if(w_diff) w_diff_cnt = -1;
    w_diff_cnt++;
    if(w_diff_cnt == 3)
      break;
    
    param1.C *= 2.0;
  }
  
  // Print the best result
  printf("======================================\n");
  printf("Best log2C: %10.5f Best Acc: %10.5f\n",log2(param1.C), best_score);
}

#ifdef __cplusplus
}
#endif
