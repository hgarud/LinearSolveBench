#include "HYPRE_parcsr_ls.h"

enum
{
   NS_RESTART = 32,
   NS_LD = NS_RESTART + 1,
   NS_R = 0,
   NS_SHADOW,
   NS_P,
   NS_IMAGE,
   NS_Z,
   NS_T,
   NS_TRIAL,
   NS_ACTUAL,
   NS_WORK_COUNT
};

typedef struct
{
   hypre_Solver base;
   HYPRE_Solver amg;
   HYPRE_Matrix matrix;
   HYPRE_Real rtol;
   HYPRE_Real atol;
   HYPRE_Int maxiter;
   HYPRE_Int capacity;
   HYPRE_Int strict;

   void *work[NS_WORK_COUNT];
   void **v;
   void **z;

   HYPRE_Real h[NS_LD * NS_RESTART];
   HYPRE_Real cs[NS_RESTART];
   HYPRE_Real sn[NS_RESTART];
   HYPRE_Real g[NS_LD];
   HYPRE_Real y[NS_RESTART];
   HYPRE_Real dots[NS_LD];
   HYPRE_Complex weights[NS_RESTART];
} nsl_Data;

static HYPRE_Int nsl_setup(HYPRE_Solver, HYPRE_Matrix,
                          HYPRE_Vector, HYPRE_Vector);
static HYPRE_Int nsl_solve(HYPRE_Solver, HYPRE_Matrix,
                          HYPRE_Vector, HYPRE_Vector);
static HYPRE_Int nsl_destroy(HYPRE_Solver);

#define NS_CALL(call)                                      \
   do                                                     \
   {                                                      \
      HYPRE_Int ns_status_ = (call);                       \
      if (ns_status_ != 0)                                \
      {                                                   \
         return ns_status_;                               \
      }                                                   \
   } while (0)

static HYPRE_Real
nsl_length(void *x)
{
   HYPRE_Real q = hypre_ParKrylovInnerProd(x, x);
   if (!isfinite(q) || q < 0.0)
   {
      return NAN;
   }
   return sqrt(q);
}

static HYPRE_Int
nsl_free_batch(void ***address, HYPRE_Int count)
{
   void **batch = *address;
   HYPRE_Int error = 0;
   HYPRE_Int status;
   HYPRE_Int i;

   if (batch == NULL)
   {
      return 0;
   }

   /* The first vector owns the contiguous numerical backing storage. */
   for (i = count; i > 0; )
   {
      --i;
      if (batch[i] != NULL)
      {
         status = hypre_ParKrylovDestroyVector(batch[i]);
         if (error == 0 && status != 0)
         {
            error = status;
         }
      }
   }
   status = hypre_ParKrylovFree(batch);
   if (error == 0 && status != 0)
   {
      error = status;
   }
   *address = NULL;
   return error;
}

static HYPRE_Int
nsl_release(nsl_Data *d)
{
   HYPRE_Int error = 0;
   HYPRE_Int status;
   HYPRE_Int i;

   if (d->amg != NULL)
   {
      status = HYPRE_BoomerAMGDestroy(d->amg);
      if (error == 0 && status != 0)
      {
         error = status;
      }
      d->amg = NULL;
   }

   status = nsl_free_batch(&d->z, d->capacity);
   if (error == 0 && status != 0)
   {
      error = status;
   }
   status = nsl_free_batch(&d->v, d->capacity + 1);
   if (error == 0 && status != 0)
   {
      error = status;
   }

   for (i = 0; i < NS_WORK_COUNT; ++i)
   {
      if (d->work[i] != NULL)
      {
         status = hypre_ParKrylovDestroyVector(d->work[i]);
         if (error == 0 && status != 0)
         {
            error = status;
         }
         d->work[i] = NULL;
      }
   }

   d->capacity = 0;
   d->matrix = NULL;
   d->strict = 0;
   d->base.is_setup = 0;
   return error;
}

static HYPRE_Int
nsl_residual(HYPRE_Matrix A, HYPRE_Vector b, void *x,
             void *r, HYPRE_Real *norm)
{
   NS_CALL(hypre_ParKrylovCopyVector(b, r));
   NS_CALL(hypre_ParKrylovMatvec(NULL, -1.0, A, x, 1.0, r));
   *norm = nsl_length(r);
   return 0;
}

static HYPRE_Int
nsl_precondition(nsl_Data *d, HYPRE_Matrix A, void *r, void *z)
{
   NS_CALL(hypre_ParKrylovClearVector(z));
   return HYPRE_BoomerAMGSolve(d->amg, (HYPRE_ParCSRMatrix) A,
                              (HYPRE_ParVector) r,
                              (HYPRE_ParVector) z);
}

/*
 * Short-recurrence predictor with a local residual-minimizing stabilizer.
 *
 * The external solution is a verified checkpoint. All speculative updates
 * remain in NS_TRIAL. Each preconditioned matrix application consumes one
 * iteration, including an unfinished final predictor step.
 *
 * Unfavorable biorthogonality, stagnation, or a numerical breakdown hands
 * the remaining budget to the explicitly orthogonalized correction space.
 */
static HYPRE_Int
nsl_predict(nsl_Data *d, HYPRE_Matrix A, HYPRE_Vector b,
            HYPRE_Vector x, HYPRE_Real tolerance,
            HYPRE_Real *residual, HYPRE_Int *used)
{
   void *r = d->work[NS_R];
   void *shadow = d->work[NS_SHADOW];
   void *p = d->work[NS_P];
   void *image = d->work[NS_IMAGE];
   void *z = d->work[NS_Z];
   void *t = d->work[NS_T];
   void *trial = d->work[NS_TRIAL];
   void *actual = d->work[NS_ACTUAL];

   HYPRE_Int limit = d->maxiter < 48 ? d->maxiter : 48;
   HYPRE_Real working_tolerance = 0.8 * tolerance;
   HYPRE_Real checkpoint;
   HYPRE_Real rnorm;
   HYPRE_Real shadow_norm;
   HYPRE_Real rho_previous;
   HYPRE_Real alpha;
   HYPRE_Real omega;
   HYPRE_Int first;
   HYPRE_Int segment;
   HYPRE_Int force_fallback;

restart:
   if (*used >= limit)
   {
      return HYPRE_ERROR_CONV;
   }

   checkpoint = *residual;
   rnorm = checkpoint;
   shadow_norm = checkpoint;
   rho_previous = 1.0;
   alpha = 1.0;
   omega = 1.0;
   first = 1;
   segment = 0;
   force_fallback = 0;

   NS_CALL(hypre_ParKrylovCopyVector(r, shadow));
   NS_CALL(hypre_ParKrylovCopyVector(x, trial));

   while (*used < limit)
   {
      HYPRE_Real rho;
      HYPRE_Real denominator;
      HYPRE_Real snorm;
      HYPRE_Real tt;
      HYPRE_Real ts;
      HYPRE_Real tnorm;
      HYPRE_Real angle;

      rho = hypre_ParKrylovInnerProd(shadow, r);
      if (!isfinite(rho) || rho == 0.0 ||
          fabs(rho) / shadow_norm <= 64.0 * DBL_EPSILON * rnorm)
      {
         force_fallback = 1;
         goto checkpoint_trial;
      }

      if (first)
      {
         NS_CALL(hypre_ParKrylovCopyVector(r, p));
         first = 0;
      }
      else
      {
         HYPRE_Real beta = (rho / rho_previous) * (alpha / omega);
         if (!isfinite(beta) || fabs(beta) > 1.0e12)
         {
            force_fallback = 1;
            goto checkpoint_trial;
         }

         NS_CALL(hypre_ParKrylovAxpy(-omega, image, p));
         NS_CALL(hypre_ParKrylovScaleVector(beta, p));
         NS_CALL(hypre_ParKrylovAxpy(1.0, r, p));
      }

      NS_CALL(nsl_precondition(d, A, p, z));
      NS_CALL(hypre_ParKrylovMatvec(NULL, 1.0, A, z, 0.0, image));
      ++*used;

      denominator = hypre_ParKrylovInnerProd(shadow, image);
      if (!isfinite(denominator) || denominator == 0.0)
      {
         force_fallback = 1;
         goto checkpoint_trial;
      }

      alpha = rho / denominator;
      if (!isfinite(alpha))
      {
         force_fallback = 1;
         goto checkpoint_trial;
      }

      NS_CALL(hypre_ParKrylovAxpy(alpha, z, trial));
      NS_CALL(hypre_ParKrylovAxpy(-alpha, image, r));
      snorm = nsl_length(r);

      if (!isfinite(snorm))
      {
         force_fallback = 1;
         goto checkpoint_trial;
      }
      if (snorm <= working_tolerance || *used >= limit)
      {
         goto checkpoint_trial;
      }

      NS_CALL(nsl_precondition(d, A, r, z));
      NS_CALL(hypre_ParKrylovMatvec(NULL, 1.0, A, z, 0.0, t));
      ++*used;

      tt = hypre_ParKrylovInnerProd(t, t);
      ts = hypre_ParKrylovInnerProd(t, r);
      if (!isfinite(tt) || !isfinite(ts) || tt <= 0.0 || ts == 0.0)
      {
         force_fallback = 1;
         goto checkpoint_trial;
      }

      tnorm = sqrt(tt);
      omega = ts / tt;
      angle = (fabs(ts) / tnorm) / snorm;

      if (!isfinite(omega) || !isfinite(angle) || omega == 0.0)
      {
         force_fallback = 1;
         goto checkpoint_trial;
      }

      /*
       * Prevent an almost orthogonal stabilizing image from making the
       * next biorthogonal recurrence divide by a vanishing omega.
       */
      if (angle < 0.05)
      {
         omega = copysign(0.05 * (snorm / tnorm), ts);
         if (!isfinite(omega) || omega == 0.0)
         {
            force_fallback = 1;
            goto checkpoint_trial;
         }
      }

      NS_CALL(hypre_ParKrylovAxpy(omega, z, trial));
      NS_CALL(hypre_ParKrylovAxpy(-omega, t, r));
      rnorm = nsl_length(r);
      rho_previous = rho;
      ++segment;

      if (!isfinite(rnorm))
      {
         force_fallback = 1;
         goto checkpoint_trial;
      }

      if (rnorm <= working_tolerance || *used >= limit ||
          segment >= 6 || rnorm / checkpoint > 8.0)
      {
         goto checkpoint_trial;
      }
   }

checkpoint_trial:
   {
      HYPRE_Real candidate = NAN;
      HYPRE_Real trial_norm = nsl_length(trial);

      if (isfinite(trial_norm))
      {
         NS_CALL(nsl_residual(A, b, trial, actual, &candidate));
      }

      if (isfinite(candidate) && candidate <= tolerance)
      {
         NS_CALL(hypre_ParKrylovCopyVector(trial, x));
         *residual = candidate;
         return 0;
      }

      if (isfinite(candidate) && candidate < *residual)
      {
         NS_CALL(hypre_ParKrylovCopyVector(trial, x));
         NS_CALL(hypre_ParKrylovCopyVector(actual, r));
         *residual = candidate;
      }
      else
      {
         NS_CALL(nsl_residual(A, b, x, r, residual));
         if (!isfinite(*residual))
         {
            return HYPRE_ERROR_GENERIC;
         }
         force_fallback = 1;
      }
   }

   if (*residual <= tolerance)
   {
      return 0;
   }
   if (force_fallback || *used >= limit ||
       *residual >= 0.7 * checkpoint)
   {
      return HYPRE_ERROR_CONV;
   }

   /* Reliable residual replacement also refreshes the shadow residual. */
   goto restart;
}

static HYPRE_Int
nsl_make_basis(nsl_Data *d, void *prototype, HYPRE_Int remaining)
{
   HYPRE_Int i;
   HYPRE_Int error = HYPRE_ERROR_GENERIC;
   HYPRE_Int status;

   if (d->v != NULL && d->z != NULL)
   {
      return 0;
   }
   if (remaining <= 0)
   {
      return HYPRE_ERROR_CONV;
   }

   d->capacity = remaining < NS_RESTART ? remaining : NS_RESTART;
   d->v = (void **) hypre_ParKrylovCreateVectorArray(
      d->capacity + 1, prototype);
   if (d->v == NULL)
   {
      goto failure;
   }
   for (i = 0; i <= d->capacity; ++i)
   {
      if (d->v[i] == NULL)
      {
         goto failure;
      }
   }

   d->z = (void **) hypre_ParKrylovCreateVectorArray(
      d->capacity, prototype);
   if (d->z == NULL)
   {
      goto failure;
   }
   for (i = 0; i < d->capacity; ++i)
   {
      if (d->z[i] == NULL)
      {
         goto failure;
      }
   }
   return 0;

failure:
   status = nsl_free_batch(&d->z, d->capacity);
   if (status != 0)
   {
      error = status;
   }
   status = nsl_free_batch(&d->v, d->capacity + 1);
   if (status != 0)
   {
      error = status;
   }
   d->capacity = 0;
   return error;
}

/*
 * Batched classical projection with a directly measured remainder and
 * selective reorthogonalization. Both mass operands are native batches.
 */
static HYPRE_Int
nsl_orthogonalize(nsl_Data *d, HYPRE_Int column,
                  HYPRE_Real *next_norm, HYPRE_Real *image_norm)
{
   HYPRE_Int n = column + 1;
   HYPRE_Int i;
   HYPRE_Int pass;
   HYPRE_Real *h = d->h + (size_t) column * NS_LD;
   void *w = d->v[n];

   NS_CALL(hypre_ParKrylovMassInnerProd(
      w, d->v, n + 1, 8, d->dots));

   if (!isfinite(d->dots[n]) || d->dots[n] < 0.0)
   {
      return HYPRE_ERROR_GENERIC;
   }
   *image_norm = sqrt(d->dots[n]);

   for (pass = 0; pass < 2; ++pass)
   {
      for (i = 0; i < n; ++i)
      {
         if (!isfinite(d->dots[i]))
         {
            return HYPRE_ERROR_GENERIC;
         }

         if (pass == 0)
         {
            h[i] = d->dots[i];
         }
         else
         {
            h[i] += d->dots[i];
         }

         if (!isfinite(h[i]))
         {
            return HYPRE_ERROR_GENERIC;
         }
         d->weights[i] = -d->dots[i];
      }

      NS_CALL(hypre_ParKrylovMassAxpy(
         d->weights, d->v, w, n, 8));
      *next_norm = nsl_length(w);
      if (!isfinite(*next_norm))
      {
         return HYPRE_ERROR_GENERIC;
      }

      if (pass == 1 || *next_norm == 0.0 ||
          (!d->strict && *next_norm >= 0.5 * *image_norm))
      {
         break;
      }

      NS_CALL(hypre_ParKrylovMassInnerProd(
         w, d->v, n, 8, d->dots));
   }

   return 0;
}

static HYPRE_Int
nsl_form_trial(nsl_Data *d, HYPRE_Int dimension, HYPRE_Vector x)
{
   HYPRE_Int i;
   HYPRE_Int j;
   HYPRE_Real norm;

   if (dimension <= 0 || dimension > d->capacity)
   {
      return HYPRE_ERROR_GENERIC;
   }

   for (i = dimension; i > 0; )
   {
      long double value;
      HYPRE_Real diagonal;

      --i;
      value = (long double) d->g[i];
      for (j = i + 1; j < dimension; ++j)
      {
         value -= (long double) d->h[(size_t) j * NS_LD + i] *
                  (long double) d->y[j];
      }

      diagonal = d->h[(size_t) i * NS_LD + i];
      if (!isfinite(diagonal) || diagonal == 0.0)
      {
         return HYPRE_ERROR_CONV;
      }

      d->y[i] = (HYPRE_Real) (value / (long double) diagonal);
      if (!isfinite(d->y[i]))
      {
         return HYPRE_ERROR_GENERIC;
      }
      d->weights[i] = d->y[i];
   }

   NS_CALL(hypre_ParKrylovCopyVector(x, d->work[NS_TRIAL]));
   NS_CALL(hypre_ParKrylovMassAxpy(
      d->weights, d->z, d->work[NS_TRIAL], dimension, 8));

   norm = nsl_length(d->work[NS_TRIAL]);
   return isfinite(norm) ? 0 : HYPRE_ERROR_GENERIC;
}

static HYPRE_Int
nsl_correct(nsl_Data *d, HYPRE_Matrix A, HYPRE_Vector b,
            HYPRE_Vector x, HYPRE_Real tolerance,
            HYPRE_Real residual, HYPRE_Int *used)
{
   HYPRE_Int stagnation = 0;
   HYPRE_Real working_tolerance = 0.8 * tolerance;

   if (*used >= d->maxiter)
   {
      return HYPRE_ERROR_CONV;
   }

   NS_CALL(nsl_make_basis(d, b, d->maxiter - *used));
   d->strict = 0;

   while (*used < d->maxiter)
   {
      HYPRE_Int column;
      HYPRE_Int dimension = 0;
      HYPRE_Real old_residual = residual;
      HYPRE_Real inverse = 1.0 / residual;
      HYPRE_Real candidate;
      HYPRE_Real predicted;

      if (!isfinite(inverse))
      {
         return HYPRE_ERROR_GENERIC;
      }

      NS_CALL(hypre_ParKrylovCopyVector(d->work[NS_R], d->v[0]));
      NS_CALL(hypre_ParKrylovScaleVector(inverse, d->v[0]));
      d->g[0] = residual;

      for (column = 0;
           column < d->capacity && *used < d->maxiter;
           ++column)
      {
         HYPRE_Real *h = d->h + (size_t) column * NS_LD;
         HYPRE_Real next_norm;
         HYPRE_Real image_norm;
         HYPRE_Real rotation;
         HYPRE_Real diagonal;
         HYPRE_Int near_breakdown;
         HYPRE_Int row;

         NS_CALL(nsl_precondition(d, A, d->v[column], d->z[column]));
         NS_CALL(hypre_ParKrylovMatvec(
            NULL, 1.0, A, d->z[column], 0.0, d->v[column + 1]));
         ++*used;

         NS_CALL(nsl_orthogonalize(d, column, &next_norm, &image_norm));
         h[column + 1] = next_norm;

         near_breakdown =
            next_norm <= HYPRE_REAL_MIN ||
            next_norm <= 32.0 * DBL_EPSILON * image_norm;

         if (!near_breakdown)
         {
            inverse = 1.0 / next_norm;
            if (!isfinite(inverse))
            {
               return HYPRE_ERROR_GENERIC;
            }
            NS_CALL(hypre_ParKrylovScaleVector(
               inverse, d->v[column + 1]));
         }

         for (row = 0; row < column; ++row)
         {
            HYPRE_Real upper = h[row];
            HYPRE_Real lower = h[row + 1];

            h[row] = d->cs[row] * upper + d->sn[row] * lower;
            h[row + 1] = -d->sn[row] * upper + d->cs[row] * lower;
            if (!isfinite(h[row]) || !isfinite(h[row + 1]))
            {
               return HYPRE_ERROR_GENERIC;
            }
         }

         diagonal = h[column];
         rotation = hypot(diagonal, h[column + 1]);
         if (!isfinite(rotation))
         {
            return HYPRE_ERROR_GENERIC;
         }
         if (rotation == 0.0)
         {
            break;
         }

         d->cs[column] = diagonal / rotation;
         d->sn[column] = h[column + 1] / rotation;
         h[column] = rotation;
         h[column + 1] = 0.0;

         d->g[column + 1] = -d->sn[column] * d->g[column];
         d->g[column] *= d->cs[column];

         if (!isfinite(d->g[column]) ||
             !isfinite(d->g[column + 1]))
         {
            return HYPRE_ERROR_GENERIC;
         }

         dimension = column + 1;
         if (fabs(d->g[dimension]) <= working_tolerance ||
             near_breakdown)
         {
            break;
         }
      }

      if (dimension == 0)
      {
         return HYPRE_ERROR_CONV;
      }

      predicted = fabs(d->g[dimension]);
      NS_CALL(nsl_form_trial(d, dimension, x));
      NS_CALL(nsl_residual(A, b, d->work[NS_TRIAL],
                           d->work[NS_R], &candidate));
      if (!isfinite(candidate))
      {
         return HYPRE_ERROR_GENERIC;
      }

      if (candidate <= tolerance)
      {
         NS_CALL(hypre_ParKrylovCopyVector(d->work[NS_TRIAL], x));
         return 0;
      }

      if (predicted < 0.125 * candidate)
      {
         d->strict = 1;
      }

      if (candidate < old_residual)
      {
         NS_CALL(hypre_ParKrylovCopyVector(d->work[NS_TRIAL], x));
         residual = candidate;
      }
      else
      {
         d->strict = 1;
         NS_CALL(nsl_residual(A, b, x, d->work[NS_R], &residual));
         if (!isfinite(residual))
         {
            return HYPRE_ERROR_GENERIC;
         }
      }

      if (residual <= tolerance)
      {
         return 0;
      }

      if (residual >= old_residual * (1.0 - 64.0 * DBL_EPSILON))
      {
         ++stagnation;
      }
      else
      {
         stagnation = 0;
      }

      if (stagnation >= 3)
      {
         return HYPRE_ERROR_CONV;
      }
   }

   return HYPRE_ERROR_CONV;
}

HYPRE_Int
solver_create(HYPRE_Solver *solver,
              HYPRE_Real relative_tolerance,
              HYPRE_Real absolute_tolerance)
{
   /* Preserve the discovery-time solver budget; see README.md. */
   const HYPRE_Int maximum_iterations = 16000;
   nsl_Data *d;

   if (solver == NULL)
   {
      return HYPRE_ERROR_GENERIC;
   }
   *solver = NULL;

   if (!isfinite(relative_tolerance) ||
       !isfinite(absolute_tolerance) ||
       relative_tolerance < 0.0 ||
       absolute_tolerance < 0.0 ||
       maximum_iterations < 0)
   {
      return HYPRE_ERROR_GENERIC;
   }

   d = (nsl_Data *) hypre_ParKrylovCAlloc(
      1, sizeof(*d), HYPRE_MEMORY_HOST);
   if (d == NULL)
   {
      return HYPRE_ERROR_GENERIC;
   }

   d->base.setup = nsl_setup;
   d->base.solve = nsl_solve;
   d->base.destroy = nsl_destroy;
   d->base.is_setup = 0;
   d->rtol = relative_tolerance;
   d->atol = absolute_tolerance;
   d->maxiter = maximum_iterations;

   *solver = (HYPRE_Solver) d;
   return 0;
}

static HYPRE_Int
nsl_setup(HYPRE_Solver solver, HYPRE_Matrix A,
           HYPRE_Vector b, HYPRE_Vector x)
{
   nsl_Data *d = (nsl_Data *) solver;
   HYPRE_BigInt rows;
   HYPRE_BigInt columns;
   HYPRE_Int count;
   HYPRE_Int i;
   HYPRE_Int error;
   HYPRE_Int status;

   if (d == NULL || A == NULL || b == NULL || x == NULL || b == x)
   {
      return HYPRE_ERROR_GENERIC;
   }

   error = nsl_release(d);
   if (error != 0)
   {
      return error;
   }

#define NS_SETUP(call)                                    \
   do                                                    \
   {                                                     \
      error = (call);                                    \
      if (error != 0)                                    \
      {                                                  \
         goto failure;                                   \
      }                                                  \
   } while (0)

   NS_SETUP(HYPRE_ParCSRMatrixGetDims(
      (HYPRE_ParCSRMatrix) A, &rows, &columns));
   if (rows != columns)
   {
      error = HYPRE_ERROR_GENERIC;
      goto failure;
   }

   /*
    * The normal path allocates only eight vectors. The long correction
    * basis is allocated only if the short recurrence proves unsuitable.
    */
   count = d->maxiter > 0 ? NS_WORK_COUNT : 1;
   for (i = 0; i < count; ++i)
   {
      d->work[i] = hypre_ParKrylovCreateVector(
         i == NS_TRIAL ? (void *) x : (void *) b);
      if (d->work[i] == NULL)
      {
         error = HYPRE_ERROR_GENERIC;
         goto failure;
      }
   }

   if (d->maxiter == 0)
   {
      d->matrix = A;
      d->base.is_setup = 1;
      return 0;
   }

   NS_SETUP(HYPRE_BoomerAMGCreate(&d->amg));
   if (d->amg == NULL)
   {
      error = HYPRE_ERROR_GENERIC;
      goto failure;
   }

   NS_SETUP(HYPRE_BoomerAMGSetCoarsenType(d->amg, 10));
   NS_SETUP(HYPRE_BoomerAMGSetInterpType(d->amg, 6));
   NS_SETUP(HYPRE_BoomerAMGSetPMaxElmts(d->amg, 4));
   NS_SETUP(HYPRE_BoomerAMGSetStrongThreshold(d->amg, 0.25));

   if (rows >= 32768)
   {
      NS_SETUP(HYPRE_BoomerAMGSetAggNumLevels(d->amg, 1));
      NS_SETUP(HYPRE_BoomerAMGSetAggInterpType(d->amg, 4));
      NS_SETUP(HYPRE_BoomerAMGSetAggPMaxElmts(d->amg, 8));
   }

   NS_SETUP(HYPRE_BoomerAMGSetMaxLevels(d->amg, 25));
   NS_SETUP(HYPRE_BoomerAMGSetMaxCoarseSize(d->amg, 32));
   NS_SETUP(HYPRE_BoomerAMGSetCycleType(d->amg, 1));
   NS_SETUP(HYPRE_BoomerAMGSetNumSweeps(d->amg, 1));
   NS_SETUP(HYPRE_BoomerAMGSetRelaxOrder(d->amg, 0));
   NS_SETUP(HYPRE_BoomerAMGSetCycleRelaxType(d->amg, 3, 1));
   NS_SETUP(HYPRE_BoomerAMGSetCycleRelaxType(d->amg, 4, 2));
   NS_SETUP(HYPRE_BoomerAMGSetCycleRelaxType(d->amg, 9, 3));
   NS_SETUP(HYPRE_BoomerAMGSetMinIter(d->amg, 1));
   NS_SETUP(HYPRE_BoomerAMGSetMaxIter(d->amg, 1));
   NS_SETUP(HYPRE_BoomerAMGSetTol(d->amg, 0.0));

   NS_SETUP(hypre_ParKrylovCopyVector(b, d->work[NS_R]));
   NS_SETUP(hypre_ParKrylovClearVector(d->work[NS_Z]));
   NS_SETUP(HYPRE_BoomerAMGSetup(
      d->amg, (HYPRE_ParCSRMatrix) A,
      (HYPRE_ParVector) d->work[NS_R],
      (HYPRE_ParVector) d->work[NS_Z]));

#undef NS_SETUP

   d->matrix = A;
   d->base.is_setup = 1;
   return 0;

failure:
   status = nsl_release(d);
   return error != 0 ? error : status;
}

static HYPRE_Int
nsl_solve(HYPRE_Solver solver, HYPRE_Matrix A,
           HYPRE_Vector b, HYPRE_Vector x)
{
   nsl_Data *d = (nsl_Data *) solver;
   HYPRE_Real bnorm;
   HYPRE_Real xnorm;
   HYPRE_Real residual;
   HYPRE_Real reference;
   HYPRE_Real tolerance;
   HYPRE_Int used = 0;
   HYPRE_Int status;

   if (d == NULL || A == NULL || b == NULL || x == NULL || b == x ||
       !d->base.is_setup || d->matrix != A || d->work[NS_R] == NULL)
   {
      return HYPRE_ERROR_GENERIC;
   }

   bnorm = nsl_length(b);
   xnorm = nsl_length(x);
   if (!isfinite(bnorm) || !isfinite(xnorm))
   {
      return HYPRE_ERROR_GENERIC;
   }

   NS_CALL(nsl_residual(A, b, x, d->work[NS_R], &residual));
   if (!isfinite(residual))
   {
      return HYPRE_ERROR_GENERIC;
   }
   if (residual == 0.0)
   {
      return 0;
   }

   reference = bnorm > 0.0 ? bnorm : residual;
   if (d->rtol > 1.0 && reference > DBL_MAX / d->rtol)
   {
      tolerance = DBL_MAX;
   }
   else
   {
      tolerance = d->rtol * reference;
   }
   if (d->atol > tolerance)
   {
      tolerance = d->atol;
   }
   if (!isfinite(tolerance))
   {
      return HYPRE_ERROR_GENERIC;
   }

   /*
    * Retain an exact supplied homogeneous solution. Otherwise verify the
    * known zero solution before committing it.
    */
   if (bnorm == 0.0 && d->maxiter > 0)
   {
      HYPRE_Real candidate;

      NS_CALL(hypre_ParKrylovClearVector(d->work[NS_TRIAL]));
      NS_CALL(nsl_residual(A, b, d->work[NS_TRIAL],
                           d->work[NS_ACTUAL], &candidate));
      if (!isfinite(candidate) || candidate != 0.0)
      {
         return HYPRE_ERROR_CONV;
      }

      NS_CALL(hypre_ParKrylovCopyVector(d->work[NS_TRIAL], x));
      return 0;
   }

   if (residual <= tolerance)
   {
      return 0;
   }
   if (d->maxiter == 0)
   {
      return HYPRE_ERROR_CONV;
   }
   if (d->amg == NULL || d->work[NS_ACTUAL] == NULL)
   {
      return HYPRE_ERROR_GENERIC;
   }

   status = nsl_predict(d, A, b, x, tolerance, &residual, &used);
   if (status == 0)
   {
      return 0;
   }
   if (status != HYPRE_ERROR_CONV)
   {
      return status;
   }

   return nsl_correct(d, A, b, x, tolerance, residual, &used);
}

static HYPRE_Int
nsl_destroy(HYPRE_Solver solver)
{
   nsl_Data *d = (nsl_Data *) solver;
   HYPRE_Int error;
   HYPRE_Int status;

   if (d == NULL)
   {
      return 0;
   }

   error = nsl_release(d);
   status = hypre_ParKrylovFree(d);
   return error != 0 ? error : status;
}

#undef NS_CALL
