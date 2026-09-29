#include "HYPRE_parcsr_ls.h"

enum { NSL_RESTART = 24, NSL_WORK = 5 };

typedef struct
{
   hypre_Solver base;
   HYPRE_Solver amg;
   HYPRE_Matrix matrix;
   HYPRE_Real relative_tolerance;
   HYPRE_Real absolute_tolerance;
   HYPRE_Int maximum_iterations;

   /* CG workspace and lazily allocated flexible-GMRES bases. */
   void **work;
   void **v;
   void **z;

   HYPRE_Real h[(NSL_RESTART + 1) * NSL_RESTART];
   HYPRE_Real cs[NSL_RESTART], sn[NSL_RESTART];
   HYPRE_Real g[NSL_RESTART + 1], y[NSL_RESTART];
   HYPRE_Real dots[NSL_RESTART + 1];
   HYPRE_Complex coefficients[NSL_RESTART];
} nsl_Data;

static HYPRE_Int nsl_setup(HYPRE_Solver, HYPRE_Matrix,
                           HYPRE_Vector, HYPRE_Vector);
static HYPRE_Int nsl_solve(HYPRE_Solver, HYPRE_Matrix,
                           HYPRE_Vector, HYPRE_Vector);
static HYPRE_Int nsl_destroy(HYPRE_Solver);

static HYPRE_Int
nsl_finite(HYPRE_Real value)
{
   return isfinite((double) value) ? 1 : 0;
}

static HYPRE_Real *
nsl_entry(nsl_Data *d, HYPRE_Int row, HYPRE_Int column)
{
   return &d->h[(size_t) row * NSL_RESTART + (size_t) column];
}

static HYPRE_Int
nsl_norm(void *vector, HYPRE_Real *result)
{
   HYPRE_Real squared = hypre_ParKrylovInnerProd(vector, vector);
   if (!nsl_finite(squared) || squared < 0.0)
   {
      return HYPRE_ERROR_GENERIC;
   }
   *result = sqrt(squared);
   return nsl_finite(*result) ? 0 : HYPRE_ERROR_GENERIC;
}

static HYPRE_Int
nsl_release_batch(void ***batch, HYPRE_Int count)
{
   HYPRE_Int i, status, error = 0;

   if (*batch == NULL)
   {
      return 0;
   }

   /* Vector zero owns the shared numerical storage. */
   for (i = count - 1; i >= 0; --i)
   {
      if ((*batch)[i] != NULL)
      {
         status = hypre_ParKrylovDestroyVector((*batch)[i]);
         if (error == 0)
         {
            error = status;
         }
      }
   }
   status = hypre_ParKrylovFree(*batch);
   if (error == 0)
   {
      error = status;
   }
   *batch = NULL;
   return error;
}

static HYPRE_Int
nsl_release_setup(nsl_Data *d)
{
   HYPRE_Int status, error = 0;

   d->base.is_setup = 0;
   d->matrix = NULL;

   if (d->amg != NULL)
   {
      error = HYPRE_BoomerAMGDestroy(d->amg);
      d->amg = NULL;
   }

   status = nsl_release_batch(&d->z, NSL_RESTART);
   if (error == 0) error = status;
   status = nsl_release_batch(&d->v, NSL_RESTART + 1);
   if (error == 0) error = status;
   status = nsl_release_batch(&d->work, NSL_WORK);
   if (error == 0) error = status;
   return error;
}

static HYPRE_Int
nsl_residual(HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x,
             void *r, HYPRE_Real *norm)
{
   HYPRE_Int status = hypre_ParKrylovCopyVector(b, r);
   if (status != 0) return status;

   status = hypre_ParKrylovMatvec(NULL, -1.0, A, x, 1.0, r);
   if (status != 0) return status;

   return nsl_norm(r, norm);
}

static HYPRE_Int
nsl_precondition(nsl_Data *d, HYPRE_Matrix A,
                 void *source, void *destination)
{
   HYPRE_Int status = hypre_ParKrylovClearVector(destination);
   if (status != 0) return status;

   return HYPRE_BoomerAMGSolve(
      d->amg, (HYPRE_ParCSRMatrix) A,
      (HYPRE_ParVector) source, (HYPRE_ParVector) destination);
}

static HYPRE_Int
nsl_prepare_gmres(nsl_Data *d, HYPRE_Vector template_vector)
{
   if (d->v == NULL)
   {
      d->v = (void **) hypre_ParKrylovCreateVectorArray(
         NSL_RESTART + 1, template_vector);
      if (d->v == NULL) return HYPRE_ERROR_GENERIC;
   }
   if (d->z == NULL)
   {
      d->z = (void **) hypre_ParKrylovCreateVectorArray(
         NSL_RESTART, template_vector);
      if (d->z == NULL) return HYPRE_ERROR_GENERIC;
   }
   return 0;
}

static HYPRE_Int
nsl_gmres_update(nsl_Data *d, HYPRE_Int dimension, HYPRE_Vector x)
{
   HYPRE_Int i, j;

   for (i = dimension - 1; i >= 0; --i)
   {
      HYPRE_Real value = d->g[i];
      HYPRE_Real diagonal = *nsl_entry(d, i, i);

      for (j = i + 1; j < dimension; ++j)
      {
         value -= *nsl_entry(d, i, j) * d->y[j];
      }
      if (!nsl_finite(value) || !nsl_finite(diagonal) ||
          diagonal == 0.0)
      {
         return HYPRE_ERROR_GENERIC;
      }
      d->y[i] = value / diagonal;
      if (!nsl_finite(d->y[i])) return HYPRE_ERROR_GENERIC;
   }

   for (i = 0; i < dimension; ++i)
   {
      d->coefficients[i] = d->y[i];
   }
   return hypre_ParKrylovMassAxpy(
      d->coefficients, d->z, x, dimension, 0);
}

static HYPRE_Int
nsl_gmres(nsl_Data *d, HYPRE_Matrix A, HYPRE_Vector b,
          HYPRE_Vector x, HYPRE_Real tolerance,
          HYPRE_Int *iterations)
{
   HYPRE_Real residual;
   HYPRE_Int status;

   status = nsl_residual(A, b, x, d->work[0], &residual);
   if (status != 0) return status;
   if (residual <= tolerance) return 0;
   if (*iterations >= d->maximum_iterations) return HYPRE_ERROR_CONV;

   status = nsl_prepare_gmres(d, b);
   if (status != 0) return status;

   status = hypre_ParKrylovCopyVector(d->work[0], d->v[0]);
   if (status != 0) return status;

   while (*iterations < d->maximum_iterations)
   {
      HYPRE_Int j, dimension = 0;
      HYPRE_Real inverse = 1.0 / residual;

      if (!nsl_finite(inverse)) return HYPRE_ERROR_GENERIC;
      status = hypre_ParKrylovScaleVector(inverse, d->v[0]);
      if (status != 0) return status;
      d->g[0] = residual;

      for (j = 0; j < NSL_RESTART &&
                  *iterations < d->maximum_iterations; ++j)
      {
         HYPRE_Real before, after, diagonal, subdiagonal, length;
         HYPRE_Int i;

         status = nsl_precondition(d, A, d->v[j], d->z[j]);
         if (status != 0) return status;

         status = hypre_ParKrylovMatvec(
            NULL, 1.0, A, d->z[j], 0.0, d->v[j + 1]);
         if (status != 0) return status;

         status = hypre_ParKrylovMassInnerProd(
            d->v[j + 1], d->v, j + 2, 0, d->dots);
         if (status != 0) return status;
         if (!nsl_finite(d->dots[j + 1]) ||
             d->dots[j + 1] < 0.0)
         {
            return HYPRE_ERROR_GENERIC;
         }
         before = sqrt(d->dots[j + 1]);

         for (i = 0; i <= j; ++i)
         {
            if (!nsl_finite(d->dots[i]))
            {
               return HYPRE_ERROR_GENERIC;
            }
            *nsl_entry(d, i, j) = d->dots[i];
            d->coefficients[i] = -d->dots[i];
         }
         status = hypre_ParKrylovMassAxpy(
            d->coefficients, d->v, d->v[j + 1], j + 1, 0);
         if (status != 0) return status;

         status = nsl_norm(d->v[j + 1], &after);
         if (status != 0) return status;

         if (after > 0.0 && after < 0.7 * before)
         {
            status = hypre_ParKrylovMassInnerProd(
               d->v[j + 1], d->v, j + 1, 0, d->dots);
            if (status != 0) return status;

            for (i = 0; i <= j; ++i)
            {
               if (!nsl_finite(d->dots[i]))
               {
                  return HYPRE_ERROR_GENERIC;
               }
               *nsl_entry(d, i, j) += d->dots[i];
               d->coefficients[i] = -d->dots[i];
            }
            status = hypre_ParKrylovMassAxpy(
               d->coefficients, d->v, d->v[j + 1], j + 1, 0);
            if (status != 0) return status;

            status = nsl_norm(d->v[j + 1], &after);
            if (status != 0) return status;
         }

         if (before > 0.0 && after <= 1.0e-14 * before)
         {
            after = 0.0;
         }
         *nsl_entry(d, j + 1, j) = after;

         if (after > 0.0)
         {
            inverse = 1.0 / after;
            if (!nsl_finite(inverse)) return HYPRE_ERROR_GENERIC;
            status = hypre_ParKrylovScaleVector(
               inverse, d->v[j + 1]);
            if (status != 0) return status;
         }

         for (i = 0; i < j; ++i)
         {
            HYPRE_Real upper = *nsl_entry(d, i, j);
            HYPRE_Real lower = *nsl_entry(d, i + 1, j);
            *nsl_entry(d, i, j) =
               d->cs[i] * upper + d->sn[i] * lower;
            *nsl_entry(d, i + 1, j) =
               -d->sn[i] * upper + d->cs[i] * lower;
         }

         diagonal = *nsl_entry(d, j, j);
         subdiagonal = *nsl_entry(d, j + 1, j);
         length = hypot(diagonal, subdiagonal);
         if (!nsl_finite(length) || length == 0.0)
         {
            return HYPRE_ERROR_GENERIC;
         }

         d->cs[j] = diagonal / length;
         d->sn[j] = subdiagonal / length;
         *nsl_entry(d, j, j) = length;
         *nsl_entry(d, j + 1, j) = 0.0;
         d->g[j + 1] = -d->sn[j] * d->g[j];
         d->g[j] *= d->cs[j];
         if (!nsl_finite(d->g[j]) || !nsl_finite(d->g[j + 1]))
         {
            return HYPRE_ERROR_GENERIC;
         }

         ++*iterations;
         dimension = j + 1;
         if (fabs(d->g[j + 1]) <= tolerance || after == 0.0)
         {
            break;
         }
      }

      if (dimension == 0) return HYPRE_ERROR_CONV;

      status = nsl_gmres_update(d, dimension, x);
      if (status != 0) return status;

      status = nsl_residual(A, b, x, d->v[0], &residual);
      if (status != 0) return status;
      if (residual <= tolerance) return 0;
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

   if (solver == NULL) return HYPRE_ERROR_GENERIC;
   *solver = NULL;

   if (!nsl_finite(relative_tolerance) ||
       !nsl_finite(absolute_tolerance) ||
       relative_tolerance < 0.0 || absolute_tolerance < 0.0 ||
       maximum_iterations < 0)
   {
      return HYPRE_ERROR_GENERIC;
   }

   d = (nsl_Data *) hypre_ParKrylovCAlloc(
      1, sizeof(*d), HYPRE_MEMORY_HOST);
   if (d == NULL) return HYPRE_ERROR_GENERIC;

   d->base.setup = nsl_setup;
   d->base.solve = nsl_solve;
   d->base.destroy = nsl_destroy;
   d->relative_tolerance = relative_tolerance;
   d->absolute_tolerance = absolute_tolerance;
   d->maximum_iterations = maximum_iterations;
   *solver = (HYPRE_Solver) d;
   return 0;
}

static HYPRE_Int
nsl_setup(HYPRE_Solver solver, HYPRE_Matrix A,
          HYPRE_Vector b, HYPRE_Vector x)
{
   nsl_Data *d = (nsl_Data *) solver;
   HYPRE_BigInt rows, columns;
   HYPRE_Int status;

   if (d == NULL || A == NULL || b == NULL || x == NULL)
   {
      return HYPRE_ERROR_GENERIC;
   }

   status = nsl_release_setup(d);
   if (status != 0) return status;

   status = HYPRE_ParCSRMatrixGetDims(
      (HYPRE_ParCSRMatrix) A, &rows, &columns);
   if (status != 0) return status;
   if (rows != columns) return HYPRE_ERROR_GENERIC;

   d->work = (void **) hypre_ParKrylovCreateVectorArray(NSL_WORK, b);
   if (d->work == NULL) return HYPRE_ERROR_GENERIC;

   status = HYPRE_BoomerAMGCreate(&d->amg);
   if (status != 0) goto failure;
   if (d->amg == NULL)
   {
      status = HYPRE_ERROR_GENERIC;
      goto failure;
   }

#define NSL_SET(call)                \
   do                               \
   {                                \
      status = (call);               \
      if (status != 0) goto failure; \
   } while (0)

   NSL_SET(HYPRE_BoomerAMGSetCoarsenType(d->amg, 10));
   NSL_SET(HYPRE_BoomerAMGSetInterpType(d->amg, 6));
   NSL_SET(HYPRE_BoomerAMGSetPMaxElmts(d->amg, 4));
   NSL_SET(HYPRE_BoomerAMGSetMaxLevels(d->amg, 25));
   NSL_SET(HYPRE_BoomerAMGSetMaxCoarseSize(d->amg, 100));
   NSL_SET(HYPRE_BoomerAMGSetStrongThreshold(d->amg, 0.4));
   if (rows >= 100000)
   {
      NSL_SET(HYPRE_BoomerAMGSetAggNumLevels(d->amg, 1));
   }

   /* Opposite sweep directions make the CG preconditioner symmetric. */
   NSL_SET(HYPRE_BoomerAMGSetRelaxOrder(d->amg, 0));
   NSL_SET(HYPRE_BoomerAMGSetNumSweeps(d->amg, 1));
   NSL_SET(HYPRE_BoomerAMGSetCycleRelaxType(d->amg, 3, 1));
   NSL_SET(HYPRE_BoomerAMGSetCycleRelaxType(d->amg, 4, 2));
   NSL_SET(HYPRE_BoomerAMGSetMaxIter(d->amg, 1));
   NSL_SET(HYPRE_BoomerAMGSetTol(d->amg, 0.0));
#undef NSL_SET

   status = hypre_ParKrylovClearVector(d->work[0]);
   if (status != 0) goto failure;
   status = hypre_ParKrylovClearVector(d->work[1]);
   if (status != 0) goto failure;

   status = HYPRE_BoomerAMGSetup(
      d->amg, (HYPRE_ParCSRMatrix) A,
      (HYPRE_ParVector) d->work[0],
      (HYPRE_ParVector) d->work[1]);
   if (status != 0) goto failure;

   d->matrix = A;
   d->base.is_setup = 1;
   return 0;

failure:
   {
      HYPRE_Int cleanup = nsl_release_setup(d);
      return status != 0 ? status : cleanup;
   }
}

static HYPRE_Int
nsl_solve(HYPRE_Solver solver, HYPRE_Matrix A,
          HYPRE_Vector b, HYPRE_Vector x)
{
   nsl_Data *d = (nsl_Data *) solver;
   HYPRE_Real bnorm, residual, initial, tolerance, previous_rho = 0.0;
   HYPRE_Int iterations = 0, status;

   if (d == NULL || A == NULL || b == NULL || x == NULL ||
       !d->base.is_setup || d->matrix != A ||
       d->amg == NULL || d->work == NULL)
   {
      return HYPRE_ERROR_GENERIC;
   }

   status = nsl_norm(b, &bnorm);
   if (status != 0) return status;

   status = nsl_residual(A, b, x, d->work[0], &residual);
   if (status != 0) return status;

   if (bnorm > 1.0 &&
       d->relative_tolerance > DBL_MAX / bnorm)
   {
      tolerance = DBL_MAX;
   }
   else
   {
      tolerance = d->relative_tolerance * bnorm;
   }
   tolerance = fmax(tolerance, d->absolute_tolerance);

   if (residual <= tolerance) return 0;
   if (d->maximum_iterations == 0) return HYPRE_ERROR_CONV;
   initial = residual;

   while (iterations < d->maximum_iterations)
   {
      HYPRE_Real rho, curvature, alpha, beta;

      status = nsl_precondition(d, A, d->work[0], d->work[1]);
      if (status != 0) return status;

      rho = hypre_ParKrylovInnerProd(d->work[0], d->work[1]);
      if (!nsl_finite(rho) || rho <= 0.0) break;

      if (previous_rho == 0.0)
      {
         status = hypre_ParKrylovCopyVector(
            d->work[1], d->work[2]);
      }
      else
      {
         beta = rho / previous_rho;
         if (!nsl_finite(beta) || beta < 0.0) break;

         status = hypre_ParKrylovScaleVector(beta, d->work[2]);
         if (status == 0)
         {
            status = hypre_ParKrylovAxpy(
               1.0, d->work[1], d->work[2]);
         }
      }
      if (status != 0) return status;

      status = hypre_ParKrylovMatvec(
         NULL, 1.0, A, d->work[2], 0.0, d->work[3]);
      if (status != 0) return status;

      curvature = hypre_ParKrylovInnerProd(
         d->work[2], d->work[3]);
      if (!nsl_finite(curvature) || curvature <= 0.0) break;

      alpha = rho / curvature;
      if (!nsl_finite(alpha) || alpha <= 0.0) break;

      status = hypre_ParKrylovAxpy(alpha, d->work[2], x);
      if (status != 0) return status;
      status = hypre_ParKrylovAxpy(
         -alpha, d->work[3], d->work[0]);
      if (status != 0) return status;

      ++iterations;
      previous_rho = rho;

      /*
       * The preconditioned dot product is needed every iteration.
       * Checking the unpreconditioned norm less often removes most
       * additional CG reductions; always verify prospective convergence
       * against the true residual.
       */
      if (iterations <= 4 || iterations % 4 == 0 ||
          iterations == d->maximum_iterations)
      {
         HYPRE_Real recursive_norm;

         status = nsl_norm(d->work[0], &recursive_norm);
         if (status != 0) return status;

         if (recursive_norm <= tolerance ||
             iterations % 24 == 0 ||
             iterations == d->maximum_iterations)
         {
            HYPRE_Real true_norm;

            status = nsl_residual(
               A, b, x, d->work[4], &true_norm);
            if (status != 0) return status;
            if (true_norm <= tolerance) return 0;

            if (recursive_norm <= tolerance ||
                true_norm > 1.1 * recursive_norm ||
                true_norm < 0.9 * recursive_norm)
            {
               status = hypre_ParKrylovCopyVector(
                  d->work[4], d->work[0]);
               if (status != 0) return status;
               previous_rho = 0.0;
            }

            if ((iterations == 48 && true_norm > 0.95 * initial) ||
                (iterations == 96 && true_norm > 0.50 * initial))
            {
               break;
            }
         }
      }
   }

   /* Flexible GMRES retains the warm-started, possibly improved iterate
      if CG encounters nonsymmetry, breakdown, or sustained stagnation. */
   return nsl_gmres(d, A, b, x, tolerance, &iterations);
}

static HYPRE_Int
nsl_destroy(HYPRE_Solver solver)
{
   nsl_Data *d = (nsl_Data *) solver;
   HYPRE_Int status, free_status;

   if (d == NULL) return 0;
   status = nsl_release_setup(d);
   free_status = hypre_ParKrylovFree(d);
   return status != 0 ? status : free_status;
}
