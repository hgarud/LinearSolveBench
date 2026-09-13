#include "HYPRE_parcsr_ls.h"

/* Flexible right-preconditioned GMRES(50) with one fixed BoomerAMG V-cycle.
   Every cycle starts its correction at zero. The Arnoldi update retains the
   actual preconditioned vectors and verifies the original residual after each
   restart. Factory creation, AMG setup, and the solve are all timed.

   Fixed AMG configuration: HMIS coarsening, extended+i interpolation (at most
   four entries per row), strength threshold 0.25, symmetric hybrid
   Gauss-Seidel smoothing, one sweep, and at most 25 levels. This source uses
   only the public candidate API and receives no source or case identities. */

enum
{
   REFERENCE_GMRES_RESTART = 50
};

typedef struct
{
   hypre_Solver base;

   HYPRE_Real relative_tolerance;
   HYPRE_Real absolute_tolerance;
   HYPRE_Int maximum_iterations;

   HYPRE_Matrix matrix;
   void **basis;
   void **preconditioned;
   HYPRE_Solver amg;
   HYPRE_Real *hessenberg;
   HYPRE_Real *cosines;
   HYPRE_Real *sines;
   HYPRE_Real *rhs;
   HYPRE_Real *coefficients;
} reference_GMRESData;

static HYPRE_Int reference_GMRESSetup(HYPRE_Solver solver, HYPRE_Matrix A,
                                HYPRE_Vector b, HYPRE_Vector x);
static HYPRE_Int reference_GMRESSolve(HYPRE_Solver solver, HYPRE_Matrix A,
                                HYPRE_Vector b, HYPRE_Vector x);
static HYPRE_Int reference_GMRESDestroy(HYPRE_Solver solver);

static HYPRE_Int
reference_is_finite(HYPRE_Real value)
{
   return isfinite((double) value) ? 1 : 0;
}

static HYPRE_Real
reference_absolute(HYPRE_Real value)
{
   return value < 0.0 ? -value : value;
}

static HYPRE_Real
reference_pair_norm(HYPRE_Real left, HYPRE_Real right)
{
   HYPRE_Real larger = reference_absolute(left);
   HYPRE_Real smaller = reference_absolute(right);
   HYPRE_Real ratio;

   if (smaller > larger)
   {
      HYPRE_Real temporary = larger;
      larger = smaller;
      smaller = temporary;
   }
   if (larger == 0.0)
   {
      return 0.0;
   }
   ratio = smaller / larger;
   return larger * sqrt(1.0 + ratio * ratio);
}

static HYPRE_Real
reference_hessenberg_get(const reference_GMRESData *data, HYPRE_Int row,
                   HYPRE_Int column)
{
   return data->hessenberg[
      (size_t) row * (size_t) REFERENCE_GMRES_RESTART + (size_t) column];
}

static void
reference_hessenberg_set(reference_GMRESData *data, HYPRE_Int row, HYPRE_Int column,
                   HYPRE_Real value)
{
   data->hessenberg[
      (size_t) row * (size_t) REFERENCE_GMRES_RESTART + (size_t) column] = value;
}

static HYPRE_Int
reference_vector_norm(void *vector, HYPRE_Real *norm)
{
   HYPRE_Real squared_norm;

   if (vector == NULL || norm == NULL)
   {
      return HYPRE_ERROR_GENERIC;
   }
   squared_norm = hypre_ParKrylovInnerProd(vector, vector);
   if (!reference_is_finite(squared_norm) || squared_norm < 0.0)
   {
      return HYPRE_ERROR_GENERIC;
   }

   *norm = sqrt(squared_norm);
   return reference_is_finite(*norm) ? 0 : HYPRE_ERROR_GENERIC;
}

static HYPRE_Int
reference_destroy_vector_array(void ***array_pointer, HYPRE_Int count)
{
   HYPRE_Int error = 0;
   HYPRE_Int i;
   void **array;

   if (array_pointer == NULL || *array_pointer == NULL)
   {
      return 0;
   }
   array = *array_pointer;

   for (i = 0; i < count; i++)
   {
      if (array[i] != NULL)
      {
         HYPRE_Int callback_error = hypre_ParKrylovDestroyVector(array[i]);
         if (error == 0 && callback_error != 0)
         {
            error = callback_error;
         }
      }
   }
   {
      HYPRE_Int callback_error = hypre_ParKrylovFree(array);
      if (error == 0 && callback_error != 0)
      {
         error = callback_error;
      }
   }
   *array_pointer = NULL;
   return error;
}

static HYPRE_Int
reference_destroy_workspace(reference_GMRESData *data)
{
   HYPRE_Int error = 0;

   if (data == NULL)
   {
      return 0;
   }

   if (data->amg != NULL)
   {
      error = HYPRE_BoomerAMGDestroy(data->amg);
      data->amg = NULL;
   }
   {
      HYPRE_Int callback_error = reference_destroy_vector_array(
         &data->preconditioned, REFERENCE_GMRES_RESTART);
      if (error == 0) { error = callback_error; }
      callback_error = reference_destroy_vector_array(
         &data->basis, REFERENCE_GMRES_RESTART + 1);
      if (error == 0) { error = callback_error; }
   }
#define REFERENCE_FREE_HOST(pointer)                                      \
   do                                                               \
   {                                                                \
      if ((pointer) != NULL)                                        \
      {                                                             \
         HYPRE_Int callback_error = hypre_ParKrylovFree(pointer);   \
         if (error == 0 && callback_error != 0)                     \
         {                                                          \
            error = callback_error;                                 \
         }                                                          \
         (pointer) = NULL;                                          \
      }                                                             \
   } while (0)

   REFERENCE_FREE_HOST(data->hessenberg);
   REFERENCE_FREE_HOST(data->cosines);
   REFERENCE_FREE_HOST(data->sines);
   REFERENCE_FREE_HOST(data->rhs);
   REFERENCE_FREE_HOST(data->coefficients);

#undef REFERENCE_FREE_HOST

   data->matrix = NULL;
   data->base.is_setup = 0;
   return error;
}

static HYPRE_Int
reference_allocate_workspace(reference_GMRESData *data, HYPRE_Vector sample)
{
   data->basis = hypre_ParKrylovCreateVectorArray(
      REFERENCE_GMRES_RESTART + 1, sample);
   data->preconditioned = hypre_ParKrylovCreateVectorArray(
      REFERENCE_GMRES_RESTART, sample);
   data->hessenberg = (HYPRE_Real *) hypre_ParKrylovCAlloc(
      (size_t) (REFERENCE_GMRES_RESTART + 1),
      (size_t) REFERENCE_GMRES_RESTART * sizeof(HYPRE_Real),
      HYPRE_MEMORY_HOST);
   data->cosines = (HYPRE_Real *) hypre_ParKrylovCAlloc(
      REFERENCE_GMRES_RESTART, sizeof(HYPRE_Real), HYPRE_MEMORY_HOST);
   data->sines = (HYPRE_Real *) hypre_ParKrylovCAlloc(
      REFERENCE_GMRES_RESTART, sizeof(HYPRE_Real), HYPRE_MEMORY_HOST);
   data->rhs = (HYPRE_Real *) hypre_ParKrylovCAlloc(
      REFERENCE_GMRES_RESTART + 1, sizeof(HYPRE_Real), HYPRE_MEMORY_HOST);
   data->coefficients = (HYPRE_Real *) hypre_ParKrylovCAlloc(
      REFERENCE_GMRES_RESTART, sizeof(HYPRE_Real), HYPRE_MEMORY_HOST);

   if (data->basis == NULL || data->preconditioned == NULL ||
       data->hessenberg == NULL ||
       data->cosines == NULL || data->sines == NULL || data->rhs == NULL ||
       data->coefficients == NULL)
   {
      reference_destroy_workspace(data);
      return HYPRE_ERROR_GENERIC;
   }
   return 0;
}

HYPRE_Int
solver_create(HYPRE_Solver *solver,
              HYPRE_Real    relative_tolerance,
              HYPRE_Real    absolute_tolerance,
              HYPRE_Int     maximum_iterations)
{
   reference_GMRESData *data;

   if (solver == NULL)
   {
      return HYPRE_ERROR_GENERIC;
   }
   *solver = NULL;
   if (!reference_is_finite(relative_tolerance) ||
       !reference_is_finite(absolute_tolerance) ||
       relative_tolerance < 0.0 || absolute_tolerance < 0.0 ||
       maximum_iterations < 0)
   {
      return HYPRE_ERROR_GENERIC;
   }

   data = (reference_GMRESData *) hypre_ParKrylovCAlloc(
      1, sizeof(*data), HYPRE_MEMORY_HOST);
   if (data == NULL)
   {
      return HYPRE_ERROR_GENERIC;
   }

   data->base.setup = reference_GMRESSetup;
   data->base.solve = reference_GMRESSolve;
   data->base.destroy = reference_GMRESDestroy;
   data->base.is_setup = 0;

   /* Leave a fixed tenfold residual margin for independent verification. */
   data->relative_tolerance = 0.1 * relative_tolerance;
   data->absolute_tolerance = absolute_tolerance;
   data->maximum_iterations = maximum_iterations;
   *solver = (HYPRE_Solver) data;
   return 0;
}

static HYPRE_Int
reference_GMRESSetup(HYPRE_Solver solver, HYPRE_Matrix A,
               HYPRE_Vector b, HYPRE_Vector x)
{
   reference_GMRESData *data = (reference_GMRESData *) solver;
   HYPRE_Int error;

   if (data == NULL || A == NULL || b == NULL || x == NULL)
   {
      return HYPRE_ERROR_GENERIC;
   }

   error = reference_destroy_workspace(data);
   if (error != 0)
   {
      return error;
   }
   error = reference_allocate_workspace(data, b);
   if (error != 0)
   {
      return error;
   }

   error = HYPRE_BoomerAMGCreate(&data->amg);
   if (error == 0)
   {
      error |= HYPRE_BoomerAMGSetTol(data->amg, 0.0);
      error |= HYPRE_BoomerAMGSetMinIter(data->amg, 1);
      error |= HYPRE_BoomerAMGSetMaxIter(data->amg, 1);
      error |= HYPRE_BoomerAMGSetCycleType(data->amg, 1);
      error |= HYPRE_BoomerAMGSetCoarsenType(data->amg, 10);
      error |= HYPRE_BoomerAMGSetInterpType(data->amg, 6);
      error |= HYPRE_BoomerAMGSetPMaxElmts(data->amg, 4);
      error |= HYPRE_BoomerAMGSetStrongThreshold(data->amg, 0.25);
      error |= HYPRE_BoomerAMGSetRelaxType(data->amg, 6);
      error |= HYPRE_BoomerAMGSetNumSweeps(data->amg, 1);
      error |= HYPRE_BoomerAMGSetMaxLevels(data->amg, 25);
      error |= HYPRE_BoomerAMGSetMaxCoarseSize(data->amg, 9);
   }
   if (error == 0)
   {
      error = HYPRE_BoomerAMGSetup(
         data->amg, (HYPRE_ParCSRMatrix) A,
         (HYPRE_ParVector) b, (HYPRE_ParVector) x);
   }
   if (error != 0)
   {
      reference_destroy_workspace(data);
      return error;
   }
   data->matrix = A;
   data->base.is_setup = 1;
   return 0;
}

static HYPRE_Int
reference_apply_update(reference_GMRESData *data, HYPRE_Int dimension, HYPRE_Vector x)
{
   HYPRE_Int row;
   HYPRE_Int column;
   HYPRE_Int callback_error;

   if (dimension <= 0 || dimension > REFERENCE_GMRES_RESTART)
   {
      return HYPRE_ERROR_GENERIC;
   }

   for (row = dimension - 1; row >= 0; row--)
   {
      HYPRE_Real value = data->rhs[row];
      HYPRE_Real diagonal;

      for (column = row + 1; column < dimension; column++)
      {
         value -= reference_hessenberg_get(data, row, column) *
                  data->coefficients[column];
      }
      diagonal = reference_hessenberg_get(data, row, row);
      if (!reference_is_finite(value) || !reference_is_finite(diagonal) ||
          reference_absolute(diagonal) < HYPRE_REAL_MIN)
      {
         return HYPRE_ERROR_GENERIC;
      }
      data->coefficients[row] = value / diagonal;
      if (!reference_is_finite(data->coefficients[row]))
      {
         return HYPRE_ERROR_GENERIC;
      }
   }

   for (column = 0; column < dimension; column++)
   {
      callback_error = hypre_ParKrylovAxpy(
         data->coefficients[column], data->preconditioned[column], x);
      if (callback_error != 0)
      {
         return callback_error;
      }
   }
   return 0;
}

static HYPRE_Int
reference_recompute_residual(reference_GMRESData *data, HYPRE_Matrix A,
                       HYPRE_Vector b, HYPRE_Vector x,
                       HYPRE_Real *residual_norm)
{
   HYPRE_Int callback_error;

   callback_error = hypre_ParKrylovCopyVector(b, data->basis[0]);
   if (callback_error != 0)
   {
      return callback_error;
   }
   callback_error = hypre_ParKrylovMatvec(
      NULL, -1.0, A, x, 1.0, data->basis[0]);
   if (callback_error != 0)
   {
      return callback_error;
   }
   return reference_vector_norm(data->basis[0], residual_norm);
}

static HYPRE_Int
reference_GMRESSolve(HYPRE_Solver solver, HYPRE_Matrix A,
               HYPRE_Vector b, HYPRE_Vector x)
{
   reference_GMRESData *data = (reference_GMRESData *) solver;
   HYPRE_Real b_norm;
   HYPRE_Real residual_norm;
   HYPRE_Real denominator;
   HYPRE_Real tolerance;
   HYPRE_Int iteration = 0;
   HYPRE_Int callback_error;

   if (data == NULL || A == NULL || b == NULL || x == NULL ||
       data->basis == NULL || data->hessenberg == NULL ||
       data->matrix != A || !data->base.is_setup)
   {
      return HYPRE_ERROR_GENERIC;
   }

   if (reference_vector_norm(b, &b_norm) != 0 ||
       reference_recompute_residual(data, A, b, x, &residual_norm) != 0)
   {
      return HYPRE_ERROR_GENERIC;
   }
   if (b_norm == 0.0)
   {
      return hypre_ParKrylovClearVector(x);
   }
   denominator = b_norm;
   tolerance = data->relative_tolerance * denominator;
   if (data->absolute_tolerance > tolerance)
   {
      tolerance = data->absolute_tolerance;
   }
   if (!reference_is_finite(tolerance))
   {
      return HYPRE_ERROR_GENERIC;
   }
   if (residual_norm == 0.0 || residual_norm <= tolerance)
   {
      return 0;
   }

   while (iteration < data->maximum_iterations)
   {
      HYPRE_Int cycle_dimension = 0;
      HYPRE_Int column;

      callback_error = hypre_ParKrylovScaleVector(
         1.0 / residual_norm, data->basis[0]);
      if (callback_error != 0)
      {
         return callback_error;
      }
      data->rhs[0] = residual_norm;

      for (column = 0;
           column < REFERENCE_GMRES_RESTART &&
           iteration < data->maximum_iterations;
           column++)
      {
         HYPRE_Real next_norm;
         HYPRE_Real diagonal;
         HYPRE_Real subdiagonal;
         HYPRE_Real rotation_norm;
         HYPRE_Int row;

         callback_error = hypre_ParKrylovClearVector(data->preconditioned[column]);
         if (callback_error == 0)
         {
            callback_error = HYPRE_BoomerAMGSolve(
               data->amg, (HYPRE_ParCSRMatrix) A,
               (HYPRE_ParVector) data->basis[column],
               (HYPRE_ParVector) data->preconditioned[column]);
         }
         if (callback_error != 0)
         {
            return callback_error;
         }
         callback_error = hypre_ParKrylovMatvec(
            NULL, 1.0, A, data->preconditioned[column], 0.0,
            data->basis[column + 1]);
         if (callback_error != 0)
         {
            return callback_error;
         }

         /* Two-pass modified Gram-Schmidt controls loss of orthogonality. */
         for (row = 0; row <= column; row++)
         {
            HYPRE_Real projection = hypre_ParKrylovInnerProd(
               data->basis[row], data->basis[column + 1]);
            if (!reference_is_finite(projection))
            {
               return HYPRE_ERROR_GENERIC;
            }
            reference_hessenberg_set(data, row, column, projection);
            callback_error = hypre_ParKrylovAxpy(
               -projection, data->basis[row], data->basis[column + 1]);
            if (callback_error != 0)
            {
               return callback_error;
            }
         }
         for (row = 0; row <= column; row++)
         {
            HYPRE_Real correction = hypre_ParKrylovInnerProd(
               data->basis[row], data->basis[column + 1]);
            HYPRE_Real projection =
               reference_hessenberg_get(data, row, column) + correction;
            if (!reference_is_finite(correction) || !reference_is_finite(projection))
            {
               return HYPRE_ERROR_GENERIC;
            }
            reference_hessenberg_set(data, row, column, projection);
            callback_error = hypre_ParKrylovAxpy(
               -correction, data->basis[row], data->basis[column + 1]);
            if (callback_error != 0)
            {
               return callback_error;
            }
         }
         if (reference_vector_norm(data->basis[column + 1], &next_norm) != 0)
         {
            return HYPRE_ERROR_GENERIC;
         }
         reference_hessenberg_set(data, column + 1, column, next_norm);
         if (next_norm > HYPRE_REAL_MIN)
         {
            callback_error = hypre_ParKrylovScaleVector(
               1.0 / next_norm, data->basis[column + 1]);
            if (callback_error != 0)
            {
               return callback_error;
            }
         }

         for (row = 0; row < column; row++)
         {
            HYPRE_Real upper = reference_hessenberg_get(data, row, column);
            HYPRE_Real lower = reference_hessenberg_get(data, row + 1, column);
            reference_hessenberg_set(
               data, row, column,
               data->cosines[row] * upper + data->sines[row] * lower);
            reference_hessenberg_set(
               data, row + 1, column,
               -data->sines[row] * upper + data->cosines[row] * lower);
         }

         diagonal = reference_hessenberg_get(data, column, column);
         subdiagonal = reference_hessenberg_get(data, column + 1, column);
         rotation_norm = reference_pair_norm(diagonal, subdiagonal);
         if (!reference_is_finite(rotation_norm) ||
             rotation_norm < HYPRE_REAL_MIN)
         {
            return HYPRE_ERROR_GENERIC;
         }
         data->cosines[column] = diagonal / rotation_norm;
         data->sines[column] = subdiagonal / rotation_norm;
         reference_hessenberg_set(data, column, column, rotation_norm);
         reference_hessenberg_set(data, column + 1, column, 0.0);

         data->rhs[column + 1] =
            -data->sines[column] * data->rhs[column];
         data->rhs[column] = data->cosines[column] * data->rhs[column];
         if (!reference_is_finite(data->rhs[column]) ||
             !reference_is_finite(data->rhs[column + 1]))
         {
            return HYPRE_ERROR_GENERIC;
         }

         iteration++;
         cycle_dimension = column + 1;
         residual_norm = reference_absolute(data->rhs[column + 1]);
         if (residual_norm <= tolerance || next_norm <= HYPRE_REAL_MIN)
         {
            break;
         }
      }

      callback_error = reference_apply_update(data, cycle_dimension, x);
      if (callback_error != 0)
      {
         return callback_error;
      }
      callback_error = reference_recompute_residual(
         data, A, b, x, &residual_norm);
      if (callback_error != 0)
      {
         return callback_error;
      }
      if (residual_norm == 0.0 || residual_norm <= tolerance)
      {
         return 0;
      }
   }

   return HYPRE_ERROR_CONV;
}

static HYPRE_Int
reference_GMRESDestroy(HYPRE_Solver solver)
{
   reference_GMRESData *data = (reference_GMRESData *) solver;
   HYPRE_Int error;
   HYPRE_Int free_error;

   if (data == NULL)
   {
      return 0;
   }
   error = reference_destroy_workspace(data);
   free_error = hypre_ParKrylovFree(data);
   if (error != 0)
   {
      return error;
   }
   return free_error;
}
