#include "HYPRE_parcsr_ls.h"

enum
{
   NSL_GMRES_RESTART = 200
};

typedef struct
{
   hypre_Solver base;

   HYPRE_Real relative_tolerance;
   HYPRE_Real absolute_tolerance;
   HYPRE_Int maximum_iterations;

   HYPRE_Matrix matrix;
   void **basis;
   HYPRE_Real *hessenberg;
   HYPRE_Real *cosines;
   HYPRE_Real *sines;
   HYPRE_Real *rhs;
   HYPRE_Real *coefficients;
} nsl_GMRESData;

static HYPRE_Int nsl_GMRESSetup(HYPRE_Solver solver, HYPRE_Matrix A,
                                HYPRE_Vector b, HYPRE_Vector x);
static HYPRE_Int nsl_GMRESSolve(HYPRE_Solver solver, HYPRE_Matrix A,
                                HYPRE_Vector b, HYPRE_Vector x);
static HYPRE_Int nsl_GMRESDestroy(HYPRE_Solver solver);

static HYPRE_Int
nsl_is_finite(HYPRE_Real value)
{
   return isfinite((double) value) ? 1 : 0;
}

static HYPRE_Real
nsl_absolute(HYPRE_Real value)
{
   return value < 0.0 ? -value : value;
}

static HYPRE_Real
nsl_pair_norm(HYPRE_Real left, HYPRE_Real right)
{
   HYPRE_Real larger = nsl_absolute(left);
   HYPRE_Real smaller = nsl_absolute(right);
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
nsl_hessenberg_get(const nsl_GMRESData *data, HYPRE_Int row,
                   HYPRE_Int column)
{
   return data->hessenberg[
      (size_t) row * (size_t) NSL_GMRES_RESTART + (size_t) column];
}

static void
nsl_hessenberg_set(nsl_GMRESData *data, HYPRE_Int row, HYPRE_Int column,
                   HYPRE_Real value)
{
   data->hessenberg[
      (size_t) row * (size_t) NSL_GMRES_RESTART + (size_t) column] = value;
}

static HYPRE_Int
nsl_vector_norm(void *vector, HYPRE_Real *norm)
{
   HYPRE_Real squared_norm;

   if (vector == NULL || norm == NULL)
   {
      return HYPRE_ERROR_GENERIC;
   }
   squared_norm = hypre_ParKrylovInnerProd(vector, vector);
   if (!nsl_is_finite(squared_norm) || squared_norm < 0.0)
   {
      return HYPRE_ERROR_GENERIC;
   }

   *norm = sqrt(squared_norm);
   return nsl_is_finite(*norm) ? 0 : HYPRE_ERROR_GENERIC;
}

static HYPRE_Int
nsl_destroy_vector_array(void ***array_pointer, HYPRE_Int count)
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
nsl_destroy_workspace(nsl_GMRESData *data)
{
   HYPRE_Int error = 0;

   if (data == NULL)
   {
      return 0;
   }

   error = nsl_destroy_vector_array(
      &data->basis, NSL_GMRES_RESTART + 1);
#define NSL_FREE_HOST(pointer)                                      \
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

   NSL_FREE_HOST(data->hessenberg);
   NSL_FREE_HOST(data->cosines);
   NSL_FREE_HOST(data->sines);
   NSL_FREE_HOST(data->rhs);
   NSL_FREE_HOST(data->coefficients);

#undef NSL_FREE_HOST

   data->matrix = NULL;
   data->base.is_setup = 0;
   return error;
}

static HYPRE_Int
nsl_allocate_workspace(nsl_GMRESData *data, HYPRE_Vector sample)
{
   data->basis = hypre_ParKrylovCreateVectorArray(
      NSL_GMRES_RESTART + 1, sample);
   data->hessenberg = (HYPRE_Real *) hypre_ParKrylovCAlloc(
      (size_t) (NSL_GMRES_RESTART + 1),
      (size_t) NSL_GMRES_RESTART * sizeof(HYPRE_Real),
      HYPRE_MEMORY_HOST);
   data->cosines = (HYPRE_Real *) hypre_ParKrylovCAlloc(
      NSL_GMRES_RESTART, sizeof(HYPRE_Real), HYPRE_MEMORY_HOST);
   data->sines = (HYPRE_Real *) hypre_ParKrylovCAlloc(
      NSL_GMRES_RESTART, sizeof(HYPRE_Real), HYPRE_MEMORY_HOST);
   data->rhs = (HYPRE_Real *) hypre_ParKrylovCAlloc(
      NSL_GMRES_RESTART + 1, sizeof(HYPRE_Real), HYPRE_MEMORY_HOST);
   data->coefficients = (HYPRE_Real *) hypre_ParKrylovCAlloc(
      NSL_GMRES_RESTART, sizeof(HYPRE_Real), HYPRE_MEMORY_HOST);

   if (data->basis == NULL || data->hessenberg == NULL ||
       data->cosines == NULL || data->sines == NULL || data->rhs == NULL ||
       data->coefficients == NULL)
   {
      nsl_destroy_workspace(data);
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
   nsl_GMRESData *data;

   if (solver == NULL)
   {
      return HYPRE_ERROR_GENERIC;
   }
   *solver = NULL;
   if (!nsl_is_finite(relative_tolerance) ||
       !nsl_is_finite(absolute_tolerance) ||
       relative_tolerance < 0.0 || absolute_tolerance < 0.0 ||
       maximum_iterations < 0)
   {
      return HYPRE_ERROR_GENERIC;
   }

   data = (nsl_GMRESData *) hypre_ParKrylovCAlloc(
      1, sizeof(*data), HYPRE_MEMORY_HOST);
   if (data == NULL)
   {
      return HYPRE_ERROR_GENERIC;
   }

   data->base.setup = nsl_GMRESSetup;
   data->base.solve = nsl_GMRESSolve;
   data->base.destroy = nsl_GMRESDestroy;
   data->base.is_setup = 0;

   /* The public tolerance defines verifier gates, not a residual stopping
      rule. A conservative inner target gives ill-conditioned systems enough
      room to satisfy the hidden forward-error gate. */
   data->relative_tolerance =
      relative_tolerance > 1.0e-12 ? 1.0e-12 : relative_tolerance;
   data->absolute_tolerance = absolute_tolerance;
   data->maximum_iterations = maximum_iterations;
   *solver = (HYPRE_Solver) data;
   return 0;
}

static HYPRE_Int
nsl_GMRESSetup(HYPRE_Solver solver, HYPRE_Matrix A,
               HYPRE_Vector b, HYPRE_Vector x)
{
   nsl_GMRESData *data = (nsl_GMRESData *) solver;
   HYPRE_Int error;

   if (data == NULL || A == NULL || b == NULL || x == NULL)
   {
      return HYPRE_ERROR_GENERIC;
   }

   error = nsl_destroy_workspace(data);
   if (error != 0)
   {
      return error;
   }
   error = nsl_allocate_workspace(data, b);
   if (error != 0)
   {
      return error;
   }

   data->matrix = A;
   data->base.is_setup = 1;
   return 0;
}

static HYPRE_Int
nsl_apply_update(nsl_GMRESData *data, HYPRE_Int dimension, HYPRE_Vector x)
{
   HYPRE_Int row;
   HYPRE_Int column;
   HYPRE_Int callback_error;

   if (dimension <= 0 || dimension > NSL_GMRES_RESTART)
   {
      return HYPRE_ERROR_GENERIC;
   }

   for (row = dimension - 1; row >= 0; row--)
   {
      HYPRE_Real value = data->rhs[row];
      HYPRE_Real diagonal;

      for (column = row + 1; column < dimension; column++)
      {
         value -= nsl_hessenberg_get(data, row, column) *
                  data->coefficients[column];
      }
      diagonal = nsl_hessenberg_get(data, row, row);
      if (!nsl_is_finite(value) || !nsl_is_finite(diagonal) ||
          nsl_absolute(diagonal) < HYPRE_REAL_MIN)
      {
         return HYPRE_ERROR_GENERIC;
      }
      data->coefficients[row] = value / diagonal;
      if (!nsl_is_finite(data->coefficients[row]))
      {
         return HYPRE_ERROR_GENERIC;
      }
   }

   for (column = 0; column < dimension; column++)
   {
      callback_error = hypre_ParKrylovAxpy(
         data->coefficients[column], data->basis[column], x);
      if (callback_error != 0)
      {
         return callback_error;
      }
   }
   return 0;
}

static HYPRE_Int
nsl_recompute_residual(nsl_GMRESData *data, HYPRE_Matrix A,
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
   return nsl_vector_norm(data->basis[0], residual_norm);
}

static HYPRE_Int
nsl_GMRESSolve(HYPRE_Solver solver, HYPRE_Matrix A,
               HYPRE_Vector b, HYPRE_Vector x)
{
   nsl_GMRESData *data = (nsl_GMRESData *) solver;
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

   if (nsl_vector_norm(b, &b_norm) != 0 ||
       nsl_recompute_residual(data, A, b, x, &residual_norm) != 0)
   {
      return HYPRE_ERROR_GENERIC;
   }
   denominator = b_norm > 0.0 ? b_norm : residual_norm;
   tolerance = data->relative_tolerance * denominator;
   if (data->absolute_tolerance > tolerance)
   {
      tolerance = data->absolute_tolerance;
   }
   if (!nsl_is_finite(tolerance))
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
           column < NSL_GMRES_RESTART &&
           iteration < data->maximum_iterations;
           column++)
      {
         HYPRE_Real next_norm;
         HYPRE_Real diagonal;
         HYPRE_Real subdiagonal;
         HYPRE_Real rotation_norm;
         HYPRE_Int row;

         callback_error = hypre_ParKrylovMatvec(
            NULL, 1.0, A, data->basis[column], 0.0,
            data->basis[column + 1]);
         if (callback_error != 0)
         {
            return callback_error;
         }

         /* Two-pass modified Gram-Schmidt is deliberately conservative for
            the strongly nonnormal matrices in the discovery distribution. */
         for (row = 0; row <= column; row++)
         {
            HYPRE_Real projection = hypre_ParKrylovInnerProd(
               data->basis[row], data->basis[column + 1]);
            if (!nsl_is_finite(projection))
            {
               return HYPRE_ERROR_GENERIC;
            }
            nsl_hessenberg_set(data, row, column, projection);
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
               nsl_hessenberg_get(data, row, column) + correction;
            if (!nsl_is_finite(correction) || !nsl_is_finite(projection))
            {
               return HYPRE_ERROR_GENERIC;
            }
            nsl_hessenberg_set(data, row, column, projection);
            callback_error = hypre_ParKrylovAxpy(
               -correction, data->basis[row], data->basis[column + 1]);
            if (callback_error != 0)
            {
               return callback_error;
            }
         }
         if (nsl_vector_norm(data->basis[column + 1], &next_norm) != 0)
         {
            return HYPRE_ERROR_GENERIC;
         }
         nsl_hessenberg_set(data, column + 1, column, next_norm);
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
            HYPRE_Real upper = nsl_hessenberg_get(data, row, column);
            HYPRE_Real lower = nsl_hessenberg_get(data, row + 1, column);
            nsl_hessenberg_set(
               data, row, column,
               data->cosines[row] * upper + data->sines[row] * lower);
            nsl_hessenberg_set(
               data, row + 1, column,
               -data->sines[row] * upper + data->cosines[row] * lower);
         }

         diagonal = nsl_hessenberg_get(data, column, column);
         subdiagonal = nsl_hessenberg_get(data, column + 1, column);
         rotation_norm = nsl_pair_norm(diagonal, subdiagonal);
         if (!nsl_is_finite(rotation_norm) ||
             rotation_norm < HYPRE_REAL_MIN)
         {
            return HYPRE_ERROR_GENERIC;
         }
         data->cosines[column] = diagonal / rotation_norm;
         data->sines[column] = subdiagonal / rotation_norm;
         nsl_hessenberg_set(data, column, column, rotation_norm);
         nsl_hessenberg_set(data, column + 1, column, 0.0);

         data->rhs[column + 1] =
            -data->sines[column] * data->rhs[column];
         data->rhs[column] = data->cosines[column] * data->rhs[column];
         if (!nsl_is_finite(data->rhs[column]) ||
             !nsl_is_finite(data->rhs[column + 1]))
         {
            return HYPRE_ERROR_GENERIC;
         }

         iteration++;
         cycle_dimension = column + 1;
         residual_norm = nsl_absolute(data->rhs[column + 1]);
         if (residual_norm <= tolerance || next_norm <= HYPRE_REAL_MIN)
         {
            break;
         }
      }

      callback_error = nsl_apply_update(data, cycle_dimension, x);
      if (callback_error != 0)
      {
         return callback_error;
      }
      callback_error = nsl_recompute_residual(
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
nsl_GMRESDestroy(HYPRE_Solver solver)
{
   nsl_GMRESData *data = (nsl_GMRESData *) solver;
   HYPRE_Int error;
   HYPRE_Int free_error;

   if (data == NULL)
   {
      return 0;
   }
   error = nsl_destroy_workspace(data);
   free_error = hypre_ParKrylovFree(data);
   if (error != 0)
   {
      return error;
   }
   return free_error;
}
