#include "HYPRE_parcsr_ls.h"

/*
   A HYPRE GMRES(200), preconditioned by ILUT solver to measure speedup against.
*/

typedef struct
{
   hypre_Solver base;
   HYPRE_Real relative_tolerance;
   HYPRE_Real absolute_tolerance;
   HYPRE_Solver gmres;
   HYPRE_Solver amg;
} reference_Solver;

static HYPRE_Int
reference_destroy_objects(reference_Solver *data)
{
   HYPRE_Int error = 0;
   if (data->gmres != NULL)
   {
      error |= HYPRE_ParCSRGMRESDestroy(data->gmres);
      data->gmres = NULL;
   }
   if (data->amg != NULL)
   {
      error |= HYPRE_ILUDestroy(data->amg);
      data->amg = NULL;
   }
   data->base.is_setup = 0;
   return error;
}

static HYPRE_Int
reference_destroy(HYPRE_Solver solver)
{
   reference_Solver *data = (reference_Solver *) solver;
   HYPRE_Int error;
   if (data == NULL)
   {
      return 0;
   }
   error = reference_destroy_objects(data);
   error |= hypre_ParKrylovFree(data);
   return error;
}

static HYPRE_Int
reference_setup(HYPRE_Solver solver, HYPRE_Matrix matrix,
                HYPRE_Vector rhs, HYPRE_Vector solution)
{
   reference_Solver *data = (reference_Solver *) solver;
   HYPRE_ParCSRMatrix par_matrix = (HYPRE_ParCSRMatrix) matrix;
   HYPRE_ParVector par_rhs = (HYPRE_ParVector) rhs;
   HYPRE_ParVector par_solution = (HYPRE_ParVector) solution;
   MPI_Comm communicator;
   HYPRE_Int error;

   if (data == NULL || matrix == NULL || rhs == NULL || solution == NULL)
   {
      return HYPRE_ERROR_GENERIC;
   }
   error = reference_destroy_objects(data);
   error |= HYPRE_ParCSRMatrixGetComm(par_matrix, &communicator);
   error |= HYPRE_ILUCreate(&data->amg);
   if (error != 0)
   {
      reference_destroy_objects(data);
      return error;
   }

   error |= HYPRE_ILUSetType(data->amg, 1);
   error |= HYPRE_ILUSetLocalReordering(data->amg, 1);
   error |= HYPRE_ILUSetMaxIter(data->amg, 1);
   error |= HYPRE_ILUSetMaxNnzPerRow(data->amg, 1000);
   error |= HYPRE_ILUSetDropThreshold(data->amg, 1.0e-4);
   error |= HYPRE_ILUSetTol(data->amg, 0.0);

   error |= HYPRE_ParCSRGMRESCreate(communicator, &data->gmres);
   if (error == 0)
   {
      error |= HYPRE_ParCSRGMRESSetKDim(data->gmres, 200);
      error |= HYPRE_ParCSRGMRESSetMaxIter(data->gmres, 10000);
      error |= HYPRE_ParCSRGMRESSetTol(
         data->gmres, data->relative_tolerance);
      error |= HYPRE_ParCSRGMRESSetAbsoluteTol(
         data->gmres, data->absolute_tolerance);
      error |= HYPRE_ParCSRGMRESSetPrecond(
         data->gmres, HYPRE_ILUSolve,
         HYPRE_ILUSetup, data->amg);
   }
   if (error == 0)
   {
      error = HYPRE_ParCSRGMRESSetup(
         data->gmres, par_matrix, par_rhs, par_solution);
   }
   if (error != 0)
   {
      reference_destroy_objects(data);
      return error;
   }
   data->base.is_setup = 1;
   return 0;
}

static HYPRE_Int
reference_solve(HYPRE_Solver solver, HYPRE_Matrix matrix,
                HYPRE_Vector rhs, HYPRE_Vector solution)
{
   reference_Solver *data = (reference_Solver *) solver;
   if (data == NULL || !data->base.is_setup || data->gmres == NULL)
   {
      return HYPRE_ERROR_GENERIC;
   }
   return HYPRE_ParCSRGMRESSolve(
      data->gmres, (HYPRE_ParCSRMatrix) matrix,
      (HYPRE_ParVector) rhs, (HYPRE_ParVector) solution);
}

HYPRE_Int
solver_create(HYPRE_Solver *solver, HYPRE_Real relative_tolerance,
              HYPRE_Real absolute_tolerance)
{
   reference_Solver *data;
   HYPRE_Real target;
   if (solver == NULL || !isfinite((double) relative_tolerance) ||
       !isfinite((double) absolute_tolerance) ||
       relative_tolerance < 0.0 || absolute_tolerance < 0.0)
   {
      return HYPRE_ERROR_GENERIC;
   }
   *solver = NULL;
   data = (reference_Solver *) hypre_ParKrylovCAlloc(
      1, sizeof(*data), HYPRE_MEMORY_HOST);
   if (data == NULL)
   {
      return HYPRE_ERROR_GENERIC;
   }
   target = 0.1 * relative_tolerance;
   data->relative_tolerance = target < 1.0e-12 ? target : 1.0e-12;
   data->absolute_tolerance = absolute_tolerance;
   data->base.setup = reference_setup;
   data->base.solve = reference_solve;
   data->base.destroy = reference_destroy;
   data->base.is_setup = 0;
   *solver = (HYPRE_Solver) data;
   return 0;
}
