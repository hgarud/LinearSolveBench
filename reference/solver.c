#include "HYPRE_parcsr_ls.h"

/*
   Fixed FLASH reference: GMRES(50) + BoomerAMG with the website's numerical
   settings and captured tolerance, without its 1,000-iteration cutoff.
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
      error |= HYPRE_BoomerAMGDestroy(data->amg);
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
   error |= HYPRE_BoomerAMGCreate(&data->amg);
   if (error != 0)
   {
      reference_destroy_objects(data);
      return error;
   }

   error |= HYPRE_BoomerAMGSetPrintLevel(data->amg, 0);
   error |= HYPRE_BoomerAMGSetCoarsenType(data->amg, 10);
   error |= HYPRE_BoomerAMGSetInterpType(data->amg, 6);
   error |= HYPRE_BoomerAMGSetRelaxType(data->amg, 6);
   error |= HYPRE_BoomerAMGSetNumSweeps(data->amg, 1);
   error |= HYPRE_BoomerAMGSetMaxLevels(data->amg, 25);
   error |= HYPRE_BoomerAMGSetStrongThreshold(data->amg, 0.25);
   /* Explicitly retain the pinned HYPRE defaults used during discovery. */
   error |= HYPRE_BoomerAMGSetPMaxElmts(data->amg, 4);
   error |= HYPRE_BoomerAMGSetTruncFactor(data->amg, 0.0);
   error |= HYPRE_BoomerAMGSetTol(data->amg, 0.0);
   error |= HYPRE_BoomerAMGSetMaxIter(data->amg, 1);

   error |= HYPRE_ParCSRGMRESCreate(communicator, &data->gmres);
   if (error == 0)
   {
      error |= HYPRE_ParCSRGMRESSetKDim(data->gmres, 50);
      /* HYPRE has no unlimited sentinel. The runtime timeout is the practical
         limit; disable residual-history allocation proportional to max_iter. */
      error |= HYPRE_ParCSRGMRESSetMaxIter(data->gmres, HYPRE_INT_MAX);
      error |= HYPRE_ParCSRGMRESSetTol(
         data->gmres, data->relative_tolerance);
      error |= HYPRE_ParCSRGMRESSetAbsoluteTol(
         data->gmres, data->absolute_tolerance);
      error |= HYPRE_ParCSRGMRESSetLogging(data->gmres, 0);
      error |= HYPRE_ParCSRGMRESSetPrintLevel(data->gmres, 0);
      error |= HYPRE_ParCSRGMRESSetPrecond(
         data->gmres, HYPRE_BoomerAMGSolve,
         HYPRE_BoomerAMGSetup, data->amg);
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
   data->relative_tolerance = relative_tolerance;
   data->absolute_tolerance = absolute_tolerance;
   data->base.setup = reference_setup;
   data->base.solve = reference_solve;
   data->base.destroy = reference_destroy;
   data->base.is_setup = 0;
   *solver = (HYPRE_Solver) data;
   return 0;
}
