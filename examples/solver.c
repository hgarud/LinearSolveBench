#include "HYPRE_parcsr_ls.h"

/* A deliberately small starting point.  Participants may replace BoomerAMG
   with any solver assembled from the functions documented in BENCHMARK.md. */

HYPRE_Int
solver_create(HYPRE_Solver *solver, HYPRE_Real relative_tolerance,
              HYPRE_Real absolute_tolerance)
{
   HYPRE_Int error;
   HYPRE_Real target = 0.1 * relative_tolerance;
   (void) absolute_tolerance;

   error = HYPRE_BoomerAMGCreate(solver);
   if (error != 0)
   {
      return error;
   }

   error |= HYPRE_BoomerAMGSetTol(
      *solver, target < 1.0e-12 ? target : 1.0e-12);
   error |= HYPRE_BoomerAMGSetMaxIter(*solver, 200);
   error |= HYPRE_BoomerAMGSetCoarsenType(*solver, 10);
   error |= HYPRE_BoomerAMGSetInterpType(*solver, 6);
   error |= HYPRE_BoomerAMGSetPMaxElmts(*solver, 4);
   error |= HYPRE_BoomerAMGSetStrongThreshold(*solver, 0.25);
   error |= HYPRE_BoomerAMGSetRelaxType(*solver, 6);
   error |= HYPRE_BoomerAMGSetNumSweeps(*solver, 1);
   error |= HYPRE_BoomerAMGSetMaxLevels(*solver, 25);
   error |= HYPRE_BoomerAMGSetMaxCoarseSize(*solver, 9);
   return error;
}
