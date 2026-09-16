#ifndef NSL_HYPRE_SOLVER_HEADER
#define NSL_HYPRE_SOLVER_HEADER

#include "HYPRE_parcsr_ls.h"

#include <float.h>
#include <math.h>
#include <stddef.h>

#ifndef HYPRE_REAL_MIN
#define HYPRE_REAL_MIN DBL_MIN
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
   HYPRE_PtrToSolverFcn  setup;
   HYPRE_PtrToSolverFcn  solve;
   HYPRE_PtrToDestroyFcn destroy;
   HYPRE_Int             is_setup;
} hypre_Solver;

void *hypre_ParKrylovCAlloc(size_t count, size_t elt_size,
                            HYPRE_MemoryLocation location);
HYPRE_Int hypre_ParKrylovFree(void *ptr);
void *hypre_ParKrylovCreateVector(void *vvector);
void *hypre_ParKrylovCreateVectorArray(HYPRE_Int n, void *vvector);
HYPRE_Int hypre_ParKrylovDestroyVector(void *vvector);
void *hypre_ParKrylovMatvecCreate(void *A, void *x);
HYPRE_Int hypre_ParKrylovMatvec(void *matvec_data, HYPRE_Complex alpha,
                                void *A, void *x, HYPRE_Complex beta,
                                void *y);
HYPRE_Int hypre_ParKrylovMatvecT(void *matvec_data, HYPRE_Complex alpha,
                                 void *A, void *x, HYPRE_Complex beta,
                                 void *y);
HYPRE_Int hypre_ParKrylovMatvecDestroy(void *matvec_data);
HYPRE_Real hypre_ParKrylovInnerProd(void *x, void *y);
HYPRE_Int hypre_ParKrylovInnerProdTagged(void *x, void *y,
                                         HYPRE_Int *num_tags_ptr,
                                         HYPRE_Complex **iprod_ptr);
HYPRE_Int hypre_ParKrylovMassInnerProd(void *x, void **y, HYPRE_Int k,
                                       HYPRE_Int unroll, void *result);
HYPRE_Int hypre_ParKrylovMassDotpTwo(void *x, void *y, void **z, HYPRE_Int k,
                                     HYPRE_Int unroll, void *result_x,
                                     void *result_y);
HYPRE_Int hypre_ParKrylovMassAxpy(HYPRE_Complex *alpha, void **x, void *y,
                                  HYPRE_Int k, HYPRE_Int unroll);
HYPRE_Int hypre_ParKrylovCopyVector(void *x, void *y);
HYPRE_Int hypre_ParKrylovClearVector(void *x);
HYPRE_Int hypre_ParKrylovScaleVector(HYPRE_Complex alpha, void *x);
HYPRE_Int hypre_ParKrylovAxpy(HYPRE_Complex alpha, void *x, void *y);
HYPRE_Int hypre_ParKrylovCommInfo(void *A, HYPRE_Int *my_id,
                                  HYPRE_Int *num_procs);
HYPRE_Int hypre_ParKrylovIdentitySetup(void *vdata, void *A, void *b,
                                       void *x);
HYPRE_Int hypre_ParKrylovIdentity(void *vdata, void *A, void *b, void *x);

HYPRE_Int solver_create(HYPRE_Solver *solver,
                        HYPRE_Real    relative_tolerance,
                        HYPRE_Real    absolute_tolerance);

#ifdef __cplusplus
}
#endif

#endif
