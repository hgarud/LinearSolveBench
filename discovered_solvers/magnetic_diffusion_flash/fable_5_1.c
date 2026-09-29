#include "HYPRE_parcsr_ls.h"

/*
 * Cost-adaptive two-stage Krylov solver for captured FLASH magnetic-diffusion
 * systems (mass + dt * diffusion: nearly symmetric, diagonally dominant,
 * warm-started from the previous time level).
 *
 * Stage 0 (probe): true residual, tolerance, trivial exits.  A Rayleigh
 *   quotient of the smooth warm start with the Jacobi-scaled operator gives
 *   an optimistic condition-number estimate used as a loose admission test.
 *   The warm start is then rescaled by the least-squares factor
 *   alpha = (b, A x0) / (A x0, A x0): for a time-stepped diffusion problem
 *   the dominant slowly decaying mode of the previous solution is removed
 *   from the residual by this single scalar (a direction the Krylov space
 *   generated from r0 does not contain).  A Cauchy-Schwarz defect of
 *   (r, D^{-1} r) detects a numerically constant diagonal, in which case
 *   Jacobi scaling (a strided, cache-line-per-row pass) is dropped.
 *
 * Stage 1 (cheap): lean (Jacobi-)CG with implicitly scaled search
 *   directions (one matvec + 5 or 6 vector passes per iteration).  Its
 *   progress is modelled from first principles: for a dense spectrum the CG
 *   residual envelope follows the Chebyshev shape 1/cosh(k L).  The rate L
 *   is fitted to the best-so-far residual history (globally and over a
 *   trailing window, pessimistic minimum) and extrapolated to the target;
 *   the Lanczos tridiagonal assembled from the CG coefficients gives an
 *   optimistic (Ritz) rate used for confident early abandonment.  Decisions
 *   compare the predicted remaining cost with the multigrid path from the
 *   current iterate (setup + iterations for the remaining reduction), with
 *   a tight cap on the total stage-1 spend to bound the worst case.
 *
 * Stage 2 (robust): lean BoomerAMG (HMIS, ext+i, two aggressive levels
 *   with two-stage ext+i, forward/backward GS V-cycle, GE on the coarsest
 *   grid), built lazily, driving a preconditioned CG iteration; loss of
 *   definiteness or stagnation switches to right-preconditioned GMRES(30)
 *   with fused CGS and selective reorthogonalization on the same hierarchy.
 *
 * Cost model units: one vector pass (axpy) = 1, calibrated for a memory
 * bound single core.
 */

enum
{
   NSL_RESTART       = 30,
   NSL_UNROLL        = 4,
   NSL_CHEAP_CAP     = 128, /* hard cap on stage-1 iterations                 */
   NSL_CHEAP_RITZ_K  = 3,   /* first Ritz-based check (Lanczos steps)          */
   NSL_CHEAP_FIT_K   = 5,   /* first cosh-fit check                            */
   NSL_CHEAP_WINDOW  = 8,   /* trailing window for the local fit               */
   NSL_CHEAP_MAXVER  = 5,   /* max failed true-residual verifications          */
   NSL_CHEAP_RESERVE = 20,  /* iterations kept in reserve for stage 2          */
   NSL_CHEAP_MIN_CAP = 8,
   NSL_PCG_STALL     = 12,  /* AMG-CG stagnation window                        */
   NSL_BISECT        = 60   /* bisection steps                                 */
};

#define NSL_COST_AXPY    1.0
#define NSL_COST_IP      0.7
#define NSL_COST_DIAG    2.5     /* strided diagonal access: ~1 line per row   */
#define NSL_COST_MATVEC  4.5
#define NSL_AMG_SETUP    420.0   /* hierarchy construction                     */
#define NSL_AMG_ITER     36.0    /* one V-cycle + CG overhead                  */
#define NSL_AMG_LNRHO    1.5     /* -ln(0.22): AMG-CG contraction per iter     */
#define NSL_AMG_MIN_IT   2.0
#define NSL_AMG_MAX_IT   25.0
#define NSL_CAP_FACTOR   0.8     /* cap on total stage-1 spend (x AMG path)    */
#define NSL_PRED_MARGIN  1.25    /* pessimism applied to the fitted prediction */
#define NSL_RITZ_TRUST   0.85    /* optimism allowance for the Ritz bound      */
#define NSL_ADMIT_SLACK  60.0    /* admission slack beyond the break-even      */
#define NSL_JAC_DEV_TOL  6.0e-3  /* 1 - cos^2 threshold for constant diagonal  */
#define NSL_SCALE_GAIN   0.9     /* accept warm-start rescaling below this     */
#define NSL_L_MIN        1.5e-3  /* rate slower than this is a stall           */
#define NSL_L_BIG        40.0
#define NSL_SIGMA_LO     1.0e-60
#define NSL_SIGMA_HI     1.0e60

#define NSL_TRY(call)                          \
   do                                          \
   {                                           \
      HYPRE_Int nsl_err_ = (call);             \
      if (nsl_err_ != 0)                       \
      {                                        \
         return nsl_err_;                      \
      }                                        \
   } while (0)

#define NSL_FREE_HOST(ptr, err)                          \
   do                                                    \
   {                                                     \
      if ((ptr) != NULL)                                 \
      {                                                  \
         HYPRE_Int nsl_fe_ = hypre_ParKrylovFree(ptr);   \
         if ((err) == 0)                                 \
         {                                               \
            (err) = nsl_fe_;                             \
         }                                               \
         (ptr) = NULL;                                   \
      }                                                  \
   } while (0)

typedef struct
{
   hypre_Solver base;

   HYPRE_Real   rtol;
   HYPRE_Real   atol;
   HYPRE_Int    max_iter;
   HYPRE_Matrix matrix;

   HYPRE_Solver amg;
   HYPRE_Int    amg_ready;

   /* individually allocated work vectors */
   void *r;
   void *z;
   void *p;
   void *q;
   void *xb;

   /* stage-1 history: best-so-far residual, Lanczos coefficients */
   HYPRE_Real *hist;
   HYPRE_Real *tal;
   HYPRE_Real *tbe;

   /* GMRES fallback workspace (allocated lazily) */
   void         **V;
   HYPRE_Real    *H;
   HYPRE_Real    *cs;
   HYPRE_Real    *sn;
   HYPRE_Real    *g;
   HYPRE_Complex *y;
   HYPRE_Complex *coef;
} nsl_Data;

static HYPRE_Int nsl_Setup(HYPRE_Solver solver, HYPRE_Matrix A,
                           HYPRE_Vector b, HYPRE_Vector x);
static HYPRE_Int nsl_Solve(HYPRE_Solver solver, HYPRE_Matrix A,
                           HYPRE_Vector b, HYPRE_Vector x);
static HYPRE_Int nsl_Destroy(HYPRE_Solver solver);

/* ------------------------------------------------------------------ */
/* small helpers                                                       */
/* ------------------------------------------------------------------ */

static HYPRE_Int
nsl_finite(HYPRE_Real value)
{
   return isfinite((double) value) ? 1 : 0;
}

static HYPRE_Real
nsl_norm(void *vector)
{
   HYPRE_Real s = hypre_ParKrylovInnerProd(vector, vector);
   return sqrt(s);
}

static HYPRE_Int
nsl_destroy_vec(void **vp)
{
   HYPRE_Int err = 0;
   if (vp != NULL && *vp != NULL)
   {
      err = hypre_ParKrylovDestroyVector(*vp);
      *vp = NULL;
   }
   return err;
}

static HYPRE_Int
nsl_true_residual(HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x,
                  void *r, HYPRE_Real *rnorm)
{
   NSL_TRY(hypre_ParKrylovCopyVector(b, r));
   NSL_TRY(hypre_ParKrylovMatvec(NULL, -1.0, A, x, 1.0, r));
   *rnorm = nsl_norm(r);
   return 0;
}

static HYPRE_Int
nsl_diag_scale(HYPRE_Matrix A, void *in, void *out)
{
   return HYPRE_ParCSRDiagScale(NULL, (HYPRE_ParCSRMatrix) A,
                                (HYPRE_ParVector) in, (HYPRE_ParVector) out);
}

static HYPRE_Int
nsl_apply_amg(nsl_Data *d, HYPRE_Matrix A, void *in, void *out)
{
   HYPRE_Int err;
   NSL_TRY(hypre_ParKrylovClearVector(out));
   err = HYPRE_BoomerAMGSolve(d->amg, (HYPRE_ParCSRMatrix) A,
                              (HYPRE_ParVector) in, (HYPRE_ParVector) out);
   if (err != 0 && err != HYPRE_ERROR_CONV)
   {
      return err;
   }
   return 0;
}

/* ------------------------------------------------------------------ */
/* workspace management                                                */
/* ------------------------------------------------------------------ */

static HYPRE_Int
nsl_release_gmres(nsl_Data *d)
{
   HYPRE_Int err = 0;
   HYPRE_Int e;
   HYPRE_Int i;

   if (d->V != NULL)
   {
      /* vector zero owns the shared storage: destroy it last */
      for (i = NSL_RESTART; i >= 0; i--)
      {
         if (d->V[i] != NULL)
         {
            e = hypre_ParKrylovDestroyVector(d->V[i]);
            if (err == 0) { err = e; }
            d->V[i] = NULL;
         }
      }
      e = hypre_ParKrylovFree(d->V);
      if (err == 0) { err = e; }
      d->V = NULL;
   }

   NSL_FREE_HOST(d->H, err);
   NSL_FREE_HOST(d->cs, err);
   NSL_FREE_HOST(d->sn, err);
   NSL_FREE_HOST(d->g, err);
   NSL_FREE_HOST(d->y, err);
   NSL_FREE_HOST(d->coef, err);
   return err;
}

static HYPRE_Int
nsl_alloc_gmres(nsl_Data *d, HYPRE_Vector sample)
{
   const HYPRE_Int m = NSL_RESTART;

   d->V = hypre_ParKrylovCreateVectorArray(m + 1, sample);
   d->H = (HYPRE_Real *) hypre_ParKrylovCAlloc(
      (size_t) (m + 1) * (size_t) m, sizeof(HYPRE_Real), HYPRE_MEMORY_HOST);
   d->cs = (HYPRE_Real *) hypre_ParKrylovCAlloc(
      (size_t) m, sizeof(HYPRE_Real), HYPRE_MEMORY_HOST);
   d->sn = (HYPRE_Real *) hypre_ParKrylovCAlloc(
      (size_t) m, sizeof(HYPRE_Real), HYPRE_MEMORY_HOST);
   d->g = (HYPRE_Real *) hypre_ParKrylovCAlloc(
      (size_t) (m + 1), sizeof(HYPRE_Real), HYPRE_MEMORY_HOST);
   d->y = (HYPRE_Complex *) hypre_ParKrylovCAlloc(
      (size_t) (m + 1), sizeof(HYPRE_Complex), HYPRE_MEMORY_HOST);
   d->coef = (HYPRE_Complex *) hypre_ParKrylovCAlloc(
      (size_t) (m + 1), sizeof(HYPRE_Complex), HYPRE_MEMORY_HOST);

   if (d->V == NULL || d->H == NULL || d->cs == NULL || d->sn == NULL ||
       d->g == NULL || d->y == NULL || d->coef == NULL)
   {
      nsl_release_gmres(d);
      return HYPRE_ERROR_GENERIC;
   }
   return 0;
}

static HYPRE_Int
nsl_release_all(nsl_Data *d)
{
   HYPRE_Int err;
   HYPRE_Int e;

   if (d == NULL)
   {
      return 0;
   }
   err = nsl_release_gmres(d);

   e = nsl_destroy_vec(&d->r);  if (err == 0) { err = e; }
   e = nsl_destroy_vec(&d->z);  if (err == 0) { err = e; }
   e = nsl_destroy_vec(&d->p);  if (err == 0) { err = e; }
   e = nsl_destroy_vec(&d->q);  if (err == 0) { err = e; }
   e = nsl_destroy_vec(&d->xb); if (err == 0) { err = e; }

   NSL_FREE_HOST(d->hist, err);
   NSL_FREE_HOST(d->tal, err);
   NSL_FREE_HOST(d->tbe, err);

   if (d->amg != NULL)
   {
      e = HYPRE_BoomerAMGDestroy(d->amg);
      if (err == 0) { err = e; }
      d->amg = NULL;
   }
   d->amg_ready = 0;
   d->matrix = NULL;
   d->base.is_setup = 0;
   return err;
}

/* ------------------------------------------------------------------ */
/* preconditioner configuration                                        */
/* ------------------------------------------------------------------ */

static HYPRE_Int
nsl_configure_amg(HYPRE_Solver amg)
{
   NSL_TRY(HYPRE_BoomerAMGSetCoarsenType(amg, 10));       /* HMIS            */
   NSL_TRY(HYPRE_BoomerAMGSetInterpType(amg, 6));         /* ext+i           */
   NSL_TRY(HYPRE_BoomerAMGSetPMaxElmts(amg, 4));
   NSL_TRY(HYPRE_BoomerAMGSetTruncFactor(amg, 0.0));
   NSL_TRY(HYPRE_BoomerAMGSetAggNumLevels(amg, 2));       /* lean hierarchy  */
   NSL_TRY(HYPRE_BoomerAMGSetAggInterpType(amg, 1));      /* 2-stage ext+i   */
   NSL_TRY(HYPRE_BoomerAMGSetAggPMaxElmts(amg, 4));
   NSL_TRY(HYPRE_BoomerAMGSetAggTruncFactor(amg, 0.0));
   NSL_TRY(HYPRE_BoomerAMGSetNumPaths(amg, 1));
   NSL_TRY(HYPRE_BoomerAMGSetStrongThreshold(amg, 0.5));
   NSL_TRY(HYPRE_BoomerAMGSetMaxRowSum(amg, 0.9));
   NSL_TRY(HYPRE_BoomerAMGSetMaxLevels(amg, 25));
   NSL_TRY(HYPRE_BoomerAMGSetMaxCoarseSize(amg, 64));
   NSL_TRY(HYPRE_BoomerAMGSetRelaxType(amg, 3));
   NSL_TRY(HYPRE_BoomerAMGSetCycleRelaxType(amg, 3, 1));  /* forward GS      */
   NSL_TRY(HYPRE_BoomerAMGSetCycleRelaxType(amg, 4, 2));  /* backward GS     */
   NSL_TRY(HYPRE_BoomerAMGSetCycleRelaxType(amg, 9, 3));  /* coarse GE       */
   NSL_TRY(HYPRE_BoomerAMGSetRelaxOrder(amg, 0));
   NSL_TRY(HYPRE_BoomerAMGSetNumSweeps(amg, 1));
   NSL_TRY(HYPRE_BoomerAMGSetCycleType(amg, 1));
   NSL_TRY(HYPRE_BoomerAMGSetMaxIter(amg, 1));
   NSL_TRY(HYPRE_BoomerAMGSetTol(amg, 0.0));
   return 0;
}

/* ------------------------------------------------------------------ */
/* factory                                                             */
/* ------------------------------------------------------------------ */

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
   if (!nsl_finite(relative_tolerance) || !nsl_finite(absolute_tolerance) ||
       relative_tolerance < 0.0 || absolute_tolerance < 0.0 ||
       maximum_iterations < 0)
   {
      return HYPRE_ERROR_GENERIC;
   }

   d = (nsl_Data *) hypre_ParKrylovCAlloc(1, sizeof(*d), HYPRE_MEMORY_HOST);
   if (d == NULL)
   {
      return HYPRE_ERROR_GENERIC;
   }
   d->base.setup    = nsl_Setup;
   d->base.solve    = nsl_Solve;
   d->base.destroy  = nsl_Destroy;
   d->base.is_setup = 0;
   d->rtol     = relative_tolerance;
   d->atol     = absolute_tolerance;
   d->max_iter = maximum_iterations;
   *solver = (HYPRE_Solver) d;
   return 0;
}

/* ------------------------------------------------------------------ */
/* setup                                                               */
/* ------------------------------------------------------------------ */

static HYPRE_Int
nsl_Setup(HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x)
{
   nsl_Data *d = (nsl_Data *) solver;
   HYPRE_Int err;

   if (d == NULL || A == NULL || b == NULL || x == NULL)
   {
      return HYPRE_ERROR_GENERIC;
   }
   err = nsl_release_all(d);
   if (err != 0)
   {
      return err;
   }

   d->r  = hypre_ParKrylovCreateVector(b);
   d->z  = hypre_ParKrylovCreateVector(b);
   d->p  = hypre_ParKrylovCreateVector(b);
   d->q  = hypre_ParKrylovCreateVector(b);
   d->xb = hypre_ParKrylovCreateVector(b);
   d->hist = (HYPRE_Real *) hypre_ParKrylovCAlloc(
      (size_t) (NSL_CHEAP_CAP + 2), sizeof(HYPRE_Real), HYPRE_MEMORY_HOST);
   d->tal = (HYPRE_Real *) hypre_ParKrylovCAlloc(
      (size_t) (NSL_CHEAP_CAP + 2), sizeof(HYPRE_Real), HYPRE_MEMORY_HOST);
   d->tbe = (HYPRE_Real *) hypre_ParKrylovCAlloc(
      (size_t) (NSL_CHEAP_CAP + 2), sizeof(HYPRE_Real), HYPRE_MEMORY_HOST);
   if (d->r == NULL || d->z == NULL || d->p == NULL || d->q == NULL ||
       d->xb == NULL || d->hist == NULL || d->tal == NULL || d->tbe == NULL)
   {
      err = HYPRE_ERROR_GENERIC;
      goto fail;
   }

   d->amg = NULL;
   err = HYPRE_BoomerAMGCreate(&d->amg);
   if (err != 0)
   {
      d->amg = NULL;
      goto fail;
   }
   err = nsl_configure_amg(d->amg);
   if (err != 0)
   {
      goto fail;
   }

   /* the hierarchy itself is built lazily, only when stage 2 is needed */
   d->amg_ready = 0;
   d->matrix = A;
   d->base.is_setup = 1;
   return 0;

fail:
   nsl_release_all(d);
   return err != 0 ? err : HYPRE_ERROR_GENERIC;
}

/* ------------------------------------------------------------------ */
/* CG convergence model helpers                                        */
/* ------------------------------------------------------------------ */

/* cost of the multigrid path for a residual reduction exp(lnred) */
static HYPRE_Real
nsl_amg_cost(HYPRE_Real lnred)
{
   HYPRE_Real n;
   if (!nsl_finite(lnred) || !(lnred > 0.0))
   {
      lnred = 0.0;
   }
   n = lnred / NSL_AMG_LNRHO + 1.0;
   if (n < NSL_AMG_MIN_IT) { n = NSL_AMG_MIN_IT; }
   if (n > NSL_AMG_MAX_IT) { n = NSL_AMG_MAX_IT; }
   return NSL_AMG_SETUP + n * NSL_AMG_ITER;
}

/* asymptotic CG rate exponent L = -ln((sqrt(k)-1)/(sqrt(k)+1)) */
static HYPRE_Real
nsl_lnrate_from_kappa(HYPRE_Real kappa)
{
   HYPRE_Real s;
   HYPRE_Real rho;
   if (!nsl_finite(kappa) || kappa < 1.0)
   {
      kappa = 1.0;
   }
   s = sqrt(kappa);
   rho = (s - 1.0) / (s + 1.0);
   if (!(rho > 0.0))
   {
      return NSL_L_BIG;
   }
   rho = -log(rho);
   if (!nsl_finite(rho) || rho > NSL_L_BIG)
   {
      return NSL_L_BIG;
   }
   return rho;
}

/* acosh(ratio): number of "cosh units" separating two residual levels */
static HYPRE_Real
nsl_cosh_target(HYPRE_Real ratio)
{
   if (!nsl_finite(ratio) || !(ratio > 1.0))
   {
      return 0.0;
   }
   return acosh(ratio);
}

static HYPRE_Real
nsl_lncosh(HYPRE_Real x)
{
   x = fabs(x);
   return x + log(0.5 * (1.0 + exp(-2.0 * x)));
}

/* Fit the Chebyshev envelope s_k = s_0 / cosh(k L) through two history
   points: solve lncosh(k2 L) - lncosh(k1 L) = R (monotone in L). */
static HYPRE_Real
nsl_fit_cosh_rate(HYPRE_Int k1, HYPRE_Int k2, HYPRE_Real R)
{
   HYPRE_Real lo = 0.0;
   HYPRE_Real hi = NSL_L_BIG;
   HYPRE_Int it;

   if (!nsl_finite(R) || !(R > 0.0) || k2 <= k1 || k1 < 0)
   {
      return 0.0;
   }
   if (nsl_lncosh((HYPRE_Real) k2 * hi) - nsl_lncosh((HYPRE_Real) k1 * hi) < R)
   {
      return hi;
   }
   for (it = 0; it < NSL_BISECT; it++)
   {
      HYPRE_Real mid = 0.5 * (lo + hi);
      HYPRE_Real g = nsl_lncosh((HYPRE_Real) k2 * mid) -
                     nsl_lncosh((HYPRE_Real) k1 * mid);
      if (g < R)
      {
         lo = mid;
      }
      else
      {
         hi = mid;
      }
   }
   return 0.5 * (lo + hi);
}

/* Sturm count: number of eigenvalues of T (diag dg, squared off-diag e2)
   strictly smaller than x */
static HYPRE_Int
nsl_sturm_count(const HYPRE_Real *dg, const HYPRE_Real *e2, HYPRE_Int m,
                HYPRE_Real x, HYPRE_Real pivmin)
{
   HYPRE_Int count = 0;
   HYPRE_Int i;
   HYPRE_Real q = dg[0] - x;

   if (q < 0.0)
   {
      count++;
   }
   for (i = 1; i < m; i++)
   {
      if (fabs(q) < pivmin)
      {
         q = (q < 0.0) ? -pivmin : pivmin;
      }
      q = dg[i] - x - e2[i - 1] / q;
      if (q < 0.0)
      {
         count++;
      }
   }
   return count;
}

/* extreme Ritz values of the Lanczos tridiagonal assembled from the CG
   coefficients alpha_i (al) and beta_i (be):
   d_i = 1/al_i + be_{i-1}/al_{i-1},  e_i = sqrt(be_i)/al_i            */
static HYPRE_Int
nsl_ritz_extremes(const HYPRE_Real *al, const HYPRE_Real *be, HYPRE_Int m,
                  HYPRE_Real *tmin, HYPRE_Real *tmax)
{
   HYPRE_Real dg[NSL_CHEAP_CAP + 2];
   HYPRE_Real e2[NSL_CHEAP_CAP + 2];
   HYPRE_Real lo = HUGE_VAL;
   HYPRE_Real hi = -HUGE_VAL;
   HYPRE_Real scale;
   HYPRE_Real pivmin;
   HYPRE_Real a;
   HYPRE_Real c;
   HYPRE_Int i;
   HYPRE_Int it;

   if (m < 1 || m > NSL_CHEAP_CAP + 1)
   {
      return 0;
   }
   for (i = 0; i < m; i++)
   {
      if (!nsl_finite(al[i]) || al[i] == 0.0)
      {
         return 0;
      }
      dg[i] = 1.0 / al[i];
      if (i > 0)
      {
         dg[i] += be[i - 1] / al[i - 1];
      }
      if (!nsl_finite(dg[i]))
      {
         return 0;
      }
   }
   for (i = 0; i < m - 1; i++)
   {
      e2[i] = be[i] / (al[i] * al[i]);
      if (!nsl_finite(e2[i]) || e2[i] < 0.0)
      {
         return 0;
      }
   }
   for (i = 0; i < m; i++)
   {
      HYPRE_Real rad = 0.0;
      if (i > 0)
      {
         rad += sqrt(e2[i - 1]);
      }
      if (i < m - 1)
      {
         rad += sqrt(e2[i]);
      }
      if (dg[i] - rad < lo) { lo = dg[i] - rad; }
      if (dg[i] + rad > hi) { hi = dg[i] + rad; }
   }
   scale = fabs(lo) > fabs(hi) ? fabs(lo) : fabs(hi);
   if (!nsl_finite(scale) || !(scale > 0.0))
   {
      return 0;
   }
   pivmin = scale * 1.0e-300;
   if (pivmin < HYPRE_REAL_MIN)
   {
      pivmin = HYPRE_REAL_MIN;
   }

   /* smallest eigenvalue: inf { x : count(x) >= 1 } */
   a = lo;
   c = hi;
   for (it = 0; it < NSL_BISECT; it++)
   {
      HYPRE_Real mid = 0.5 * (a + c);
      if (nsl_sturm_count(dg, e2, m, mid, pivmin) >= 1)
      {
         c = mid;
      }
      else
      {
         a = mid;
      }
   }
   *tmin = c;

   /* largest eigenvalue: sup { x : count(x) < m } */
   a = lo;
   c = hi;
   for (it = 0; it < NSL_BISECT; it++)
   {
      HYPRE_Real mid = 0.5 * (a + c);
      if (nsl_sturm_count(dg, e2, m, mid, pivmin) >= m)
      {
         c = mid;
      }
      else
      {
         a = mid;
      }
   }
   *tmax = a;
   if (*tmax < *tmin)
   {
      *tmax = *tmin;
   }
   return 1;
}

/* stage-1 admission from the Rayleigh-quotient probe (loose: the probe is
   optimistic, the in-loop monitors take the real decisions) */
static HYPRE_Int
nsl_predict_cheap(HYPRE_Real rq, HYPRE_Real rnorm, HYPRE_Real tol,
                  HYPRE_Real iter_cost, HYPRE_Int cap)
{
   HYPRE_Real kappa;
   HYPRE_Real L;
   HYPRE_Real target;
   HYPRE_Real ktot;
   HYPRE_Real amg;

   if (cap <= 0)
   {
      return 0;
   }
   if (!(tol > 0.0) || !nsl_finite(tol) || !(rnorm > 0.0))
   {
      return 0;
   }
   if (rnorm <= tol)
   {
      return 1;
   }
   if (!nsl_finite(rq) || !(rq > 0.0))
   {
      return 1;   /* no information: probe with the cheap iteration */
   }
   /* spectrum of D^{-1}A for these stencils is ~symmetric about 1 inside
      (0,2); the Rayleigh quotient of a smooth vector overestimates
      lambda_min, so 2/rq is a (roughly) lower-bound-type kappa estimate */
   kappa = 2.0 / rq;
   L = nsl_lnrate_from_kappa(kappa);
   target = nsl_cosh_target(rnorm / (0.9 * tol));
   ktot = target / L;
   amg = nsl_amg_cost(log(rnorm / tol));
   if (!nsl_finite(ktot))
   {
      return 0;
   }
   if (ktot > (HYPRE_Real) cap)
   {
      return 0;
   }
   return ktot * iter_cost <= amg + NSL_ADMIT_SLACK ? 1 : 0;
}

/* stage-1 bail-out decision: 1 = abandon the cheap phase */
static HYPRE_Int
nsl_cheap_bail(nsl_Data *d, HYPRE_Int k, HYPRE_Int m, HYPRE_Real s,
               HYPRE_Real s_tol, HYPRE_Real iter_cost, HYPRE_Real cap_total)
{
   const HYPRE_Real *hist = d->hist;
   HYPRE_Real spent = (HYPRE_Real) k * iter_cost;
   HYPRE_Real s0 = hist[0];
   HYPRE_Real lnred;
   HYPRE_Real amg_here;
   HYPRE_Real target;

   if (!(s > s_tol))
   {
      return 0;
   }
   lnred = log(s / s_tol);
   if (!nsl_finite(lnred))
   {
      return 1;
   }
   if (spent >= cap_total)
   {
      return 1;
   }
   amg_here = nsl_amg_cost(lnred);
   if (!(s0 > s))
   {
      s0 = s;
   }
   target = nsl_cosh_target(s0 / s_tol);
   if (!(target > 0.0))
   {
      return 0;
   }

   /* optimistic Ritz-based bound: exceeding the budget is a confident bail */
   if (m >= NSL_CHEAP_RITZ_K)
   {
      HYPRE_Real tmin;
      HYPRE_Real tmax;
      HYPRE_Real kappa = -1.0;
      if (nsl_ritz_extremes(d->tal, d->tbe, m, &tmin, &tmax))
      {
         if (tmin > 0.0)
         {
            kappa = tmax / tmin;
         }
         else if (tmax < 0.0)
         {
            kappa = tmin / tmax;
         }
      }
      if (nsl_finite(kappa) && kappa > 0.0)
      {
         HYPRE_Real Lr = nsl_lnrate_from_kappa(kappa);
         HYPRE_Real rem = target / Lr - (HYPRE_Real) k;
         if (nsl_finite(rem) && rem > 0.0 &&
             NSL_RITZ_TRUST * rem * iter_cost > amg_here)
         {
            return 1;
         }
      }
   }

   /* Chebyshev-envelope fit of the observed history (pessimistic minimum
      of the global and the trailing-window rate) */
   if (k >= NSL_CHEAP_FIT_K)
   {
      HYPRE_Int w = k < NSL_CHEAP_WINDOW ? k : NSL_CHEAP_WINDOW;
      HYPRE_Real sk = hist[k];
      HYPRE_Real sw = hist[k - w];
      HYPRE_Real Lg;
      HYPRE_Real Ll;
      HYPRE_Real L;
      HYPRE_Real rem;
      HYPRE_Real pred;

      if (!(sk > 0.0) || !(sw > 0.0) || !nsl_finite(sk) || !nsl_finite(sw))
      {
         return 1;
      }
      Lg = (hist[0] > sk) ? nsl_cosh_target(hist[0] / sk) / (HYPRE_Real) k
                          : 0.0;
      Ll = (sw > sk) ? nsl_fit_cosh_rate(k - w, k, log(sw / sk)) : 0.0;
      L = Lg < Ll ? Lg : Ll;
      if (!nsl_finite(L) || !(L > NSL_L_MIN))
      {
         return 1;   /* stagnation */
      }
      rem = target / L - (HYPRE_Real) k;
      if (rem < 1.0)
      {
         rem = 1.0;
      }
      pred = NSL_PRED_MARGIN * rem * iter_cost;
      if (!nsl_finite(pred) || pred > amg_here)
      {
         return 1;
      }
      if (spent + pred > cap_total)
      {
         return 1;
      }
   }
   return 0;
}

/* ------------------------------------------------------------------ */
/* stage 1: lean (Jacobi-)CG with implicitly scaled directions, Lanczos */
/* Ritz monitoring and Chebyshev-envelope projected bail-out            */
/* ------------------------------------------------------------------ */

static HYPRE_Int
nsl_cheap_phase(nsl_Data *d, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x,
                HYPRE_Real tol, HYPRE_Int cap, HYPRE_Int jac,
                HYPRE_Int z_ready, HYPRE_Real iter_cost, HYPRE_Real cap_total,
                HYPRE_Real *rnorm_io, HYPRE_Int *iters_io,
                HYPRE_Int *converged)
{
   void *r  = d->r;
   void *z  = d->z;
   void *P  = d->p;    /* scaled search direction: p = sigma * P         */
   void *Q  = d->q;    /* A P                                             */
   void *zv = jac ? z : r;   /* preconditioned residual                   */
   HYPRE_Real *hist = d->hist;
   const HYPRE_Real tol2 = tol * tol;
   HYPRE_Real rnorm = *rnorm_io;
   HYPRE_Real rz;
   HYPRE_Real sgn_m;          /* sign of r' M^{-1} r                      */
   HYPRE_Real sgn_a = 0.0;    /* sign of p' A p, fixed at first iteration */
   HYPRE_Real ratio;          /* ||r||^2 / |r' M^{-1} r|, slowly varying  */
   HYPRE_Real sigma = 1.0;
   HYPRE_Real s0;
   HYPRE_Real sbest;
   HYPRE_Real s;
   HYPRE_Int k = 0;
   HYPRE_Int kt = 0;          /* Lanczos steps since the last (re)start   */
   HYPRE_Int nver = 0;
   HYPRE_Int have_true = 1;   /* r holds b - A x */

   *converged = 0;

   if (jac && !z_ready)
   {
      NSL_TRY(nsl_diag_scale(A, r, z));
   }
   rz = hypre_ParKrylovInnerProd(r, zv);
   if (!nsl_finite(rz) || rz == 0.0)
   {
      goto done;
   }
   sgn_m = rz > 0.0 ? 1.0 : -1.0;
   ratio = jac ? (rnorm * rnorm) / fabs(rz) : 1.0;
   if (!nsl_finite(ratio) || !(ratio > 0.0))
   {
      goto done;
   }
   NSL_TRY(hypre_ParKrylovCopyVector(zv, P));   /* p_0 = z_0 */
   s0 = sqrt(fabs(rz));
   sbest = s0;
   hist[0] = s0;

   while (k < cap && *iters_io < d->max_iter)
   {
      HYPRE_Real pq;
      HYPRE_Real as;
      HYPRE_Real alpha;
      HYPRE_Real beta;
      HYPRE_Real rz_new;
      HYPRE_Real sig_new;
      HYPRE_Real s_tol;

      NSL_TRY(hypre_ParKrylovMatvec(NULL, 1.0, A, P, 0.0, Q));
      pq = hypre_ParKrylovInnerProd(P, Q);
      if (!nsl_finite(pq) || pq == 0.0)
      {
         break;
      }
      if (sgn_a == 0.0)
      {
         sgn_a = pq > 0.0 ? 1.0 : -1.0;
      }
      else if (!(sgn_a * pq > 0.0))
      {
         break;   /* indefinite / non-symmetric behaviour: hand over */
      }
      /* alpha = rz / (sigma^2 pq); the update coefficient is alpha*sigma */
      as = rz / (sigma * pq);
      alpha = as / sigma;
      if (!nsl_finite(as) || !nsl_finite(alpha))
      {
         break;
      }

      NSL_TRY(hypre_ParKrylovAxpy(as, P, x));
      NSL_TRY(hypre_ParKrylovAxpy(-as, Q, r));
      have_true = 0;
      k++;
      (*iters_io)++;
      if (kt > NSL_CHEAP_CAP)
      {
         kt = 0;   /* cannot happen (k <= cap), keep the arrays safe */
      }
      d->tal[kt] = alpha;

      if (jac)
      {
         NSL_TRY(nsl_diag_scale(A, r, z));
      }
      rz_new = hypre_ParKrylovInnerProd(r, zv);
      if (!nsl_finite(rz_new) || !(sgn_m * rz_new > 0.0))
      {
         break;   /* breakdown, or residual (numerically) zero */
      }

      /* predicted convergence: verify with the true residual */
      if (ratio * fabs(rz_new) <= tol2)
      {
         HYPRE_Real tn2;
         HYPRE_Real rrec2;

         NSL_TRY(hypre_ParKrylovCopyVector(b, Q));
         NSL_TRY(hypre_ParKrylovMatvec(NULL, -1.0, A, x, 1.0, Q));
         tn2 = hypre_ParKrylovInnerProd(Q, Q);
         if (!nsl_finite(tn2))
         {
            break;
         }
         if (tn2 <= tol2)
         {
            NSL_TRY(hypre_ParKrylovCopyVector(Q, r));
            rnorm = sqrt(tn2);
            have_true = 1;
            *converged = 1;
            break;
         }
         nver++;
         ratio = tn2 / fabs(rz_new);
         rrec2 = jac ? hypre_ParKrylovInnerProd(r, r) : rz_new;
         if (nver >= NSL_CHEAP_MAXVER || !nsl_finite(rrec2) ||
             !nsl_finite(ratio))
         {
            NSL_TRY(hypre_ParKrylovCopyVector(Q, r));
            rnorm = sqrt(tn2);
            have_true = 1;
            break;
         }
         if (tn2 > 4.0 * rrec2)
         {
            /* recurrence drifted: restart CG from the true residual */
            NSL_TRY(hypre_ParKrylovCopyVector(Q, r));
            if (jac)
            {
               NSL_TRY(nsl_diag_scale(A, r, z));
            }
            rz_new = hypre_ParKrylovInnerProd(r, zv);
            if (!nsl_finite(rz_new) || !(sgn_m * rz_new > 0.0))
            {
               rnorm = sqrt(tn2);
               have_true = 1;
               break;
            }
            ratio = tn2 / fabs(rz_new);
            rz = rz_new;
            s = sqrt(fabs(rz_new));
            if (s < sbest)
            {
               sbest = s;
            }
            hist[k] = sbest;
            NSL_TRY(hypre_ParKrylovCopyVector(zv, P));   /* p = z */
            sigma = 1.0;
            kt = 0;   /* the Lanczos recurrence restarts as well */
            continue;
         }
      }

      /* projected-cost bail-out in the M^{-1}-norm of the residual */
      s = sqrt(fabs(rz_new));
      if (s < sbest)
      {
         sbest = s;
      }
      hist[k] = sbest;
      s_tol = tol / sqrt(ratio);
      if (nsl_cheap_bail(d, k, kt + 1, s, s_tol, iter_cost, cap_total))
      {
         break;   /* finishing is predicted to cost more than the AMG path */
      }

      beta = rz_new / rz;
      if (!nsl_finite(beta))
      {
         break;
      }
      rz = rz_new;
      d->tbe[kt] = beta;
      kt++;

      /* p_new = z_new + beta p_old  <=>  P += z_new / sigma_new */
      sig_new = beta * sigma;
      if (!(sig_new > NSL_SIGMA_LO) || !(sig_new < NSL_SIGMA_HI))
      {
         NSL_TRY(hypre_ParKrylovScaleVector(sigma, P));   /* P <- p */
         sigma = 1.0;
         sig_new = beta;
         if (!(sig_new > 0.0) || !nsl_finite(1.0 / sig_new) ||
             !(sig_new < NSL_SIGMA_HI))
         {
            break;
         }
      }
      NSL_TRY(hypre_ParKrylovAxpy(1.0 / sig_new, zv, P));
      sigma = sig_new;
   }

done:
   if (!have_true)
   {
      NSL_TRY(nsl_true_residual(A, b, x, r, &rnorm));
   }
   *rnorm_io = rnorm;
   return 0;
}

/* ------------------------------------------------------------------ */
/* stage 2a: BoomerAMG-preconditioned CG                                */
/* status: 0 converged, 1 fallback to GMRES requested, 2 iteration limit */
/* ------------------------------------------------------------------ */

static HYPRE_Int
nsl_amg_pcg(nsl_Data *d, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x,
            HYPRE_Real tol, HYPRE_Real *rnorm_io, HYPRE_Int *iters_io,
            HYPRE_Int *status)
{
   void *r  = d->r;
   void *q  = d->q;
   void *pv = d->z;
   void *zv = d->p;
   void *tmp;
   HYPRE_Real r0 = *rnorm_io;
   HYPRE_Real rnorm = r0;
   HYPRE_Real rbest = r0;
   HYPRE_Real rz;
   HYPRE_Real sgn;
   HYPRE_Int k = 0;
   HYPRE_Int nver = 0;
   HYPRE_Int have_true = 1;
   HYPRE_Int reason = 2;

   *status = 1;
   NSL_TRY(hypre_ParKrylovCopyVector(x, d->xb));

   NSL_TRY(nsl_apply_amg(d, A, r, pv));         /* p_0 = z_0 = M^{-1} r_0 */
   rz = hypre_ParKrylovInnerProd(r, pv);
   if (!nsl_finite(rz) || rz == 0.0)
   {
      reason = 1;
      goto done;
   }
   sgn = rz > 0.0 ? 1.0 : -1.0;

   while (*iters_io < d->max_iter)
   {
      HYPRE_Real pq;
      HYPRE_Real alpha;
      HYPRE_Real beta;
      HYPRE_Real rz_new;

      NSL_TRY(hypre_ParKrylovMatvec(NULL, 1.0, A, pv, 0.0, q));
      pq = hypre_ParKrylovInnerProd(pv, q);
      if (!nsl_finite(pq) || !(sgn * pq > 0.0))
      {
         reason = 1;
         break;
      }
      alpha = rz / pq;
      if (!nsl_finite(alpha))
      {
         reason = 1;
         break;
      }

      NSL_TRY(hypre_ParKrylovAxpy(alpha, pv, x));
      NSL_TRY(hypre_ParKrylovAxpy(-alpha, q, r));
      have_true = 0;
      (*iters_io)++;
      k++;
      rnorm = nsl_norm(r);
      if (!nsl_finite(rnorm))
      {
         reason = 1;
         break;
      }

      if (rnorm <= tol)
      {
         HYPRE_Real tn;
         NSL_TRY(hypre_ParKrylovCopyVector(b, q));
         NSL_TRY(hypre_ParKrylovMatvec(NULL, -1.0, A, x, 1.0, q));
         tn = nsl_norm(q);
         if (!nsl_finite(tn))
         {
            reason = 1;
            break;
         }
         if (tn <= tol)
         {
            NSL_TRY(hypre_ParKrylovCopyVector(q, r));
            rnorm = tn;
            have_true = 1;
            reason = 0;
            break;
         }
         if (++nver > 3)
         {
            NSL_TRY(hypre_ParKrylovCopyVector(q, r));
            rnorm = tn;
            have_true = 1;
            reason = 1;
            break;
         }
         if (tn > 4.0 * rnorm)
         {
            /* recurrence drifted: restart from the true residual */
            NSL_TRY(hypre_ParKrylovCopyVector(q, r));
            rnorm = tn;
            have_true = 1;
            if (rnorm < rbest)
            {
               rbest = rnorm;
            }
            NSL_TRY(nsl_apply_amg(d, A, r, zv));
            rz = hypre_ParKrylovInnerProd(r, zv);
            if (!nsl_finite(rz) || !(sgn * rz > 0.0))
            {
               reason = 1;
               break;
            }
            tmp = pv; pv = zv; zv = tmp;
            continue;
         }
         /* small mismatch: keep the recurrence running */
      }

      if (rnorm > 10.0 * rbest)
      {
         reason = 1;   /* erratic residual: non-symmetry dominates */
         break;
      }
      if (rnorm < rbest)
      {
         rbest = rnorm;
      }
      if (k >= NSL_PCG_STALL && rbest > 0.3 * r0)
      {
         reason = 1;   /* stagnation: AMG-CG should do far better */
         break;
      }

      NSL_TRY(nsl_apply_amg(d, A, r, zv));
      rz_new = hypre_ParKrylovInnerProd(r, zv);
      if (!nsl_finite(rz_new) || !(sgn * rz_new > 0.0))
      {
         reason = 1;
         break;
      }
      beta = rz_new / rz;
      if (!nsl_finite(beta))
      {
         reason = 1;
         break;
      }
      rz = rz_new;
      NSL_TRY(hypre_ParKrylovAxpy(beta, pv, zv));
      tmp = pv; pv = zv; zv = tmp;
   }

done:
   if (!have_true)
   {
      NSL_TRY(nsl_true_residual(A, b, x, r, &rnorm));
   }
   if (reason != 0 && nsl_finite(rnorm) && rnorm <= tol)
   {
      reason = 0;
   }
   if (reason != 0 && (!nsl_finite(rnorm) || rnorm > r0))
   {
      /* no gain (or corruption): restore the entry iterate */
      NSL_TRY(hypre_ParKrylovCopyVector(d->xb, x));
      NSL_TRY(nsl_true_residual(A, b, x, r, &rnorm));
      if (!nsl_finite(rnorm))
      {
         return HYPRE_ERROR_GENERIC;
      }
      if (rnorm <= tol)
      {
         reason = 0;
      }
   }
   *rnorm_io = rnorm;
   *status = reason;
   return 0;
}

/* ------------------------------------------------------------------ */
/* stage 2b: GMRES(30) + BoomerAMG, fused CGS with selective reorth.     */
/* ------------------------------------------------------------------ */

static HYPRE_Int
nsl_gmres_phase(nsl_Data *d, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x,
                HYPRE_Real tol, HYPRE_Real rnorm, HYPRE_Int *iters_io)
{
   const HYPRE_Int m = NSL_RESTART;
   const HYPRE_Real tol_in = 0.95 * tol;
   void **V;
   HYPRE_Real *H;
   HYPRE_Int i;
   HYPRE_Int j;
   HYPRE_Int k;

   if (d->V == NULL)
   {
      NSL_TRY(nsl_alloc_gmres(d, b));
   }
   V = d->V;
   H = d->H;

   NSL_TRY(hypre_ParKrylovCopyVector(d->r, V[0]));

   while (*iters_io < d->max_iter)
   {
      if (!nsl_finite(rnorm) || !(rnorm > 0.0))
      {
         return HYPRE_ERROR_GENERIC;
      }
      NSL_TRY(hypre_ParKrylovScaleVector(1.0 / rnorm, V[0]));
      d->g[0] = rnorm;
      for (i = 1; i <= m; i++)
      {
         d->g[i] = 0.0;
      }
      k = 0;

      for (j = 0; j < m && *iters_io < d->max_iter; j++)
      {
         HYPRE_Real hn;
         HYPRE_Real hn2;
         HYPRE_Real sum2 = 0.0;
         HYPRE_Real h1;
         HYPRE_Real h2;
         HYPRE_Real rn;

         /* w = A M^{-1} v_j */
         NSL_TRY(nsl_apply_amg(d, A, V[j], d->z));
         NSL_TRY(hypre_ParKrylovMatvec(NULL, 1.0, A, d->z, 0.0, V[j + 1]));

         /* classical Gram-Schmidt, fused */
         NSL_TRY(hypre_ParKrylovMassInnerProd(V[j + 1], V, j + 1,
                                              NSL_UNROLL, d->coef));
         for (i = 0; i <= j; i++)
         {
            HYPRE_Real c = (HYPRE_Real) d->coef[i];
            if (!nsl_finite(c))
            {
               return HYPRE_ERROR_GENERIC;
            }
            H[i * m + j] = c;
            sum2 += c * c;
            d->coef[i] = -c;
         }
         NSL_TRY(hypre_ParKrylovMassAxpy(d->coef, V, V[j + 1], j + 1,
                                         NSL_UNROLL));
         hn2 = hypre_ParKrylovInnerProd(V[j + 1], V[j + 1]);
         if (!nsl_finite(hn2))
         {
            return HYPRE_ERROR_GENERIC;
         }
         if (hn2 <= 1.0e-6 * sum2)
         {
            /* severe cancellation (||w||^2 ~ hn^2 + sum2): reorthogonalize */
            NSL_TRY(hypre_ParKrylovMassInnerProd(V[j + 1], V, j + 1,
                                                 NSL_UNROLL, d->coef));
            for (i = 0; i <= j; i++)
            {
               HYPRE_Real c = (HYPRE_Real) d->coef[i];
               if (!nsl_finite(c))
               {
                  return HYPRE_ERROR_GENERIC;
               }
               H[i * m + j] += c;
               d->coef[i] = -c;
            }
            NSL_TRY(hypre_ParKrylovMassAxpy(d->coef, V, V[j + 1], j + 1,
                                            NSL_UNROLL));
            hn2 = hypre_ParKrylovInnerProd(V[j + 1], V[j + 1]);
            if (!nsl_finite(hn2))
            {
               return HYPRE_ERROR_GENERIC;
            }
         }
         hn = sqrt(hn2);
         H[(j + 1) * m + j] = hn;
         if (hn > HYPRE_REAL_MIN)
         {
            NSL_TRY(hypre_ParKrylovScaleVector(1.0 / hn, V[j + 1]));
         }

         /* apply stored rotations to the new column */
         for (i = 0; i < j; i++)
         {
            h1 = H[i * m + j];
            h2 = H[(i + 1) * m + j];
            H[i * m + j]       =  d->cs[i] * h1 + d->sn[i] * h2;
            H[(i + 1) * m + j] = -d->sn[i] * h1 + d->cs[i] * h2;
         }
         h1 = H[j * m + j];
         h2 = H[(j + 1) * m + j];
         rn = hypot(h1, h2);
         if (!nsl_finite(rn))
         {
            return HYPRE_ERROR_GENERIC;
         }
         if (rn < HYPRE_REAL_MIN)
         {
            break;   /* Krylov space exhausted */
         }
         d->cs[j] = h1 / rn;
         d->sn[j] = h2 / rn;
         H[j * m + j] = rn;
         H[(j + 1) * m + j] = 0.0;
         d->g[j + 1] = -d->sn[j] * d->g[j];
         d->g[j]     =  d->cs[j] * d->g[j];
         if (!nsl_finite(d->g[j]) || !nsl_finite(d->g[j + 1]))
         {
            return HYPRE_ERROR_GENERIC;
         }

         (*iters_io)++;
         k = j + 1;
         if (fabs(d->g[j + 1]) <= tol_in || hn <= HYPRE_REAL_MIN)
         {
            break;
         }
      }

      if (k == 0)
      {
         return HYPRE_ERROR_CONV;
      }

      /* back substitution */
      for (i = k - 1; i >= 0; i--)
      {
         HYPRE_Real val = d->g[i];
         HYPRE_Real diag = H[i * m + i];
         HYPRE_Int l;
         for (l = i + 1; l < k; l++)
         {
            val -= H[i * m + l] * (HYPRE_Real) d->y[l];
         }
         if (!nsl_finite(val) || !nsl_finite(diag) ||
             fabs(diag) < HYPRE_REAL_MIN)
         {
            return HYPRE_ERROR_GENERIC;
         }
         d->y[i] = val / diag;
         if (!nsl_finite((HYPRE_Real) d->y[i]))
         {
            return HYPRE_ERROR_GENERIC;
         }
      }

      /* x += M^{-1} (V y) */
      NSL_TRY(hypre_ParKrylovClearVector(d->z));
      NSL_TRY(hypre_ParKrylovMassAxpy(d->y, V, d->z, k, NSL_UNROLL));
      NSL_TRY(nsl_apply_amg(d, A, d->z, d->q));
      NSL_TRY(hypre_ParKrylovAxpy(1.0, d->q, x));

      /* true residual for the restart / termination decision */
      NSL_TRY(nsl_true_residual(A, b, x, V[0], &rnorm));
      if (!nsl_finite(rnorm))
      {
         return HYPRE_ERROR_GENERIC;
      }
      if (rnorm <= tol)
      {
         return 0;
      }
   }
   return HYPRE_ERROR_CONV;
}

/* ------------------------------------------------------------------ */
/* solve                                                               */
/* ------------------------------------------------------------------ */

static HYPRE_Int
nsl_Solve(HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x)
{
   nsl_Data *d = (nsl_Data *) solver;
   HYPRE_Real bb;
   HYPRE_Real rr;
   HYPRE_Real xx;
   HYPRE_Real bnorm;
   HYPRE_Real rnorm;
   HYPRE_Real r0;
   HYPRE_Real denom;
   HYPRE_Real tol;
   HYPRE_Real rq = -1.0;
   HYPRE_Real iter_cost;
   HYPRE_Real cap_total;
   HYPRE_Int iters = 0;
   HYPRE_Int jac = 1;
   HYPRE_Int cheap_ok = 1;
   HYPRE_Int q_is_ax = 1;
   HYPRE_Int cap;
   HYPRE_Int status;

   if (d == NULL || A == NULL || b == NULL || x == NULL ||
       !d->base.is_setup || d->matrix != A || d->amg == NULL ||
       d->r == NULL || d->z == NULL || d->p == NULL || d->q == NULL ||
       d->xb == NULL || d->hist == NULL || d->tal == NULL || d->tbe == NULL)
   {
      return HYPRE_ERROR_GENERIC;
   }

   /* q = A x0 (reused by the probe and the rescaling), r = b - q */
   NSL_TRY(hypre_ParKrylovMatvec(NULL, 1.0, A, x, 0.0, d->q));
   NSL_TRY(hypre_ParKrylovCopyVector(b, d->r));
   NSL_TRY(hypre_ParKrylovAxpy(-1.0, d->q, d->r));

   bb = hypre_ParKrylovInnerProd(b, b);
   rr = hypre_ParKrylovInnerProd(d->r, d->r);
   xx = hypre_ParKrylovInnerProd(x, x);
   if (!nsl_finite(bb) || !nsl_finite(rr) || !nsl_finite(xx))
   {
      return HYPRE_ERROR_GENERIC;
   }
   bnorm = sqrt(bb);
   rnorm = sqrt(rr);
   denom = bnorm > 0.0 ? bnorm : rnorm;
   tol = d->rtol * denom;
   if (d->atol > tol)
   {
      tol = d->atol;
   }
   if (!nsl_finite(tol))
   {
      return HYPRE_ERROR_GENERIC;
   }
   if (rnorm == 0.0 || rnorm <= tol)
   {
      return 0;   /* exact initial guess (or zero right-hand side) */
   }
   if (d->max_iter <= 0)
   {
      return HYPRE_ERROR_CONV;
   }

   /* Rayleigh-quotient probe of D^{-1}A with the smooth warm-start vector
      (q = A x0 is already available); cold starts fall back to b */
   if (xx > 0.0)
   {
      HYPRE_Real num;
      NSL_TRY(nsl_diag_scale(A, d->q, d->z));
      num = hypre_ParKrylovInnerProd(x, d->z);
      if (nsl_finite(num) && num > 0.0)
      {
         rq = num / xx;
      }
   }
   else if (bnorm > 0.0)
   {
      HYPRE_Real num;
      NSL_TRY(hypre_ParKrylovMatvec(NULL, 1.0, A, b, 0.0, d->q));
      q_is_ax = 0;
      NSL_TRY(nsl_diag_scale(A, d->q, d->z));
      num = hypre_ParKrylovInnerProd(b, d->z);
      if (nsl_finite(num) && num > 0.0)
      {
         rq = num / bb;
      }
   }

   /* Least-squares rescaling of the warm start: x1 = alpha x0 with
      alpha = (b, A x0)/(A x0, A x0) minimises ||b - alpha A x0||.  For a
      time-stepped diffusion problem this removes the dominant decaying
      mode of the previous solution from the residual at the cost of two
      inner products; the Krylov space generated from r0 does not contain
      the x0 direction, so this is a genuine gain.  Since alpha = 1 is a
      feasible choice the residual can only decrease. */
   if (xx > 0.0 && q_is_ax && bnorm > 0.0)
   {
      HYPRE_Real bq = hypre_ParKrylovInnerProd(b, d->q);
      HYPRE_Real qq = hypre_ParKrylovInnerProd(d->q, d->q);
      if (nsl_finite(bq) && nsl_finite(qq) && qq > 0.0 && bq != 0.0)
      {
         HYPRE_Real alpha = bq / qq;
         HYPRE_Real rr_new = bb - alpha * bq;
         if (nsl_finite(alpha) && nsl_finite(rr_new) && alpha != 1.0 &&
             rr_new < NSL_SCALE_GAIN * rr)
         {
            NSL_TRY(hypre_ParKrylovScaleVector(alpha, x));
            NSL_TRY(hypre_ParKrylovCopyVector(b, d->r));
            NSL_TRY(hypre_ParKrylovAxpy(-alpha, d->q, d->r));
            rr = hypre_ParKrylovInnerProd(d->r, d->r);
            if (!nsl_finite(rr))
            {
               return HYPRE_ERROR_GENERIC;
            }
            rnorm = sqrt(rr);
            xx *= alpha * alpha;
            if (rnorm <= tol)
            {
               /* confirm with an explicitly recomputed residual */
               NSL_TRY(nsl_true_residual(A, b, x, d->r, &rnorm));
               if (!nsl_finite(rnorm))
               {
                  return HYPRE_ERROR_GENERIC;
               }
               if (rnorm <= tol)
               {
                  return 0;
               }
               rr = rnorm * rnorm;
            }
         }
      }
   }

   /* Is the diagonal numerically constant?  Then Jacobi scaling is pure
      overhead: Cauchy-Schwarz defect of (r, D^{-1} r).  z = D^{-1} r is
      needed by stage 1 anyway, so this costs one extra inner product. */
   {
      HYPRE_Real ru;
      HYPRE_Real uu;

      NSL_TRY(nsl_diag_scale(A, d->r, d->z));   /* z = D^{-1} r, kept */
      ru = hypre_ParKrylovInnerProd(d->r, d->z);
      uu = hypre_ParKrylovInnerProd(d->z, d->z);
      if (nsl_finite(ru) && nsl_finite(uu) && uu > 0.0 && ru != 0.0 &&
          rr > 0.0)
      {
         HYPRE_Real dev = 1.0 - (ru * ru) / (rr * uu);
         if (nsl_finite(dev) && dev <= NSL_JAC_DEV_TOL)
         {
            jac = 0;
         }
      }
      else
      {
         cheap_ok = 0;
      }
   }

   /* cost model: stage-1 iteration versus the whole multigrid path */
   iter_cost = NSL_COST_MATVEC + 2.0 * NSL_COST_IP + 3.0 * NSL_COST_AXPY +
               (jac ? NSL_COST_DIAG : 0.0);
   cap_total = NSL_CAP_FACTOR * nsl_amg_cost(log(rnorm / tol));
   r0 = rnorm;
   cap = NSL_CHEAP_CAP;
   if (cap > d->max_iter - NSL_CHEAP_RESERVE)
   {
      cap = d->max_iter - NSL_CHEAP_RESERVE;
   }

   if (cheap_ok && cap >= NSL_CHEAP_MIN_CAP &&
       nsl_predict_cheap(rq, rnorm, tol, iter_cost, cap))
   {
      HYPRE_Int conv = 0;

      NSL_TRY(hypre_ParKrylovCopyVector(x, d->xb));
      NSL_TRY(nsl_cheap_phase(d, A, b, x, tol, cap, jac, jac, iter_cost,
                              cap_total, &rnorm, &iters, &conv));
      /* d->r now holds the true residual b - A x with norm rnorm */
      if (nsl_finite(rnorm) && rnorm <= tol)
      {
         return 0;
      }
      if (!nsl_finite(rnorm) || rnorm >= r0)
      {
         /* no gain (or corruption): restore the entry iterate */
         NSL_TRY(hypre_ParKrylovCopyVector(d->xb, x));
         NSL_TRY(nsl_true_residual(A, b, x, d->r, &rnorm));
         if (!nsl_finite(rnorm))
         {
            return HYPRE_ERROR_GENERIC;
         }
         if (rnorm <= tol)
         {
            return 0;
         }
      }
      if (iters >= d->max_iter)
      {
         return HYPRE_ERROR_CONV;
      }
   }

   /* stage 2: build the AMG hierarchy now that it is known to be needed */
   if (!d->amg_ready)
   {
      NSL_TRY(HYPRE_BoomerAMGSetup(d->amg, (HYPRE_ParCSRMatrix) A,
                                   (HYPRE_ParVector) b, (HYPRE_ParVector) x));
      d->amg_ready = 1;
   }

   NSL_TRY(nsl_amg_pcg(d, A, b, x, tol, &rnorm, &iters, &status));
   if (status == 0)
   {
      return 0;
   }
   if (status == 2 || iters >= d->max_iter)
   {
      return HYPRE_ERROR_CONV;
   }

   /* CG lost definiteness / stagnated: robust GMRES on the same AMG */
   return nsl_gmres_phase(d, A, b, x, tol, rnorm, &iters);
}

/* ------------------------------------------------------------------ */
/* destroy                                                             */
/* ------------------------------------------------------------------ */

static HYPRE_Int
nsl_Destroy(HYPRE_Solver solver)
{
   nsl_Data *d = (nsl_Data *) solver;
   HYPRE_Int err;
   HYPRE_Int e;

   if (d == NULL)
   {
      return 0;
   }
   err = nsl_release_all(d);
   e = hypre_ParKrylovFree(d);
   return err != 0 ? err : e;
}
