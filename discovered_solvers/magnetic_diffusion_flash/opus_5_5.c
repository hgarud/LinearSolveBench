#include "HYPRE_parcsr_ls.h"

#define NSL_RESTART 40
#define NSL_LCAP 128
#define NSL_PROBE 5
#define NSL_CHK 5
#define NSL_C_SETUP 25.0
#define NSL_C_CYC 4.0
#define NSL_RHO_AMG 0.2
#define NSL_HUGE 1.0e30
#define NSL_BISECT 64

enum
{
   NSL_STATUS_CONVERGED = 0,
   NSL_STATUS_SWITCH = 1,
   NSL_STATUS_MAXIT = 2
};

enum
{
   NSL_JST_CONT = 0,
   NSL_JST_CONV = 1,
   NSL_JST_BREAK = 2,
   NSL_JST_MAXIT = 3
};

enum
{
   NSL_MODE_ZERO = 0,
   NSL_MODE_DONE = 1,
   NSL_MODE_JAC = 2,
   NSL_MODE_AMG = 3,
   NSL_MODE_NOITER = 4
};

typedef struct
{
   hypre_Solver base;

   HYPRE_Solver amg;
   HYPRE_Int amg_ready;
   HYPRE_Int amg_failed;

   HYPRE_Real relative_tolerance;
   HYPRE_Real absolute_tolerance;
   HYPRE_Int maximum_iterations;

   HYPRE_Matrix matrix;

   void **rv;
   void *r;
   void *z;
   void *p;
   void *q;
   void *e;

   void **V;
   void **Z;
   HYPRE_Real *H;
   HYPRE_Real *cs;
   HYPRE_Real *sn;
   HYPRE_Real *g;
   HYPRE_Real *y;
   HYPRE_Real *h;

   HYPRE_Int have_state;
   HYPRE_Int mode;
   HYPRE_Int jst;
   HYPRE_Int iter0;
   HYPRE_Int e_used;
   void *b_ptr;
   void *x_ptr;
   HYPRE_Real fp_bb;
   HYPRE_Real fp_xx;
   HYPRE_Real fp_bx;
   HYPRE_Real bnorm;
   HYPRE_Real tol;

   HYPRE_Real rz;
   HYPRE_Real rnorm;
   HYPRE_Int jk;
   HYPRE_Real la[NSL_LCAP];
   HYPRE_Real lb[NSL_LCAP];
   HYPRE_Real rhist[16];
} nsl_Data;

static HYPRE_Int nsl_Setup(HYPRE_Solver solver, HYPRE_Matrix A,
                           HYPRE_Vector b, HYPRE_Vector x);
static HYPRE_Int nsl_Solve(HYPRE_Solver solver, HYPRE_Matrix A,
                           HYPRE_Vector b, HYPRE_Vector x);
static HYPRE_Int nsl_Destroy(HYPRE_Solver solver);

#define NSL_CHECK(call)                  \
   do                                    \
   {                                     \
      HYPRE_Int nsl_e_ = (call);         \
      if (nsl_e_ != 0)                   \
      {                                  \
         return nsl_e_;                  \
      }                                  \
   } while (0)

static HYPRE_Int
nsl_finite(HYPRE_Real v)
{
   return isfinite((double) v) ? 1 : 0;
}

static HYPRE_Real
nsl_abs(HYPRE_Real v)
{
   return v < 0.0 ? -v : v;
}

static void
nsl_swap_pz(nsl_Data *d)
{
   void *t = d->p;
   d->p = d->z;
   d->z = t;
}

static HYPRE_Real
nsl_pair_norm(HYPRE_Real a, HYPRE_Real b)
{
   HYPRE_Real x = nsl_abs(a);
   HYPRE_Real y = nsl_abs(b);
   HYPRE_Real t;
   if (y > x)
   {
      t = x;
      x = y;
      y = t;
   }
   if (x == 0.0)
   {
      return 0.0;
   }
   t = y / x;
   return x * sqrt(1.0 + t * t);
}

static HYPRE_Int
nsl_norm(void *v, HYPRE_Real *out)
{
   HYPRE_Real s = hypre_ParKrylovInnerProd(v, v);
   if (!nsl_finite(s) || s < 0.0)
   {
      return HYPRE_ERROR_GENERIC;
   }
   *out = sqrt(s);
   return nsl_finite(*out) ? 0 : HYPRE_ERROR_GENERIC;
}

static HYPRE_Int
nsl_destroy_array(void ***arr_ptr, HYPRE_Int n)
{
   HYPRE_Int error = 0;
   HYPRE_Int i;
   HYPRE_Int e;
   void **arr;

   if (arr_ptr == NULL || *arr_ptr == NULL)
   {
      return 0;
   }
   arr = *arr_ptr;
   for (i = n - 1; i >= 1; i--)
   {
      if (arr[i] != NULL)
      {
         e = hypre_ParKrylovDestroyVector(arr[i]);
         if (error == 0 && e != 0)
         {
            error = e;
         }
         arr[i] = NULL;
      }
   }
   if (arr[0] != NULL)
   {
      e = hypre_ParKrylovDestroyVector(arr[0]);
      if (error == 0 && e != 0)
      {
         error = e;
      }
      arr[0] = NULL;
   }
   e = hypre_ParKrylovFree(arr);
   if (error == 0 && e != 0)
   {
      error = e;
   }
   *arr_ptr = NULL;
   return error;
}

static HYPRE_Int
nsl_free_host(void **ptr)
{
   HYPRE_Int e = 0;
   if (*ptr != NULL)
   {
      e = hypre_ParKrylovFree(*ptr);
      *ptr = NULL;
   }
   return e;
}

static HYPRE_Int
nsl_free_gmres(nsl_Data *d)
{
   HYPRE_Int error = 0;
   HYPRE_Int e;

   e = nsl_destroy_array(&d->V, NSL_RESTART + 1);
   if (error == 0 && e != 0) error = e;
   e = nsl_destroy_array(&d->Z, NSL_RESTART);
   if (error == 0 && e != 0) error = e;
   e = nsl_free_host((void **) &d->H);
   if (error == 0 && e != 0) error = e;
   e = nsl_free_host((void **) &d->cs);
   if (error == 0 && e != 0) error = e;
   e = nsl_free_host((void **) &d->sn);
   if (error == 0 && e != 0) error = e;
   e = nsl_free_host((void **) &d->g);
   if (error == 0 && e != 0) error = e;
   e = nsl_free_host((void **) &d->y);
   if (error == 0 && e != 0) error = e;
   e = nsl_free_host((void **) &d->h);
   if (error == 0 && e != 0) error = e;
   return error;
}

static HYPRE_Int
nsl_free_vec(void **v)
{
   HYPRE_Int e = 0;
   if (*v != NULL)
   {
      e = hypre_ParKrylovDestroyVector(*v);
      *v = NULL;
   }
   return e;
}

static HYPRE_Int
nsl_free_work(nsl_Data *d)
{
   HYPRE_Int error = 0;
   HYPRE_Int e;
   e = nsl_destroy_array(&d->rv, 1);
   if (error == 0 && e != 0) error = e;
   d->r = NULL;
   e = nsl_free_vec(&d->z);
   if (error == 0 && e != 0) error = e;
   e = nsl_free_vec(&d->p);
   if (error == 0 && e != 0) error = e;
   e = nsl_free_vec(&d->q);
   if (error == 0 && e != 0) error = e;
   e = nsl_free_vec(&d->e);
   if (error == 0 && e != 0) error = e;
   return error;
}

static HYPRE_Int
nsl_free_amg(nsl_Data *d)
{
   HYPRE_Int e = 0;
   if (d->amg != NULL)
   {
      e = HYPRE_BoomerAMGDestroy(d->amg);
      d->amg = NULL;
   }
   d->amg_ready = 0;
   return e;
}

static HYPRE_Int
nsl_free_all(nsl_Data *d)
{
   HYPRE_Int error = 0;
   HYPRE_Int e;
   e = nsl_free_gmres(d);
   if (error == 0 && e != 0) error = e;
   e = nsl_free_work(d);
   if (error == 0 && e != 0) error = e;
   e = nsl_free_amg(d);
   if (error == 0 && e != 0) error = e;
   d->amg_failed = 0;
   d->have_state = 0;
   d->matrix = NULL;
   d->base.is_setup = 0;
   return error;
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
   d->base.setup = nsl_Setup;
   d->base.solve = nsl_Solve;
   d->base.destroy = nsl_Destroy;
   d->base.is_setup = 0;
   d->relative_tolerance = relative_tolerance;
   d->absolute_tolerance = absolute_tolerance;
   d->maximum_iterations = maximum_iterations;
   *solver = (HYPRE_Solver) d;
   return 0;
}

static HYPRE_Int
nsl_configure_amg(HYPRE_Solver amg)
{
   NSL_CHECK(HYPRE_BoomerAMGSetCoarsenType(amg, 10));
   NSL_CHECK(HYPRE_BoomerAMGSetInterpType(amg, 6));
   NSL_CHECK(HYPRE_BoomerAMGSetPMaxElmts(amg, 4));
   NSL_CHECK(HYPRE_BoomerAMGSetStrongThreshold(amg, 0.25));
   NSL_CHECK(HYPRE_BoomerAMGSetAggNumLevels(amg, 1));
   NSL_CHECK(HYPRE_BoomerAMGSetAggInterpType(amg, 4));
   NSL_CHECK(HYPRE_BoomerAMGSetAggPMaxElmts(amg, 4));
   NSL_CHECK(HYPRE_BoomerAMGSetMaxLevels(amg, 25));
   NSL_CHECK(HYPRE_BoomerAMGSetMaxCoarseSize(amg, 48));
   NSL_CHECK(HYPRE_BoomerAMGSetRelaxType(amg, 3));
   NSL_CHECK(HYPRE_BoomerAMGSetCycleRelaxType(amg, 3, 1));
   NSL_CHECK(HYPRE_BoomerAMGSetCycleRelaxType(amg, 4, 2));
   NSL_CHECK(HYPRE_BoomerAMGSetCycleRelaxType(amg, 9, 3));
   NSL_CHECK(HYPRE_BoomerAMGSetNumSweeps(amg, 1));
   NSL_CHECK(HYPRE_BoomerAMGSetMaxIter(amg, 1));
   NSL_CHECK(HYPRE_BoomerAMGSetTol(amg, 0.0));
   return 0;
}

static HYPRE_Int
nsl_amg_build(nsl_Data *d, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x)
{
   HYPRE_Int e;
   if (d->amg_ready || d->amg_failed)
   {
      return 0;
   }
   e = HYPRE_BoomerAMGCreate(&d->amg);
   if (e != 0)
   {
      d->amg = NULL;
      d->amg_failed = 1;
      return 0;
   }
   e = nsl_configure_amg(d->amg);
   if (e == 0)
   {
      e = HYPRE_BoomerAMGSetup(d->amg, (HYPRE_ParCSRMatrix) A,
                               (HYPRE_ParVector) b, (HYPRE_ParVector) x);
   }
   if (e != 0)
   {
      HYPRE_BoomerAMGDestroy(d->amg);
      d->amg = NULL;
      d->amg_failed = 1;
      return 0;
   }
   d->amg_ready = 1;
   return 0;
}

static HYPRE_Int
nsl_prec(nsl_Data *d, HYPRE_Matrix A, void *in, void *out)
{
   if (d->amg_ready)
   {
      NSL_CHECK(hypre_ParKrylovClearVector(out));
      return HYPRE_BoomerAMGSolve(d->amg, (HYPRE_ParCSRMatrix) A,
                                  (HYPRE_ParVector) in, (HYPRE_ParVector) out);
   }
   return HYPRE_ParCSRDiagScale(NULL, (HYPRE_ParCSRMatrix) A,
                                (HYPRE_ParVector) in, (HYPRE_ParVector) out);
}

static HYPRE_Int
nsl_true_residual(HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x,
                  void *r, HYPRE_Real *rnorm)
{
   NSL_CHECK(hypre_ParKrylovCopyVector(b, r));
   NSL_CHECK(hypre_ParKrylovMatvec(NULL, -1.0, A, x, 1.0, r));
   return nsl_norm(r, rnorm);
}

/* ---------------- Jacobi-PCG with Lanczos spectral tracking ------------- */

static HYPRE_Int
nsl_jac_init(nsl_Data *d, HYPRE_Matrix A, HYPRE_Int *st)
{
   NSL_CHECK(HYPRE_ParCSRDiagScale(NULL, (HYPRE_ParCSRMatrix) A,
                                   (HYPRE_ParVector) d->r,
                                   (HYPRE_ParVector) d->z));
   d->rz = hypre_ParKrylovInnerProd(d->r, d->z);
   d->jk = 0;
   d->rhist[0] = d->rnorm;
   if (!nsl_finite(d->rz) || d->rz <= 0.0)
   {
      *st = NSL_JST_BREAK;
      return 0;
   }
   nsl_swap_pz(d);
   *st = NSL_JST_CONT;
   return 0;
}

static HYPRE_Int
nsl_jac_steps(nsl_Data *d, HYPRE_Matrix A, void *xt, HYPRE_Int nmax,
              HYPRE_Int *iter, HYPRE_Int *st)
{
   HYPRE_Int s;
   HYPRE_Real pq, alpha, rr, rzn, beta;

   for (s = 0; s < nmax; s++)
   {
      if (*iter >= d->maximum_iterations)
      {
         *st = NSL_JST_MAXIT;
         return 0;
      }
      NSL_CHECK(hypre_ParKrylovMatvec(NULL, 1.0, A, d->p, 0.0, d->q));
      pq = hypre_ParKrylovInnerProd(d->p, d->q);
      if (!nsl_finite(pq) || pq <= 0.0)
      {
         *st = NSL_JST_BREAK;
         return 0;
      }
      alpha = d->rz / pq;
      if (!nsl_finite(alpha) || alpha <= 0.0)
      {
         *st = NSL_JST_BREAK;
         return 0;
      }
      NSL_CHECK(hypre_ParKrylovAxpy(alpha, d->p, xt));
      NSL_CHECK(hypre_ParKrylovAxpy(-alpha, d->q, d->r));
      (*iter)++;
      if (d->jk < NSL_LCAP)
      {
         d->la[d->jk] = alpha;
      }
      d->jk++;

      NSL_CHECK(HYPRE_ParCSRDiagScale(NULL, (HYPRE_ParCSRMatrix) A,
                                      (HYPRE_ParVector) d->r,
                                      (HYPRE_ParVector) d->z));
      rr = 0.0;
      rzn = 0.0;
      NSL_CHECK(hypre_ParKrylovMassDotpTwo(d->r, d->z, d->rv, 1, 0,
                                           &rr, &rzn));
      if (!nsl_finite(rr) || rr < 0.0)
      {
         d->rnorm = rr;
         *st = NSL_JST_BREAK;
         return 0;
      }
      d->rnorm = sqrt(rr);
      d->rhist[d->jk & 15] = d->rnorm;
      if (d->rnorm <= d->tol)
      {
         *st = NSL_JST_CONV;
         return 0;
      }
      if (!nsl_finite(rzn) || rzn <= 0.0)
      {
         *st = NSL_JST_BREAK;
         return 0;
      }
      beta = rzn / d->rz;
      if (d->jk - 1 < NSL_LCAP)
      {
         d->lb[d->jk - 1] = beta;
      }
      d->rz = rzn;
      NSL_CHECK(hypre_ParKrylovAxpy(beta, d->p, d->z));
      nsl_swap_pz(d);
   }
   *st = NSL_JST_CONT;
   return 0;
}

static HYPRE_Int
nsl_sturm(const HYPRE_Real *dg, const HYPRE_Real *of, HYPRE_Int m,
          HYPRE_Real x)
{
   HYPRE_Int i, c = 0;
   HYPRE_Real q = dg[0] - x;
   if (q < 0.0) c++;
   for (i = 1; i < m; i++)
   {
      if (nsl_abs(q) < 1.0e-300)
      {
         q = (q < 0.0) ? -1.0e-300 : 1.0e-300;
      }
      q = dg[i] - x - of[i - 1] * of[i - 1] / q;
      if (q < 0.0) c++;
   }
   return c;
}

static HYPRE_Int
nsl_ritz(const nsl_Data *d, HYPRE_Int m, HYPRE_Real *lmin, HYPRE_Real *lmax)
{
   HYPRE_Real dg[NSL_LCAP];
   HYPRE_Real of[NSL_LCAP];
   HYPRE_Real lo = 0.0, hi = 0.0, rad, a, b, mid, pad;
   HYPRE_Int j, it;

   if (m < 1 || m > NSL_LCAP)
   {
      return 1;
   }
   for (j = 0; j < m; j++)
   {
      HYPRE_Real al = d->la[j];
      if (!nsl_finite(al) || al <= 0.0)
      {
         return 1;
      }
      dg[j] = 1.0 / al;
      if (j > 0)
      {
         dg[j] += d->lb[j - 1] / d->la[j - 1];
      }
      if (j < m - 1)
      {
         HYPRE_Real be = d->lb[j];
         if (!nsl_finite(be) || be <= 0.0)
         {
            return 1;
         }
         of[j] = sqrt(be) / al;
      }
   }
   for (j = 0; j < m; j++)
   {
      rad = 0.0;
      if (j > 0) rad += nsl_abs(of[j - 1]);
      if (j < m - 1) rad += nsl_abs(of[j]);
      if (j == 0 || dg[j] - rad < lo) lo = dg[j] - rad;
      if (j == 0 || dg[j] + rad > hi) hi = dg[j] + rad;
   }
   if (!nsl_finite(lo) || !nsl_finite(hi))
   {
      return 1;
   }
   pad = 1.0e-12 * (nsl_abs(lo) + nsl_abs(hi)) + 1.0e-300;
   lo -= pad;
   hi += pad;

   a = lo;
   b = hi;
   for (it = 0; it < NSL_BISECT; it++)
   {
      mid = 0.5 * (a + b);
      if (nsl_sturm(dg, of, m, mid) >= 1) b = mid;
      else a = mid;
   }
   *lmin = b;

   a = lo;
   b = hi;
   for (it = 0; it < NSL_BISECT; it++)
   {
      mid = 0.5 * (a + b);
      if (nsl_sturm(dg, of, m, mid) >= m) b = mid;
      else a = mid;
   }
   *lmax = b;
   return 0;
}

static HYPRE_Real
nsl_jac_pred(const nsl_Data *d, HYPRE_Real lmin, HYPRE_Real lmax,
             HYPRE_Real eps)
{
   HYPRE_Real kap, sk, itk, itr, rho, rn, ro;
   HYPRE_Int w;

   if (!(lmin > 0.0) || !nsl_finite(lmax) || lmax < lmin)
   {
      return NSL_HUGE;
   }
   kap = lmax / lmin;
   sk = sqrt(kap);
   if (sk <= 1.0 + 1.0e-10)
   {
      itk = 1.0;
   }
   else
   {
      itk = log(2.0 / eps) / log((sk + 1.0) / (sk - 1.0));
   }
   if (!nsl_finite(itk))
   {
      itk = NSL_HUGE;
   }
   w = d->jk < 8 ? d->jk : 8;
   if (w < 1)
   {
      return itk;
   }
   rn = d->rhist[d->jk & 15];
   ro = d->rhist[(d->jk - w) & 15];
   if (!(ro > 0.0) || !nsl_finite(rn) || !(rn > 0.0))
   {
      return itk;
   }
   rho = pow(rn / ro, 1.0 / (HYPRE_Real) w);
   if (!nsl_finite(rho) || !(rho < 1.0))
   {
      itr = NSL_HUGE;
   }
   else
   {
      itr = log(eps) / log(rho);
   }
   return itk > itr ? itk : itr;
}

static HYPRE_Real
nsl_amg_cost(const nsl_Data *d, HYPRE_Real eps)
{
   HYPRE_Real its = log(eps) / log(NSL_RHO_AMG);
   if (!nsl_finite(its) || its < 1.0)
   {
      its = 1.0;
   }
   return (d->amg_ready ? 0.0 : NSL_C_SETUP) + NSL_C_CYC * (its + 1.0);
}

static HYPRE_Int
nsl_should_switch(const nsl_Data *d)
{
   HYPRE_Real lmin, lmax, eps;
   HYPRE_Int m;
   if (d->amg_failed || d->jk < 2 || !(d->rnorm > d->tol))
   {
      return 0;
   }
   m = d->jk < NSL_LCAP ? d->jk : NSL_LCAP;
   eps = d->tol / d->rnorm;
   if (nsl_ritz(d, m, &lmin, &lmax) != 0)
   {
      return 1;
   }
   return nsl_jac_pred(d, lmin, lmax, eps) > nsl_amg_cost(d, eps);
}

static HYPRE_Int
nsl_probe_prefers_jacobi(const nsl_Data *d)
{
   HYPRE_Real lmin, lmax, lmin1, lmax1, delta, eps;
   HYPRE_Int m;

   if (d->amg_failed || !(d->rnorm > d->tol))
   {
      return 1;
   }
   m = d->jk < NSL_LCAP ? d->jk : NSL_LCAP;
   if (m < 3)
   {
      return 0;
   }
   if (nsl_ritz(d, m, &lmin, &lmax) != 0 ||
       nsl_ritz(d, m - 1, &lmin1, &lmax1) != 0)
   {
      return 0;
   }
   if (!(lmin1 > 0.0) || !(lmin > 0.0))
   {
      return 0;
   }
   delta = (lmin1 - lmin) / lmin1;
   if (!(delta < 0.05))
   {
      return 0;
   }
   eps = d->tol / d->rnorm;
   return nsl_jac_pred(d, lmin, lmax, eps) <= 0.8 * nsl_amg_cost(d, eps);
}

static HYPRE_Int
nsl_start(nsl_Data *d, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x)
{
   HYPRE_Real bb, xx, bx, r0, tol;
   HYPRE_Int st;

   d->have_state = 0;
   bb = hypre_ParKrylovInnerProd(b, b);
   xx = hypre_ParKrylovInnerProd(x, x);
   bx = hypre_ParKrylovInnerProd(b, x);
   if (!nsl_finite(bb) || !nsl_finite(xx) || !nsl_finite(bx) || bb < 0.0)
   {
      return HYPRE_ERROR_GENERIC;
   }
   d->b_ptr = (void *) b;
   d->x_ptr = (void *) x;
   d->fp_bb = bb;
   d->fp_xx = xx;
   d->fp_bx = bx;
   d->iter0 = 0;
   d->e_used = 0;
   d->jst = NSL_JST_CONT;
   d->jk = 0;
   d->bnorm = sqrt(bb);

   if (d->bnorm == 0.0)
   {
      d->mode = NSL_MODE_ZERO;
      d->have_state = 1;
      return 0;
   }
   tol = d->relative_tolerance * d->bnorm;
   if (d->absolute_tolerance > tol)
   {
      tol = d->absolute_tolerance;
   }
   if (!nsl_finite(tol))
   {
      return HYPRE_ERROR_GENERIC;
   }
   d->tol = tol;

   NSL_CHECK(nsl_true_residual(A, b, x, d->r, &d->rnorm));
   if (d->rnorm <= tol)
   {
      d->mode = NSL_MODE_DONE;
      d->have_state = 1;
      return 0;
   }
   if (d->maximum_iterations <= 0)
   {
      d->mode = NSL_MODE_NOITER;
      d->have_state = 1;
      return 0;
   }
   r0 = d->rnorm;

   NSL_CHECK(hypre_ParKrylovClearVector(d->e));
   NSL_CHECK(nsl_jac_init(d, A, &st));
   if (st == NSL_JST_BREAK)
   {
      d->mode = NSL_MODE_AMG;
      d->jst = st;
      d->have_state = 1;
      return 0;
   }
   NSL_CHECK(nsl_jac_steps(d, A, d->e, NSL_PROBE, &d->iter0, &st));
   d->e_used = 1;
   d->jst = st;
   if (st == NSL_JST_BREAK)
   {
      d->mode = NSL_MODE_AMG;
      if (!(nsl_finite(d->rnorm) && d->rnorm < r0))
      {
         d->e_used = 0;
      }
   }
   else if (st == NSL_JST_CONV || st == NSL_JST_MAXIT)
   {
      d->mode = NSL_MODE_JAC;
   }
   else
   {
      d->mode = nsl_probe_prefers_jacobi(d) ? NSL_MODE_JAC : NSL_MODE_AMG;
   }
   d->have_state = 1;
   return 0;
}

static HYPRE_Int
nsl_Setup(HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x)
{
   nsl_Data *d = (nsl_Data *) solver;
   HYPRE_Int error;

   if (d == NULL || A == NULL || b == NULL || x == NULL)
   {
      return HYPRE_ERROR_GENERIC;
   }
   error = nsl_free_all(d);
   if (error != 0)
   {
      return error;
   }

   d->rv = (void **) hypre_ParKrylovCreateVectorArray(1, b);
   if (d->rv != NULL)
   {
      d->r = d->rv[0];
   }
   d->z = hypre_ParKrylovCreateVector(b);
   d->p = hypre_ParKrylovCreateVector(b);
   d->q = hypre_ParKrylovCreateVector(b);
   d->e = hypre_ParKrylovCreateVector(b);
   if (d->rv == NULL || d->r == NULL || d->z == NULL || d->p == NULL ||
       d->q == NULL || d->e == NULL)
   {
      error = HYPRE_ERROR_GENERIC;
      goto failure;
   }

   error = HYPRE_ParCSRDiagScaleSetup(NULL, (HYPRE_ParCSRMatrix) A,
                                      (HYPRE_ParVector) b,
                                      (HYPRE_ParVector) x);
   if (error != 0)
   {
      goto failure;
   }
   d->matrix = A;

   error = nsl_start(d, A, b, x);
   if (error != 0)
   {
      goto failure;
   }
   if (d->mode == NSL_MODE_AMG)
   {
      error = nsl_amg_build(d, A, b, x);
      if (error != 0)
      {
         goto failure;
      }
   }

   d->base.is_setup = 1;
   return 0;

failure:
   nsl_free_all(d);
   return error != 0 ? error : HYPRE_ERROR_GENERIC;
}

static HYPRE_Int
nsl_pcg(nsl_Data *d, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x,
        HYPRE_Real tol, HYPRE_Real rnorm, HYPRE_Int *iter, HYPRE_Int *status)
{
   HYPRE_Real rz, rz_new, pq, alpha, beta, rr, best;
   HYPRE_Int stall;
   HYPRE_Int restarts = 0;

   *status = NSL_STATUS_MAXIT;

restart:
   NSL_CHECK(nsl_prec(d, A, d->r, d->z));
   rz = hypre_ParKrylovInnerProd(d->r, d->z);
   if (!nsl_finite(rz) || rz <= 0.0)
   {
      *status = NSL_STATUS_SWITCH;
      return 0;
   }
   nsl_swap_pz(d);
   best = rnorm;
   stall = 0;

   while (*iter < d->maximum_iterations)
   {
      NSL_CHECK(hypre_ParKrylovMatvec(NULL, 1.0, A, d->p, 0.0, d->q));
      pq = hypre_ParKrylovInnerProd(d->p, d->q);
      if (!nsl_finite(pq) || pq <= 0.0)
      {
         *status = NSL_STATUS_SWITCH;
         return 0;
      }
      alpha = rz / pq;
      if (!nsl_finite(alpha))
      {
         *status = NSL_STATUS_SWITCH;
         return 0;
      }
      NSL_CHECK(hypre_ParKrylovAxpy(alpha, d->p, x));
      NSL_CHECK(hypre_ParKrylovAxpy(-alpha, d->q, d->r));
      (*iter)++;

      rr = hypre_ParKrylovInnerProd(d->r, d->r);
      if (!nsl_finite(rr) || rr < 0.0)
      {
         *status = NSL_STATUS_SWITCH;
         return 0;
      }
      rnorm = sqrt(rr);

      if (rnorm <= tol)
      {
         NSL_CHECK(nsl_true_residual(A, b, x, d->r, &rnorm));
         if (rnorm <= tol)
         {
            *status = NSL_STATUS_CONVERGED;
            return 0;
         }
         restarts++;
         if (restarts > 3 || *iter >= d->maximum_iterations)
         {
            *status = (*iter >= d->maximum_iterations) ?
                      NSL_STATUS_MAXIT : NSL_STATUS_SWITCH;
            return 0;
         }
         goto restart;
      }

      if (rnorm < best)
      {
         best = rnorm;
         stall = 0;
      }
      else
      {
         stall++;
         if (stall >= 30 || rnorm > 1.0e6 * best)
         {
            *status = NSL_STATUS_SWITCH;
            return 0;
         }
      }

      NSL_CHECK(nsl_prec(d, A, d->r, d->z));
      rz_new = hypre_ParKrylovInnerProd(d->r, d->z);
      if (!nsl_finite(rz_new) || rz_new <= 0.0)
      {
         *status = NSL_STATUS_SWITCH;
         return 0;
      }
      beta = rz_new / rz;
      rz = rz_new;
      NSL_CHECK(hypre_ParKrylovAxpy(beta, d->p, d->z));
      nsl_swap_pz(d);
   }

   NSL_CHECK(nsl_true_residual(A, b, x, d->r, &rnorm));
   *status = (rnorm <= tol) ? NSL_STATUS_CONVERGED : NSL_STATUS_MAXIT;
   return 0;
}

static HYPRE_Int
nsl_alloc_gmres(nsl_Data *d, HYPRE_Vector sample)
{
   if (d->V != NULL)
   {
      return 0;
   }
   d->V = (void **) hypre_ParKrylovCreateVectorArray(NSL_RESTART + 1, sample);
   d->Z = (void **) hypre_ParKrylovCreateVectorArray(NSL_RESTART, sample);
   d->H = (HYPRE_Real *) hypre_ParKrylovCAlloc(
      (size_t) (NSL_RESTART + 1) * (size_t) NSL_RESTART, sizeof(HYPRE_Real),
      HYPRE_MEMORY_HOST);
   d->cs = (HYPRE_Real *) hypre_ParKrylovCAlloc(NSL_RESTART, sizeof(HYPRE_Real),
                                                HYPRE_MEMORY_HOST);
   d->sn = (HYPRE_Real *) hypre_ParKrylovCAlloc(NSL_RESTART, sizeof(HYPRE_Real),
                                                HYPRE_MEMORY_HOST);
   d->g = (HYPRE_Real *) hypre_ParKrylovCAlloc(NSL_RESTART + 1,
                                               sizeof(HYPRE_Real),
                                               HYPRE_MEMORY_HOST);
   d->y = (HYPRE_Real *) hypre_ParKrylovCAlloc(NSL_RESTART + 1,
                                               sizeof(HYPRE_Real),
                                               HYPRE_MEMORY_HOST);
   d->h = (HYPRE_Real *) hypre_ParKrylovCAlloc(NSL_RESTART + 1,
                                               sizeof(HYPRE_Real),
                                               HYPRE_MEMORY_HOST);
   if (d->V == NULL || d->Z == NULL || d->H == NULL || d->cs == NULL ||
       d->sn == NULL || d->g == NULL || d->y == NULL || d->h == NULL)
   {
      nsl_free_gmres(d);
      return HYPRE_ERROR_GENERIC;
   }
   return 0;
}

#define NSL_H(i, j) d->H[(size_t) (i) * (size_t) NSL_RESTART + (size_t) (j)]

static HYPRE_Int
nsl_gmres(nsl_Data *d, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x,
          HYPRE_Real tol, HYPRE_Int *iter)
{
   HYPRE_Real beta;
   HYPRE_Int i, j, k, pass;

   NSL_CHECK(nsl_alloc_gmres(d, b));

   for (;;)
   {
      NSL_CHECK(nsl_true_residual(A, b, x, d->V[0], &beta));
      if (beta <= tol)
      {
         return 0;
      }
      if (*iter >= d->maximum_iterations)
      {
         return HYPRE_ERROR_CONV;
      }
      NSL_CHECK(hypre_ParKrylovScaleVector(1.0 / beta, d->V[0]));
      d->g[0] = beta;
      k = 0;

      for (j = 0; j < NSL_RESTART && *iter < d->maximum_iterations; j++)
      {
         HYPRE_Real nrm, diag, sub, rn, res;

         NSL_CHECK(nsl_prec(d, A, d->V[j], d->Z[j]));
         NSL_CHECK(hypre_ParKrylovMatvec(NULL, 1.0, A, d->Z[j], 0.0,
                                         d->V[j + 1]));
         for (i = 0; i <= j; i++)
         {
            NSL_H(i, j) = 0.0;
         }
         for (pass = 0; pass < 2; pass++)
         {
            NSL_CHECK(hypre_ParKrylovMassInnerProd(d->V[j + 1], d->V, j + 1,
                                                   0, d->h));
            for (i = 0; i <= j; i++)
            {
               if (!nsl_finite(d->h[i]))
               {
                  return HYPRE_ERROR_GENERIC;
               }
               NSL_H(i, j) += d->h[i];
               d->h[i] = -d->h[i];
            }
            NSL_CHECK(hypre_ParKrylovMassAxpy(d->h, d->V, d->V[j + 1],
                                              j + 1, 0));
         }
         NSL_CHECK(nsl_norm(d->V[j + 1], &nrm));
         NSL_H(j + 1, j) = nrm;
         if (nrm > HYPRE_REAL_MIN)
         {
            NSL_CHECK(hypre_ParKrylovScaleVector(1.0 / nrm, d->V[j + 1]));
         }

         for (i = 0; i < j; i++)
         {
            HYPRE_Real u = NSL_H(i, j);
            HYPRE_Real l = NSL_H(i + 1, j);
            NSL_H(i, j) = d->cs[i] * u + d->sn[i] * l;
            NSL_H(i + 1, j) = -d->sn[i] * u + d->cs[i] * l;
         }
         diag = NSL_H(j, j);
         sub = NSL_H(j + 1, j);
         rn = nsl_pair_norm(diag, sub);
         if (!nsl_finite(rn) || rn < HYPRE_REAL_MIN)
         {
            if (k == 0)
            {
               return HYPRE_ERROR_GENERIC;
            }
            break;
         }
         d->cs[j] = diag / rn;
         d->sn[j] = sub / rn;
         NSL_H(j, j) = rn;
         NSL_H(j + 1, j) = 0.0;
         d->g[j + 1] = -d->sn[j] * d->g[j];
         d->g[j] = d->cs[j] * d->g[j];
         if (!nsl_finite(d->g[j]) || !nsl_finite(d->g[j + 1]))
         {
            return HYPRE_ERROR_GENERIC;
         }
         (*iter)++;
         k = j + 1;
         res = nsl_abs(d->g[j + 1]);
         if (res <= tol || nrm <= HYPRE_REAL_MIN)
         {
            break;
         }
      }

      if (k <= 0)
      {
         return HYPRE_ERROR_GENERIC;
      }
      for (i = k - 1; i >= 0; i--)
      {
         HYPRE_Real v = d->g[i];
         HYPRE_Real dg = NSL_H(i, i);
         for (j = i + 1; j < k; j++)
         {
            v -= NSL_H(i, j) * d->y[j];
         }
         if (!nsl_finite(v) || !nsl_finite(dg) || nsl_abs(dg) < HYPRE_REAL_MIN)
         {
            return HYPRE_ERROR_GENERIC;
         }
         d->y[i] = v / dg;
         if (!nsl_finite(d->y[i]))
         {
            return HYPRE_ERROR_GENERIC;
         }
      }
      NSL_CHECK(hypre_ParKrylovMassAxpy(d->y, d->Z, x, k, 0));
   }
}

#undef NSL_H

static HYPRE_Int
nsl_Solve(HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x)
{
   nsl_Data *d = (nsl_Data *) solver;
   HYPRE_Real rnorm;
   HYPRE_Int iter, st, status = NSL_STATUS_MAXIT;
   HYPRE_Int restarts = 0;
   HYPRE_Int valid = 0;
   HYPRE_Int r_ok = 0;

   if (d == NULL || A == NULL || b == NULL || x == NULL ||
       !d->base.is_setup || d->matrix != A ||
       d->rv == NULL || d->r == NULL || d->z == NULL || d->p == NULL ||
       d->q == NULL || d->e == NULL)
   {
      return HYPRE_ERROR_GENERIC;
   }

   if (d->have_state && d->b_ptr == (void *) b && d->x_ptr == (void *) x)
   {
      HYPRE_Real bb = hypre_ParKrylovInnerProd(b, b);
      HYPRE_Real xx = hypre_ParKrylovInnerProd(x, x);
      HYPRE_Real bx = hypre_ParKrylovInnerProd(b, x);
      valid = (bb == d->fp_bb && xx == d->fp_xx && bx == d->fp_bx) ? 1 : 0;
   }
   if (!valid)
   {
      NSL_CHECK(nsl_start(d, A, b, x));
   }
   d->have_state = 0;

   if (d->mode == NSL_MODE_ZERO)
   {
      return hypre_ParKrylovClearVector(x);
   }
   if (d->mode == NSL_MODE_DONE)
   {
      return 0;
   }
   if (d->mode == NSL_MODE_NOITER)
   {
      return HYPRE_ERROR_CONV;
   }

   if (d->e_used)
   {
      NSL_CHECK(hypre_ParKrylovAxpy(1.0, d->e, x));
   }
   iter = d->iter0;
   st = d->jst;

   if (d->mode == NSL_MODE_AMG && st == NSL_JST_CONT && d->e_used &&
       nsl_finite(d->rnorm))
   {
      r_ok = 1;
   }

   if (d->mode == NSL_MODE_JAC)
   {
      for (;;)
      {
         if (st == NSL_JST_CONV)
         {
            NSL_CHECK(nsl_true_residual(A, b, x, d->r, &d->rnorm));
            if (d->rnorm <= d->tol)
            {
               return 0;
            }
            if (iter >= d->maximum_iterations)
            {
               return HYPRE_ERROR_CONV;
            }
            restarts++;
            if (restarts > 2)
            {
               r_ok = 1;
               break;
            }
            NSL_CHECK(nsl_jac_init(d, A, &st));
            continue;
         }
         if (st == NSL_JST_BREAK)
         {
            r_ok = 0;
            break;
         }
         if (iter >= d->maximum_iterations)
         {
            NSL_CHECK(nsl_true_residual(A, b, x, d->r, &rnorm));
            return (rnorm <= d->tol) ? 0 : HYPRE_ERROR_CONV;
         }
         if (nsl_should_switch(d))
         {
            r_ok = nsl_finite(d->rnorm) ? 1 : 0;
            break;
         }
         NSL_CHECK(nsl_jac_steps(d, A, x, NSL_CHK, &iter, &st));
      }
   }

   NSL_CHECK(nsl_amg_build(d, A, b, x));
   if (r_ok)
   {
      rnorm = d->rnorm;
   }
   else
   {
      NSL_CHECK(nsl_true_residual(A, b, x, d->r, &rnorm));
      if (rnorm <= d->tol)
      {
         return 0;
      }
   }
   if (iter >= d->maximum_iterations)
   {
      NSL_CHECK(nsl_true_residual(A, b, x, d->r, &rnorm));
      return (rnorm <= d->tol) ? 0 : HYPRE_ERROR_CONV;
   }

   NSL_CHECK(nsl_pcg(d, A, b, x, d->tol, rnorm, &iter, &status));
   if (status == NSL_STATUS_CONVERGED)
   {
      return 0;
   }
   if (status == NSL_STATUS_MAXIT)
   {
      return HYPRE_ERROR_CONV;
   }
   return nsl_gmres(d, A, b, x, d->tol, &iter);
}

static HYPRE_Int
nsl_Destroy(HYPRE_Solver solver)
{
   nsl_Data *d = (nsl_Data *) solver;
   HYPRE_Int error;
   HYPRE_Int e;

   if (d == NULL)
   {
      return 0;
   }
   error = nsl_free_all(d);
   e = hypre_ParKrylovFree(d);
   return error != 0 ? error : e;
}
