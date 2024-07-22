/* This file was automatically generated by CasADi 3.6.4.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) ship_expl_ode_fun_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[12] = {8, 1, 0, 8, 0, 1, 2, 3, 4, 5, 6, 7};
static const casadi_int casadi_s1[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s2[29] = {25, 1, 0, 25, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

/* ship_expl_ode_fun:(i0[8],i1[2],i2[25])->(o0[8]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2, a3, a4, a5, a6, a7, a8;
  a0=arg[0]? arg[0][3] : 0;
  a1=arg[0]? arg[0][2] : 0;
  a2=cos(a1);
  a2=(a0*a2);
  a3=arg[0]? arg[0][4] : 0;
  a4=sin(a1);
  a4=(a3*a4);
  a2=(a2+a4);
  if (res[0]!=0) res[0][0]=a2;
  a2=sin(a1);
  a2=(a0*a2);
  a1=cos(a1);
  a1=(a3*a1);
  a2=(a2-a1);
  if (res[0]!=0) res[0][1]=a2;
  a2=arg[0]? arg[0][5] : 0;
  if (res[0]!=0) res[0][2]=a2;
  a1=arg[0]? arg[0][6] : 0;
  a4=36.;
  a5=(a4*a3);
  a5=(a5*a2);
  a1=(a1+a5);
  a5=10.;
  a6=1.6899999999999999e+01;
  a7=casadi_sq(a0);
  a8=1.0000000000000001e-05;
  a7=(a7+a8);
  a7=sqrt(a7);
  a6=(a6*a7);
  a5=(a5+a6);
  a5=(a5*a0);
  a1=(a1-a5);
  a1=(a1/a4);
  if (res[0]!=0) res[0][3]=a1;
  a1=-36.;
  a1=(a1*a0);
  a1=(a1*a2);
  a0=20.;
  a0=(a0*a3);
  a1=(a1-a0);
  a1=(a1/a4);
  if (res[0]!=0) res[0][4]=a1;
  a1=arg[0]? arg[0][7] : 0;
  a4=15.;
  a0=40.;
  a3=casadi_sq(a2);
  a3=(a3+a8);
  a3=sqrt(a3);
  a0=(a0*a3);
  a4=(a4+a0);
  a4=(a4*a2);
  a1=(a1-a4);
  a4=5.8350000000000001e+01;
  a1=(a1/a4);
  if (res[0]!=0) res[0][5]=a1;
  a1=arg[1]? arg[1][0] : 0;
  if (res[0]!=0) res[0][6]=a1;
  a1=arg[1]? arg[1][1] : 0;
  if (res[0]!=0) res[0][7]=a1;
  return 0;
}

CASADI_SYMBOL_EXPORT int ship_expl_ode_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int ship_expl_ode_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int ship_expl_ode_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void ship_expl_ode_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int ship_expl_ode_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void ship_expl_ode_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void ship_expl_ode_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void ship_expl_ode_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int ship_expl_ode_fun_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int ship_expl_ode_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real ship_expl_ode_fun_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* ship_expl_ode_fun_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* ship_expl_ode_fun_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* ship_expl_ode_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* ship_expl_ode_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int ship_expl_ode_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 3;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
