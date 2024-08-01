/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) heron_expl_vde_adj_ ## ID
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
#define casadi_s3 CASADI_PREFIX(s3)
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

static const casadi_int casadi_s0[11] = {7, 1, 0, 7, 0, 1, 2, 3, 4, 5, 6};
static const casadi_int casadi_s1[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s2[9] = {5, 1, 0, 5, 0, 1, 2, 3, 4};
static const casadi_int casadi_s3[13] = {9, 1, 0, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8};

/* heron_expl_vde_adj:(i0[7],i1[7],i2[2],i3[5])->(o0[9]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=0.;
  if (res[0]!=0) res[0][0]=a0;
  if (res[0]!=0) res[0][1]=a0;
  a0=arg[0]? arg[0][2] : 0;
  a1=cos(a0);
  a2=arg[0]? arg[0][3] : 0;
  a3=arg[1]? arg[1][1] : 0;
  a4=(a2*a3);
  a1=(a1*a4);
  a4=sin(a0);
  a5=arg[1]? arg[1][0] : 0;
  a6=(a2*a5);
  a4=(a4*a6);
  a1=(a1-a4);
  if (res[0]!=0) res[0][2]=a1;
  a1=sin(a0);
  a1=(a1*a3);
  a3=10.;
  a4=1.6899999999999999e+01;
  a6=casadi_sq(a2);
  a7=1.0000000000000001e-05;
  a6=(a6+a7);
  a6=sqrt(a6);
  a8=(a4*a6);
  a3=(a3+a8);
  a8=2.7777777777777776e-02;
  a9=arg[1]? arg[1][3] : 0;
  a8=(a8*a9);
  a3=(a3*a8);
  a9=(a2+a2);
  a2=(a2*a8);
  a4=(a4*a2);
  a6=(a6+a6);
  a4=(a4/a6);
  a9=(a9*a4);
  a3=(a3+a9);
  a1=(a1-a3);
  a0=cos(a0);
  a0=(a0*a5);
  a1=(a1+a0);
  if (res[0]!=0) res[0][3]=a1;
  a1=arg[1]? arg[1][2] : 0;
  a0=5.;
  a5=13.;
  a3=arg[0]? arg[0][4] : 0;
  a9=casadi_sq(a3);
  a9=(a9+a7);
  a9=sqrt(a9);
  a7=(a5*a9);
  a0=(a0+a7);
  a7=1.1976047904191617e-01;
  a4=arg[1]? arg[1][4] : 0;
  a7=(a7*a4);
  a0=(a0*a7);
  a4=(a3+a3);
  a3=(a3*a7);
  a5=(a5*a3);
  a9=(a9+a9);
  a5=(a5/a9);
  a4=(a4*a5);
  a0=(a0+a4);
  a1=(a1-a0);
  if (res[0]!=0) res[0][4]=a1;
  a1=7.2999999999999998e-01;
  a0=5.0000000000000000e-01;
  a0=(a0*a7);
  a1=(a1*a0);
  a0=(a8-a1);
  if (res[0]!=0) res[0][5]=a0;
  a1=(a1+a8);
  if (res[0]!=0) res[0][6]=a1;
  a1=arg[1]? arg[1][5] : 0;
  if (res[0]!=0) res[0][7]=a1;
  a1=arg[1]? arg[1][6] : 0;
  if (res[0]!=0) res[0][8]=a1;
  return 0;
}

CASADI_SYMBOL_EXPORT int heron_expl_vde_adj(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int heron_expl_vde_adj_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int heron_expl_vde_adj_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void heron_expl_vde_adj_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int heron_expl_vde_adj_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void heron_expl_vde_adj_release(int mem) {
}

CASADI_SYMBOL_EXPORT void heron_expl_vde_adj_incref(void) {
}

CASADI_SYMBOL_EXPORT void heron_expl_vde_adj_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int heron_expl_vde_adj_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int heron_expl_vde_adj_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real heron_expl_vde_adj_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* heron_expl_vde_adj_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* heron_expl_vde_adj_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* heron_expl_vde_adj_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* heron_expl_vde_adj_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int heron_expl_vde_adj_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif