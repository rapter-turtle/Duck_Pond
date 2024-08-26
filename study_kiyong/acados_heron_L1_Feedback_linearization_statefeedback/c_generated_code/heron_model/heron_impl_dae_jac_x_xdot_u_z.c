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
  #define CASADI_PREFIX(ID) heron_impl_dae_jac_x_xdot_u_z_ ## ID
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
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_s7 CASADI_PREFIX(s7)
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
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[9] = {5, 1, 0, 5, 0, 1, 2, 3, 4};
static const casadi_int casadi_s4[21] = {7, 7, 0, 0, 0, 2, 5, 7, 9, 11, 0, 1, 0, 1, 3, 2, 4, 3, 4, 3, 4};
static const casadi_int casadi_s5[17] = {7, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6};
static const casadi_int casadi_s6[7] = {7, 2, 0, 1, 2, 5, 6};
static const casadi_int casadi_s7[3] = {7, 0, 0};

/* heron_impl_dae_jac_x_xdot_u_z:(i0[7],i1[7],i2[2],i3[],i4[],i5[5])->(o0[7x7,11nz],o1[7x7,7nz],o2[7x2,2nz],o3[7x0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2, a3, a4, a5;
  a0=arg[0]? arg[0][3] : 0;
  a1=arg[0]? arg[0][2] : 0;
  a2=sin(a1);
  a2=(a0*a2);
  if (res[0]!=0) res[0][0]=a2;
  a2=cos(a1);
  a2=(a0*a2);
  a2=(-a2);
  if (res[0]!=0) res[0][1]=a2;
  a2=cos(a1);
  a2=(-a2);
  if (res[0]!=0) res[0][2]=a2;
  a1=sin(a1);
  a1=(-a1);
  if (res[0]!=0) res[0][3]=a1;
  a1=2.7777777777777776e-02;
  a2=1.6899999999999999e+01;
  a3=casadi_sq(a0);
  a4=1.0000000000000001e-05;
  a3=(a3+a4);
  a3=sqrt(a3);
  a5=(a0/a3);
  a5=(a2*a5);
  a0=(a0*a5);
  a5=10.;
  a2=(a2*a3);
  a5=(a5+a2);
  a0=(a0+a5);
  a1=(a1*a0);
  if (res[0]!=0) res[0][4]=a1;
  a1=-1.;
  if (res[0]!=0) res[0][5]=a1;
  a0=1.1976047904191617e-01;
  a5=arg[0]? arg[0][4] : 0;
  a2=13.;
  a3=casadi_sq(a5);
  a3=(a3+a4);
  a3=sqrt(a3);
  a4=(a5/a3);
  a4=(a2*a4);
  a5=(a5*a4);
  a4=5.;
  a2=(a2*a3);
  a4=(a4+a2);
  a5=(a5+a4);
  a0=(a0*a5);
  if (res[0]!=0) res[0][6]=a0;
  a0=-2.7777777777777776e-02;
  if (res[0]!=0) res[0][7]=a0;
  a5=4.3712574850299397e-02;
  if (res[0]!=0) res[0][8]=a5;
  if (res[0]!=0) res[0][9]=a0;
  a0=-4.3712574850299397e-02;
  if (res[0]!=0) res[0][10]=a0;
  a0=1.;
  if (res[1]!=0) res[1][0]=a0;
  if (res[1]!=0) res[1][1]=a0;
  if (res[1]!=0) res[1][2]=a0;
  if (res[1]!=0) res[1][3]=a0;
  if (res[1]!=0) res[1][4]=a0;
  if (res[1]!=0) res[1][5]=a0;
  if (res[1]!=0) res[1][6]=a0;
  if (res[2]!=0) res[2][0]=a1;
  if (res[2]!=0) res[2][1]=a1;
  return 0;
}

CASADI_SYMBOL_EXPORT int heron_impl_dae_jac_x_xdot_u_z(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int heron_impl_dae_jac_x_xdot_u_z_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int heron_impl_dae_jac_x_xdot_u_z_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void heron_impl_dae_jac_x_xdot_u_z_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int heron_impl_dae_jac_x_xdot_u_z_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void heron_impl_dae_jac_x_xdot_u_z_release(int mem) {
}

CASADI_SYMBOL_EXPORT void heron_impl_dae_jac_x_xdot_u_z_incref(void) {
}

CASADI_SYMBOL_EXPORT void heron_impl_dae_jac_x_xdot_u_z_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int heron_impl_dae_jac_x_xdot_u_z_n_in(void) { return 6;}

CASADI_SYMBOL_EXPORT casadi_int heron_impl_dae_jac_x_xdot_u_z_n_out(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_real heron_impl_dae_jac_x_xdot_u_z_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* heron_impl_dae_jac_x_xdot_u_z_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    case 5: return "i5";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* heron_impl_dae_jac_x_xdot_u_z_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* heron_impl_dae_jac_x_xdot_u_z_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    case 4: return casadi_s2;
    case 5: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* heron_impl_dae_jac_x_xdot_u_z_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    case 1: return casadi_s5;
    case 2: return casadi_s6;
    case 3: return casadi_s7;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int heron_impl_dae_jac_x_xdot_u_z_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 6;
  if (sz_res) *sz_res = 4;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
