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

static const casadi_int casadi_s0[12] = {8, 1, 0, 8, 0, 1, 2, 3, 4, 5, 6, 7};
static const casadi_int casadi_s1[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s4[27] = {8, 8, 0, 0, 0, 2, 5, 9, 12, 14, 16, 0, 1, 0, 1, 3, 0, 1, 4, 5, 2, 4, 5, 3, 5, 3, 5};
static const casadi_int casadi_s5[19] = {8, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7};
static const casadi_int casadi_s6[7] = {8, 2, 0, 1, 2, 6, 7};
static const casadi_int casadi_s7[3] = {8, 0, 0};

/* heron_impl_dae_jac_x_xdot_u_z:(i0[8],i1[8],i2[2],i3[],i4[],i5[6])->(o0[8x8,16nz],o1[8x8,8nz],o2[8x2,2nz],o3[8x0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2, a3, a4, a5, a6, a7, a8;
  a0=arg[0]? arg[0][4] : 0;
  a1=arg[0]? arg[0][2] : 0;
  a2=cos(a1);
  a2=(a0*a2);
  a3=arg[0]? arg[0][3] : 0;
  a4=sin(a1);
  a4=(a3*a4);
  a2=(a2-a4);
  a2=(-a2);
  if (res[0]!=0) res[0][0]=a2;
  a2=cos(a1);
  a2=(a3*a2);
  a4=sin(a1);
  a4=(a0*a4);
  a2=(a2+a4);
  a2=(-a2);
  if (res[0]!=0) res[0][1]=a2;
  a2=cos(a1);
  a2=(-a2);
  if (res[0]!=0) res[0][2]=a2;
  a2=sin(a1);
  a2=(-a2);
  if (res[0]!=0) res[0][3]=a2;
  a2=2.6484453625721698e-02;
  a4=-8.9148999999999994e+00;
  a5=1.1210100000000001e+01;
  a6=casadi_sq(a3);
  a7=1.0000000000000001e-05;
  a6=(a6+a7);
  a6=sqrt(a6);
  a8=(a3/a6);
  a8=(a5*a8);
  a3=(a3*a8);
  a5=(a5*a6);
  a3=(a3+a5);
  a4=(a4-a3);
  a4=(a2*a4);
  a4=(-a4);
  if (res[0]!=0) res[0][4]=a4;
  a4=sin(a1);
  a4=(-a4);
  if (res[0]!=0) res[0][5]=a4;
  a1=cos(a1);
  if (res[0]!=0) res[0][6]=a1;
  a1=-15.;
  a4=3.;
  a3=casadi_sq(a0);
  a3=(a3+a7);
  a3=sqrt(a3);
  a7=(a0/a3);
  a7=(a4*a7);
  a0=(a0*a7);
  a4=(a4*a3);
  a0=(a0+a4);
  a1=(a1-a0);
  a2=(a2*a1);
  a2=(-a2);
  if (res[0]!=0) res[0][7]=a2;
  a2=3.2697547683923700e-01;
  if (res[0]!=0) res[0][8]=a2;
  a2=-1.;
  if (res[0]!=0) res[0][9]=a2;
  a1=1.5890672175433018e-01;
  if (res[0]!=0) res[0][10]=a1;
  a1=5.4495912806539502e-02;
  a0=-1.6954200000000000e+01;
  a4=arg[0]? arg[0][5] : 0;
  a3=1.2896599999999999e+01;
  a7=(a3*a4);
  a3=(a3*a4);
  a7=(a7+a3);
  a7=(a4*a7);
  a3=(a3*a4);
  a7=(a7+a3);
  a0=(a0-a7);
  a1=(a1*a0);
  a1=(-a1);
  if (res[0]!=0) res[0][11]=a1;
  a1=-2.6484453625721698e-02;
  if (res[0]!=0) res[0][12]=a1;
  a0=1.6348773841961851e-02;
  if (res[0]!=0) res[0][13]=a0;
  if (res[0]!=0) res[0][14]=a1;
  a1=-1.6348773841961851e-02;
  if (res[0]!=0) res[0][15]=a1;
  a1=1.;
  if (res[1]!=0) res[1][0]=a1;
  if (res[1]!=0) res[1][1]=a1;
  if (res[1]!=0) res[1][2]=a1;
  if (res[1]!=0) res[1][3]=a1;
  if (res[1]!=0) res[1][4]=a1;
  if (res[1]!=0) res[1][5]=a1;
  if (res[1]!=0) res[1][6]=a1;
  if (res[1]!=0) res[1][7]=a1;
  if (res[2]!=0) res[2][0]=a2;
  if (res[2]!=0) res[2][1]=a2;
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

CASADI_SYMBOL_EXPORT casadi_real heron_impl_dae_jac_x_xdot_u_z_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* heron_impl_dae_jac_x_xdot_u_z_name_in(casadi_int i) {
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

CASADI_SYMBOL_EXPORT const char* heron_impl_dae_jac_x_xdot_u_z_name_out(casadi_int i) {
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
