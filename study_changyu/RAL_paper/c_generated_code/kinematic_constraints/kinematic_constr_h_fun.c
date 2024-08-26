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
  #define CASADI_PREFIX(ID) kinematic_constr_h_fun_ ## ID
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

static const casadi_int casadi_s0[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s1[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[29] = {25, 1, 0, 25, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
static const casadi_int casadi_s4[9] = {5, 1, 0, 5, 0, 1, 2, 3, 4};

/* kinematic_constr_h_fun:(i0[6],i1[2],i2[],i3[25])->(o0[5]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a3, a4, a5, a6, a7, a8, a9;
  a0=2.0000000000000001e-01;
  a1=5.;
  a2=arg[3]? arg[3][0] : 0;
  a3=1.0000000000000001e-01;
  a4=arg[3]? arg[3][3] : 0;
  a4=(a3*a4);
  a4=(a2+a4);
  a5=arg[0]? arg[0][0] : 0;
  a6=arg[0]? arg[0][3] : 0;
  a7=arg[0]? arg[0][2] : 0;
  a8=cos(a7);
  a8=(a6*a8);
  a8=(a3*a8);
  a8=(a5+a8);
  a9=(a4-a8);
  a10=arg[0]? arg[0][5] : 0;
  a10=(a3*a10);
  a10=(a6+a10);
  a11=(a10/a0);
  a12=arg[0]? arg[0][4] : 0;
  a12=(a3*a12);
  a12=(a7+a12);
  a13=1.5707963267948966e+00;
  a14=(a12-a13);
  a14=cos(a14);
  a14=(a11*a14);
  a9=(a9-a14);
  a9=casadi_sq(a9);
  a14=arg[3]? arg[3][1] : 0;
  a15=arg[3]? arg[3][4] : 0;
  a15=(a3*a15);
  a15=(a14+a15);
  a16=arg[0]? arg[0][1] : 0;
  a17=sin(a7);
  a17=(a6*a17);
  a17=(a3*a17);
  a17=(a16+a17);
  a18=(a15-a17);
  a19=(a12-a13);
  a19=sin(a19);
  a19=(a11*a19);
  a18=(a18-a19);
  a18=casadi_sq(a18);
  a9=(a9+a18);
  a9=sqrt(a9);
  a18=arg[3]? arg[3][2] : 0;
  a19=(a18+a11);
  a9=(a9-a19);
  a9=(a1*a9);
  a9=exp(a9);
  a4=(a4-a8);
  a19=(a12+a13);
  a19=cos(a19);
  a19=(a11*a19);
  a4=(a4-a19);
  a4=casadi_sq(a4);
  a15=(a15-a17);
  a19=(a12+a13);
  a19=sin(a19);
  a19=(a11*a19);
  a15=(a15-a19);
  a15=casadi_sq(a15);
  a4=(a4+a15);
  a4=sqrt(a4);
  a11=(a18+a11);
  a4=(a4-a11);
  a4=(a1*a4);
  a4=exp(a4);
  a9=(a9+a4);
  a4=2.;
  a9=(a9/a4);
  a9=log(a9);
  a9=(a0*a9);
  a11=9.4999999999999996e-01;
  a15=(a2-a5);
  a19=(a6/a0);
  a20=(a7-a13);
  a20=cos(a20);
  a20=(a19*a20);
  a15=(a15-a20);
  a15=casadi_sq(a15);
  a20=(a14-a16);
  a21=(a7-a13);
  a21=sin(a21);
  a21=(a19*a21);
  a20=(a20-a21);
  a20=casadi_sq(a20);
  a15=(a15+a20);
  a15=sqrt(a15);
  a20=(a18+a19);
  a15=(a15-a20);
  a15=(a1*a15);
  a15=exp(a15);
  a2=(a2-a5);
  a20=(a7+a13);
  a20=cos(a20);
  a20=(a19*a20);
  a2=(a2-a20);
  a2=casadi_sq(a2);
  a14=(a14-a16);
  a20=(a7+a13);
  a20=sin(a20);
  a20=(a19*a20);
  a14=(a14-a20);
  a14=casadi_sq(a14);
  a2=(a2+a14);
  a2=sqrt(a2);
  a18=(a18+a19);
  a2=(a2-a18);
  a2=(a1*a2);
  a2=exp(a2);
  a15=(a15+a2);
  a15=(a15/a4);
  a15=log(a15);
  a15=(a0*a15);
  a15=(a11*a15);
  a9=(a9-a15);
  if (res[0]!=0) res[0][0]=a9;
  a9=arg[3]? arg[3][5] : 0;
  a15=arg[3]? arg[3][8] : 0;
  a15=(a3*a15);
  a15=(a9+a15);
  a2=(a15-a8);
  a18=(a10/a0);
  a19=(a12-a13);
  a19=cos(a19);
  a19=(a18*a19);
  a2=(a2-a19);
  a2=casadi_sq(a2);
  a19=arg[3]? arg[3][6] : 0;
  a14=arg[3]? arg[3][9] : 0;
  a14=(a3*a14);
  a14=(a19+a14);
  a20=(a14-a17);
  a21=(a12-a13);
  a21=sin(a21);
  a21=(a18*a21);
  a20=(a20-a21);
  a20=casadi_sq(a20);
  a2=(a2+a20);
  a2=sqrt(a2);
  a20=arg[3]? arg[3][7] : 0;
  a21=(a20+a18);
  a2=(a2-a21);
  a2=(a1*a2);
  a2=exp(a2);
  a15=(a15-a8);
  a21=(a12+a13);
  a21=cos(a21);
  a21=(a18*a21);
  a15=(a15-a21);
  a15=casadi_sq(a15);
  a14=(a14-a17);
  a21=(a12+a13);
  a21=sin(a21);
  a21=(a18*a21);
  a14=(a14-a21);
  a14=casadi_sq(a14);
  a15=(a15+a14);
  a15=sqrt(a15);
  a18=(a20+a18);
  a15=(a15-a18);
  a15=(a1*a15);
  a15=exp(a15);
  a2=(a2+a15);
  a2=(a2/a4);
  a2=log(a2);
  a2=(a0*a2);
  a15=(a9-a5);
  a18=(a6/a0);
  a14=(a7-a13);
  a14=cos(a14);
  a14=(a18*a14);
  a15=(a15-a14);
  a15=casadi_sq(a15);
  a14=(a19-a16);
  a21=(a7-a13);
  a21=sin(a21);
  a21=(a18*a21);
  a14=(a14-a21);
  a14=casadi_sq(a14);
  a15=(a15+a14);
  a15=sqrt(a15);
  a14=(a20+a18);
  a15=(a15-a14);
  a15=(a1*a15);
  a15=exp(a15);
  a9=(a9-a5);
  a14=(a7+a13);
  a14=cos(a14);
  a14=(a18*a14);
  a9=(a9-a14);
  a9=casadi_sq(a9);
  a19=(a19-a16);
  a14=(a7+a13);
  a14=sin(a14);
  a14=(a18*a14);
  a19=(a19-a14);
  a19=casadi_sq(a19);
  a9=(a9+a19);
  a9=sqrt(a9);
  a20=(a20+a18);
  a9=(a9-a20);
  a9=(a1*a9);
  a9=exp(a9);
  a15=(a15+a9);
  a15=(a15/a4);
  a15=log(a15);
  a15=(a0*a15);
  a15=(a11*a15);
  a2=(a2-a15);
  if (res[0]!=0) res[0][1]=a2;
  a2=arg[3]? arg[3][10] : 0;
  a15=arg[3]? arg[3][13] : 0;
  a15=(a3*a15);
  a15=(a2+a15);
  a9=(a15-a8);
  a20=(a10/a0);
  a18=(a12-a13);
  a18=cos(a18);
  a18=(a20*a18);
  a9=(a9-a18);
  a9=casadi_sq(a9);
  a18=arg[3]? arg[3][11] : 0;
  a19=arg[3]? arg[3][14] : 0;
  a19=(a3*a19);
  a19=(a18+a19);
  a14=(a19-a17);
  a21=(a12-a13);
  a21=sin(a21);
  a21=(a20*a21);
  a14=(a14-a21);
  a14=casadi_sq(a14);
  a9=(a9+a14);
  a9=sqrt(a9);
  a14=arg[3]? arg[3][12] : 0;
  a21=(a14+a20);
  a9=(a9-a21);
  a9=(a1*a9);
  a9=exp(a9);
  a15=(a15-a8);
  a21=(a12+a13);
  a21=cos(a21);
  a21=(a20*a21);
  a15=(a15-a21);
  a15=casadi_sq(a15);
  a19=(a19-a17);
  a21=(a12+a13);
  a21=sin(a21);
  a21=(a20*a21);
  a19=(a19-a21);
  a19=casadi_sq(a19);
  a15=(a15+a19);
  a15=sqrt(a15);
  a20=(a14+a20);
  a15=(a15-a20);
  a15=(a1*a15);
  a15=exp(a15);
  a9=(a9+a15);
  a9=(a9/a4);
  a9=log(a9);
  a9=(a0*a9);
  a15=(a2-a5);
  a20=(a6/a0);
  a19=(a7-a13);
  a19=cos(a19);
  a19=(a20*a19);
  a15=(a15-a19);
  a15=casadi_sq(a15);
  a19=(a18-a16);
  a21=(a7-a13);
  a21=sin(a21);
  a21=(a20*a21);
  a19=(a19-a21);
  a19=casadi_sq(a19);
  a15=(a15+a19);
  a15=sqrt(a15);
  a19=(a14+a20);
  a15=(a15-a19);
  a15=(a1*a15);
  a15=exp(a15);
  a2=(a2-a5);
  a19=(a7+a13);
  a19=cos(a19);
  a19=(a20*a19);
  a2=(a2-a19);
  a2=casadi_sq(a2);
  a18=(a18-a16);
  a19=(a7+a13);
  a19=sin(a19);
  a19=(a20*a19);
  a18=(a18-a19);
  a18=casadi_sq(a18);
  a2=(a2+a18);
  a2=sqrt(a2);
  a14=(a14+a20);
  a2=(a2-a14);
  a2=(a1*a2);
  a2=exp(a2);
  a15=(a15+a2);
  a15=(a15/a4);
  a15=log(a15);
  a15=(a0*a15);
  a15=(a11*a15);
  a9=(a9-a15);
  if (res[0]!=0) res[0][2]=a9;
  a9=arg[3]? arg[3][15] : 0;
  a15=arg[3]? arg[3][18] : 0;
  a15=(a3*a15);
  a15=(a9+a15);
  a2=(a15-a8);
  a14=(a10/a0);
  a20=(a12-a13);
  a20=cos(a20);
  a20=(a14*a20);
  a2=(a2-a20);
  a2=casadi_sq(a2);
  a20=arg[3]? arg[3][16] : 0;
  a18=arg[3]? arg[3][19] : 0;
  a18=(a3*a18);
  a18=(a20+a18);
  a19=(a18-a17);
  a21=(a12-a13);
  a21=sin(a21);
  a21=(a14*a21);
  a19=(a19-a21);
  a19=casadi_sq(a19);
  a2=(a2+a19);
  a2=sqrt(a2);
  a19=arg[3]? arg[3][17] : 0;
  a21=(a19+a14);
  a2=(a2-a21);
  a2=(a1*a2);
  a2=exp(a2);
  a15=(a15-a8);
  a21=(a12+a13);
  a21=cos(a21);
  a21=(a14*a21);
  a15=(a15-a21);
  a15=casadi_sq(a15);
  a18=(a18-a17);
  a21=(a12+a13);
  a21=sin(a21);
  a21=(a14*a21);
  a18=(a18-a21);
  a18=casadi_sq(a18);
  a15=(a15+a18);
  a15=sqrt(a15);
  a14=(a19+a14);
  a15=(a15-a14);
  a15=(a1*a15);
  a15=exp(a15);
  a2=(a2+a15);
  a2=(a2/a4);
  a2=log(a2);
  a2=(a0*a2);
  a15=(a9-a5);
  a14=(a6/a0);
  a18=(a7-a13);
  a18=cos(a18);
  a18=(a14*a18);
  a15=(a15-a18);
  a15=casadi_sq(a15);
  a18=(a20-a16);
  a21=(a7-a13);
  a21=sin(a21);
  a21=(a14*a21);
  a18=(a18-a21);
  a18=casadi_sq(a18);
  a15=(a15+a18);
  a15=sqrt(a15);
  a18=(a19+a14);
  a15=(a15-a18);
  a15=(a1*a15);
  a15=exp(a15);
  a9=(a9-a5);
  a18=(a7+a13);
  a18=cos(a18);
  a18=(a14*a18);
  a9=(a9-a18);
  a9=casadi_sq(a9);
  a20=(a20-a16);
  a18=(a7+a13);
  a18=sin(a18);
  a18=(a14*a18);
  a20=(a20-a18);
  a20=casadi_sq(a20);
  a9=(a9+a20);
  a9=sqrt(a9);
  a19=(a19+a14);
  a9=(a9-a19);
  a9=(a1*a9);
  a9=exp(a9);
  a15=(a15+a9);
  a15=(a15/a4);
  a15=log(a15);
  a15=(a0*a15);
  a15=(a11*a15);
  a2=(a2-a15);
  if (res[0]!=0) res[0][3]=a2;
  a2=arg[3]? arg[3][20] : 0;
  a15=arg[3]? arg[3][23] : 0;
  a15=(a3*a15);
  a15=(a2+a15);
  a9=(a15-a8);
  a10=(a10/a0);
  a19=(a12-a13);
  a19=cos(a19);
  a19=(a10*a19);
  a9=(a9-a19);
  a9=casadi_sq(a9);
  a19=arg[3]? arg[3][21] : 0;
  a14=arg[3]? arg[3][24] : 0;
  a3=(a3*a14);
  a3=(a19+a3);
  a14=(a3-a17);
  a20=(a12-a13);
  a20=sin(a20);
  a20=(a10*a20);
  a14=(a14-a20);
  a14=casadi_sq(a14);
  a9=(a9+a14);
  a9=sqrt(a9);
  a14=arg[3]? arg[3][22] : 0;
  a20=(a14+a10);
  a9=(a9-a20);
  a9=(a1*a9);
  a9=exp(a9);
  a15=(a15-a8);
  a8=(a12+a13);
  a8=cos(a8);
  a8=(a10*a8);
  a15=(a15-a8);
  a15=casadi_sq(a15);
  a3=(a3-a17);
  a12=(a12+a13);
  a12=sin(a12);
  a12=(a10*a12);
  a3=(a3-a12);
  a3=casadi_sq(a3);
  a15=(a15+a3);
  a15=sqrt(a15);
  a10=(a14+a10);
  a15=(a15-a10);
  a15=(a1*a15);
  a15=exp(a15);
  a9=(a9+a15);
  a9=(a9/a4);
  a9=log(a9);
  a9=(a0*a9);
  a15=(a2-a5);
  a6=(a6/a0);
  a10=(a7-a13);
  a10=cos(a10);
  a10=(a6*a10);
  a15=(a15-a10);
  a15=casadi_sq(a15);
  a10=(a19-a16);
  a3=(a7-a13);
  a3=sin(a3);
  a3=(a6*a3);
  a10=(a10-a3);
  a10=casadi_sq(a10);
  a15=(a15+a10);
  a15=sqrt(a15);
  a10=(a14+a6);
  a15=(a15-a10);
  a15=(a1*a15);
  a15=exp(a15);
  a2=(a2-a5);
  a5=(a7+a13);
  a5=cos(a5);
  a5=(a6*a5);
  a2=(a2-a5);
  a2=casadi_sq(a2);
  a19=(a19-a16);
  a7=(a7+a13);
  a7=sin(a7);
  a7=(a6*a7);
  a19=(a19-a7);
  a19=casadi_sq(a19);
  a2=(a2+a19);
  a2=sqrt(a2);
  a14=(a14+a6);
  a2=(a2-a14);
  a1=(a1*a2);
  a1=exp(a1);
  a15=(a15+a1);
  a15=(a15/a4);
  a15=log(a15);
  a0=(a0*a15);
  a11=(a11*a0);
  a9=(a9-a11);
  if (res[0]!=0) res[0][4]=a9;
  return 0;
}

CASADI_SYMBOL_EXPORT int kinematic_constr_h_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int kinematic_constr_h_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int kinematic_constr_h_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void kinematic_constr_h_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int kinematic_constr_h_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void kinematic_constr_h_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void kinematic_constr_h_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void kinematic_constr_h_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int kinematic_constr_h_fun_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int kinematic_constr_h_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real kinematic_constr_h_fun_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* kinematic_constr_h_fun_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* kinematic_constr_h_fun_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* kinematic_constr_h_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* kinematic_constr_h_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int kinematic_constr_h_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
