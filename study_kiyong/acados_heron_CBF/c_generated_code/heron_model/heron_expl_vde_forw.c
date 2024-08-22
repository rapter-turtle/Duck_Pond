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
  #define CASADI_PREFIX(ID) heron_expl_vde_forw_ ## ID
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

static const casadi_int casadi_s0[12] = {8, 1, 0, 8, 0, 1, 2, 3, 4, 5, 6, 7};
static const casadi_int casadi_s1[75] = {8, 8, 0, 8, 16, 24, 32, 40, 48, 56, 64, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
static const casadi_int casadi_s2[21] = {8, 2, 0, 8, 16, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
static const casadi_int casadi_s3[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s4[9] = {5, 1, 0, 5, 0, 1, 2, 3, 4};

/* heron_expl_vde_forw:(i0[8],i1[8x8],i2[8x2],i3[2],i4[5])->(o0[8],o1[8x8],o2[8x2]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a5, a6, a7, a8, a9;
  a0=arg[0]? arg[0][3] : 0;
  a1=arg[0]? arg[0][2] : 0;
  a2=cos(a1);
  a3=(a0*a2);
  a4=arg[0]? arg[0][4] : 0;
  a5=sin(a1);
  a6=(a4*a5);
  a3=(a3-a6);
  a6=arg[0]? arg[0][5] : 0;
  a7=sin(a1);
  a8=(a6*a7);
  a3=(a3-a8);
  if (res[0]!=0) res[0][0]=a3;
  a3=sin(a1);
  a8=(a0*a3);
  a9=cos(a1);
  a10=(a4*a9);
  a8=(a8+a10);
  a10=cos(a1);
  a11=(a6*a10);
  a8=(a8+a11);
  if (res[0]!=0) res[0][1]=a8;
  if (res[0]!=0) res[0][2]=a6;
  a8=arg[0]? arg[0][6] : 0;
  a11=arg[0]? arg[0][7] : 0;
  a12=(a8+a11);
  a13=8.9148999999999994e+00;
  a14=(a13*a0);
  a12=(a12-a14);
  a14=1.1210100000000001e+01;
  a15=casadi_sq(a0);
  a16=1.0000000000000001e-05;
  a15=(a15+a16);
  a15=sqrt(a15);
  a17=(a14*a15);
  a18=(a17*a0);
  a12=(a12-a18);
  a18=3.7758000000000003e+01;
  a12=(a12/a18);
  if (res[0]!=0) res[0][3]=a12;
  a12=-15.;
  a19=(a12*a4);
  a20=3.;
  a21=casadi_sq(a4);
  a21=(a21+a16);
  a21=sqrt(a21);
  a16=(a20*a21);
  a22=(a16*a4);
  a19=(a19-a22);
  a22=6.;
  a23=(a22*a6);
  a19=(a19-a23);
  a19=(a19/a18);
  if (res[0]!=0) res[0][4]=a19;
  a19=2.9999999999999999e-01;
  a11=(a11-a8);
  a11=(a19*a11);
  a8=1.6954200000000000e+01;
  a18=(a8*a6);
  a11=(a11-a18);
  a18=1.2896599999999999e+01;
  a23=(a18*a6);
  a24=(a23*a6);
  a25=(a24*a6);
  a11=(a11-a25);
  a25=(a22*a4);
  a11=(a11-a25);
  a25=1.8350000000000001e+01;
  a11=(a11/a25);
  if (res[0]!=0) res[0][5]=a11;
  a11=arg[3]? arg[3][0] : 0;
  if (res[0]!=0) res[0][6]=a11;
  a11=arg[3]? arg[3][1] : 0;
  if (res[0]!=0) res[0][7]=a11;
  a11=arg[1]? arg[1][3] : 0;
  a25=(a2*a11);
  a26=sin(a1);
  a27=arg[1]? arg[1][2] : 0;
  a28=(a26*a27);
  a28=(a0*a28);
  a25=(a25-a28);
  a28=arg[1]? arg[1][4] : 0;
  a29=(a5*a28);
  a30=cos(a1);
  a31=(a30*a27);
  a31=(a4*a31);
  a29=(a29+a31);
  a25=(a25-a29);
  a29=arg[1]? arg[1][5] : 0;
  a31=(a7*a29);
  a32=cos(a1);
  a33=(a32*a27);
  a33=(a6*a33);
  a31=(a31+a33);
  a25=(a25-a31);
  if (res[1]!=0) res[1][0]=a25;
  a25=(a3*a11);
  a31=cos(a1);
  a33=(a31*a27);
  a33=(a0*a33);
  a25=(a25+a33);
  a33=(a9*a28);
  a34=sin(a1);
  a35=(a34*a27);
  a35=(a4*a35);
  a33=(a33-a35);
  a25=(a25+a33);
  a33=(a10*a29);
  a35=sin(a1);
  a27=(a35*a27);
  a27=(a6*a27);
  a33=(a33-a27);
  a25=(a25+a33);
  if (res[1]!=0) res[1][1]=a25;
  if (res[1]!=0) res[1][2]=a29;
  a25=2.6484453625721698e-02;
  a33=arg[1]? arg[1][6] : 0;
  a27=arg[1]? arg[1][7] : 0;
  a36=(a33+a27);
  a37=(a13*a11);
  a36=(a36-a37);
  a37=(a0+a0);
  a38=(a37*a11);
  a39=(a15+a15);
  a38=(a38/a39);
  a38=(a14*a38);
  a38=(a0*a38);
  a11=(a17*a11);
  a38=(a38+a11);
  a36=(a36-a38);
  a36=(a25*a36);
  if (res[1]!=0) res[1][3]=a36;
  a36=(a12*a28);
  a38=(a4+a4);
  a11=(a38*a28);
  a40=(a21+a21);
  a11=(a11/a40);
  a11=(a20*a11);
  a11=(a4*a11);
  a41=(a16*a28);
  a11=(a11+a41);
  a36=(a36-a11);
  a11=(a22*a29);
  a36=(a36-a11);
  a36=(a25*a36);
  if (res[1]!=0) res[1][4]=a36;
  a36=5.4495912806539502e-02;
  a27=(a27-a33);
  a27=(a19*a27);
  a33=(a8*a29);
  a27=(a27-a33);
  a33=(a18*a29);
  a33=(a6*a33);
  a11=(a23*a29);
  a33=(a33+a11);
  a33=(a6*a33);
  a29=(a24*a29);
  a33=(a33+a29);
  a27=(a27-a33);
  a28=(a22*a28);
  a27=(a27-a28);
  a27=(a36*a27);
  if (res[1]!=0) res[1][5]=a27;
  a27=0.;
  if (res[1]!=0) res[1][6]=a27;
  if (res[1]!=0) res[1][7]=a27;
  a28=arg[1]? arg[1][11] : 0;
  a33=(a2*a28);
  a29=arg[1]? arg[1][10] : 0;
  a11=(a26*a29);
  a11=(a0*a11);
  a33=(a33-a11);
  a11=arg[1]? arg[1][12] : 0;
  a41=(a5*a11);
  a42=(a30*a29);
  a42=(a4*a42);
  a41=(a41+a42);
  a33=(a33-a41);
  a41=arg[1]? arg[1][13] : 0;
  a42=(a7*a41);
  a43=(a32*a29);
  a43=(a6*a43);
  a42=(a42+a43);
  a33=(a33-a42);
  if (res[1]!=0) res[1][8]=a33;
  a33=(a3*a28);
  a42=(a31*a29);
  a42=(a0*a42);
  a33=(a33+a42);
  a42=(a9*a11);
  a43=(a34*a29);
  a43=(a4*a43);
  a42=(a42-a43);
  a33=(a33+a42);
  a42=(a10*a41);
  a29=(a35*a29);
  a29=(a6*a29);
  a42=(a42-a29);
  a33=(a33+a42);
  if (res[1]!=0) res[1][9]=a33;
  if (res[1]!=0) res[1][10]=a41;
  a33=arg[1]? arg[1][14] : 0;
  a42=arg[1]? arg[1][15] : 0;
  a29=(a33+a42);
  a43=(a13*a28);
  a29=(a29-a43);
  a43=(a37*a28);
  a43=(a43/a39);
  a43=(a14*a43);
  a43=(a0*a43);
  a28=(a17*a28);
  a43=(a43+a28);
  a29=(a29-a43);
  a29=(a25*a29);
  if (res[1]!=0) res[1][11]=a29;
  a29=(a12*a11);
  a43=(a38*a11);
  a43=(a43/a40);
  a43=(a20*a43);
  a43=(a4*a43);
  a28=(a16*a11);
  a43=(a43+a28);
  a29=(a29-a43);
  a43=(a22*a41);
  a29=(a29-a43);
  a29=(a25*a29);
  if (res[1]!=0) res[1][12]=a29;
  a42=(a42-a33);
  a42=(a19*a42);
  a33=(a8*a41);
  a42=(a42-a33);
  a33=(a18*a41);
  a33=(a6*a33);
  a29=(a23*a41);
  a33=(a33+a29);
  a33=(a6*a33);
  a41=(a24*a41);
  a33=(a33+a41);
  a42=(a42-a33);
  a11=(a22*a11);
  a42=(a42-a11);
  a42=(a36*a42);
  if (res[1]!=0) res[1][13]=a42;
  if (res[1]!=0) res[1][14]=a27;
  if (res[1]!=0) res[1][15]=a27;
  a42=arg[1]? arg[1][19] : 0;
  a11=(a2*a42);
  a33=arg[1]? arg[1][18] : 0;
  a41=(a26*a33);
  a41=(a0*a41);
  a11=(a11-a41);
  a41=arg[1]? arg[1][20] : 0;
  a29=(a5*a41);
  a43=(a30*a33);
  a43=(a4*a43);
  a29=(a29+a43);
  a11=(a11-a29);
  a29=arg[1]? arg[1][21] : 0;
  a43=(a7*a29);
  a28=(a32*a33);
  a28=(a6*a28);
  a43=(a43+a28);
  a11=(a11-a43);
  if (res[1]!=0) res[1][16]=a11;
  a11=(a3*a42);
  a43=(a31*a33);
  a43=(a0*a43);
  a11=(a11+a43);
  a43=(a9*a41);
  a28=(a34*a33);
  a28=(a4*a28);
  a43=(a43-a28);
  a11=(a11+a43);
  a43=(a10*a29);
  a33=(a35*a33);
  a33=(a6*a33);
  a43=(a43-a33);
  a11=(a11+a43);
  if (res[1]!=0) res[1][17]=a11;
  if (res[1]!=0) res[1][18]=a29;
  a11=arg[1]? arg[1][22] : 0;
  a43=arg[1]? arg[1][23] : 0;
  a33=(a11+a43);
  a28=(a13*a42);
  a33=(a33-a28);
  a28=(a37*a42);
  a28=(a28/a39);
  a28=(a14*a28);
  a28=(a0*a28);
  a42=(a17*a42);
  a28=(a28+a42);
  a33=(a33-a28);
  a33=(a25*a33);
  if (res[1]!=0) res[1][19]=a33;
  a33=(a12*a41);
  a28=(a38*a41);
  a28=(a28/a40);
  a28=(a20*a28);
  a28=(a4*a28);
  a42=(a16*a41);
  a28=(a28+a42);
  a33=(a33-a28);
  a28=(a22*a29);
  a33=(a33-a28);
  a33=(a25*a33);
  if (res[1]!=0) res[1][20]=a33;
  a43=(a43-a11);
  a43=(a19*a43);
  a11=(a8*a29);
  a43=(a43-a11);
  a11=(a18*a29);
  a11=(a6*a11);
  a33=(a23*a29);
  a11=(a11+a33);
  a11=(a6*a11);
  a29=(a24*a29);
  a11=(a11+a29);
  a43=(a43-a11);
  a41=(a22*a41);
  a43=(a43-a41);
  a43=(a36*a43);
  if (res[1]!=0) res[1][21]=a43;
  if (res[1]!=0) res[1][22]=a27;
  if (res[1]!=0) res[1][23]=a27;
  a43=arg[1]? arg[1][27] : 0;
  a41=(a2*a43);
  a11=arg[1]? arg[1][26] : 0;
  a29=(a26*a11);
  a29=(a0*a29);
  a41=(a41-a29);
  a29=arg[1]? arg[1][28] : 0;
  a33=(a5*a29);
  a28=(a30*a11);
  a28=(a4*a28);
  a33=(a33+a28);
  a41=(a41-a33);
  a33=arg[1]? arg[1][29] : 0;
  a28=(a7*a33);
  a42=(a32*a11);
  a42=(a6*a42);
  a28=(a28+a42);
  a41=(a41-a28);
  if (res[1]!=0) res[1][24]=a41;
  a41=(a3*a43);
  a28=(a31*a11);
  a28=(a0*a28);
  a41=(a41+a28);
  a28=(a9*a29);
  a42=(a34*a11);
  a42=(a4*a42);
  a28=(a28-a42);
  a41=(a41+a28);
  a28=(a10*a33);
  a11=(a35*a11);
  a11=(a6*a11);
  a28=(a28-a11);
  a41=(a41+a28);
  if (res[1]!=0) res[1][25]=a41;
  if (res[1]!=0) res[1][26]=a33;
  a41=arg[1]? arg[1][30] : 0;
  a28=arg[1]? arg[1][31] : 0;
  a11=(a41+a28);
  a42=(a13*a43);
  a11=(a11-a42);
  a42=(a37*a43);
  a42=(a42/a39);
  a42=(a14*a42);
  a42=(a0*a42);
  a43=(a17*a43);
  a42=(a42+a43);
  a11=(a11-a42);
  a11=(a25*a11);
  if (res[1]!=0) res[1][27]=a11;
  a11=(a12*a29);
  a42=(a38*a29);
  a42=(a42/a40);
  a42=(a20*a42);
  a42=(a4*a42);
  a43=(a16*a29);
  a42=(a42+a43);
  a11=(a11-a42);
  a42=(a22*a33);
  a11=(a11-a42);
  a11=(a25*a11);
  if (res[1]!=0) res[1][28]=a11;
  a28=(a28-a41);
  a28=(a19*a28);
  a41=(a8*a33);
  a28=(a28-a41);
  a41=(a18*a33);
  a41=(a6*a41);
  a11=(a23*a33);
  a41=(a41+a11);
  a41=(a6*a41);
  a33=(a24*a33);
  a41=(a41+a33);
  a28=(a28-a41);
  a29=(a22*a29);
  a28=(a28-a29);
  a28=(a36*a28);
  if (res[1]!=0) res[1][29]=a28;
  if (res[1]!=0) res[1][30]=a27;
  if (res[1]!=0) res[1][31]=a27;
  a28=arg[1]? arg[1][35] : 0;
  a29=(a2*a28);
  a41=arg[1]? arg[1][34] : 0;
  a33=(a26*a41);
  a33=(a0*a33);
  a29=(a29-a33);
  a33=arg[1]? arg[1][36] : 0;
  a11=(a5*a33);
  a42=(a30*a41);
  a42=(a4*a42);
  a11=(a11+a42);
  a29=(a29-a11);
  a11=arg[1]? arg[1][37] : 0;
  a42=(a7*a11);
  a43=(a32*a41);
  a43=(a6*a43);
  a42=(a42+a43);
  a29=(a29-a42);
  if (res[1]!=0) res[1][32]=a29;
  a29=(a3*a28);
  a42=(a31*a41);
  a42=(a0*a42);
  a29=(a29+a42);
  a42=(a9*a33);
  a43=(a34*a41);
  a43=(a4*a43);
  a42=(a42-a43);
  a29=(a29+a42);
  a42=(a10*a11);
  a41=(a35*a41);
  a41=(a6*a41);
  a42=(a42-a41);
  a29=(a29+a42);
  if (res[1]!=0) res[1][33]=a29;
  if (res[1]!=0) res[1][34]=a11;
  a29=arg[1]? arg[1][38] : 0;
  a42=arg[1]? arg[1][39] : 0;
  a41=(a29+a42);
  a43=(a13*a28);
  a41=(a41-a43);
  a43=(a37*a28);
  a43=(a43/a39);
  a43=(a14*a43);
  a43=(a0*a43);
  a28=(a17*a28);
  a43=(a43+a28);
  a41=(a41-a43);
  a41=(a25*a41);
  if (res[1]!=0) res[1][35]=a41;
  a41=(a12*a33);
  a43=(a38*a33);
  a43=(a43/a40);
  a43=(a20*a43);
  a43=(a4*a43);
  a28=(a16*a33);
  a43=(a43+a28);
  a41=(a41-a43);
  a43=(a22*a11);
  a41=(a41-a43);
  a41=(a25*a41);
  if (res[1]!=0) res[1][36]=a41;
  a42=(a42-a29);
  a42=(a19*a42);
  a29=(a8*a11);
  a42=(a42-a29);
  a29=(a18*a11);
  a29=(a6*a29);
  a41=(a23*a11);
  a29=(a29+a41);
  a29=(a6*a29);
  a11=(a24*a11);
  a29=(a29+a11);
  a42=(a42-a29);
  a33=(a22*a33);
  a42=(a42-a33);
  a42=(a36*a42);
  if (res[1]!=0) res[1][37]=a42;
  if (res[1]!=0) res[1][38]=a27;
  if (res[1]!=0) res[1][39]=a27;
  a42=arg[1]? arg[1][43] : 0;
  a33=(a2*a42);
  a29=arg[1]? arg[1][42] : 0;
  a11=(a26*a29);
  a11=(a0*a11);
  a33=(a33-a11);
  a11=arg[1]? arg[1][44] : 0;
  a41=(a5*a11);
  a43=(a30*a29);
  a43=(a4*a43);
  a41=(a41+a43);
  a33=(a33-a41);
  a41=arg[1]? arg[1][45] : 0;
  a43=(a7*a41);
  a28=(a32*a29);
  a28=(a6*a28);
  a43=(a43+a28);
  a33=(a33-a43);
  if (res[1]!=0) res[1][40]=a33;
  a33=(a3*a42);
  a43=(a31*a29);
  a43=(a0*a43);
  a33=(a33+a43);
  a43=(a9*a11);
  a28=(a34*a29);
  a28=(a4*a28);
  a43=(a43-a28);
  a33=(a33+a43);
  a43=(a10*a41);
  a29=(a35*a29);
  a29=(a6*a29);
  a43=(a43-a29);
  a33=(a33+a43);
  if (res[1]!=0) res[1][41]=a33;
  if (res[1]!=0) res[1][42]=a41;
  a33=arg[1]? arg[1][46] : 0;
  a43=arg[1]? arg[1][47] : 0;
  a29=(a33+a43);
  a28=(a13*a42);
  a29=(a29-a28);
  a28=(a37*a42);
  a28=(a28/a39);
  a28=(a14*a28);
  a28=(a0*a28);
  a42=(a17*a42);
  a28=(a28+a42);
  a29=(a29-a28);
  a29=(a25*a29);
  if (res[1]!=0) res[1][43]=a29;
  a29=(a12*a11);
  a28=(a38*a11);
  a28=(a28/a40);
  a28=(a20*a28);
  a28=(a4*a28);
  a42=(a16*a11);
  a28=(a28+a42);
  a29=(a29-a28);
  a28=(a22*a41);
  a29=(a29-a28);
  a29=(a25*a29);
  if (res[1]!=0) res[1][44]=a29;
  a43=(a43-a33);
  a43=(a19*a43);
  a33=(a8*a41);
  a43=(a43-a33);
  a33=(a18*a41);
  a33=(a6*a33);
  a29=(a23*a41);
  a33=(a33+a29);
  a33=(a6*a33);
  a41=(a24*a41);
  a33=(a33+a41);
  a43=(a43-a33);
  a11=(a22*a11);
  a43=(a43-a11);
  a43=(a36*a43);
  if (res[1]!=0) res[1][45]=a43;
  if (res[1]!=0) res[1][46]=a27;
  if (res[1]!=0) res[1][47]=a27;
  a43=arg[1]? arg[1][51] : 0;
  a11=(a2*a43);
  a33=arg[1]? arg[1][50] : 0;
  a41=(a26*a33);
  a41=(a0*a41);
  a11=(a11-a41);
  a41=arg[1]? arg[1][52] : 0;
  a29=(a5*a41);
  a28=(a30*a33);
  a28=(a4*a28);
  a29=(a29+a28);
  a11=(a11-a29);
  a29=arg[1]? arg[1][53] : 0;
  a28=(a7*a29);
  a42=(a32*a33);
  a42=(a6*a42);
  a28=(a28+a42);
  a11=(a11-a28);
  if (res[1]!=0) res[1][48]=a11;
  a11=(a3*a43);
  a28=(a31*a33);
  a28=(a0*a28);
  a11=(a11+a28);
  a28=(a9*a41);
  a42=(a34*a33);
  a42=(a4*a42);
  a28=(a28-a42);
  a11=(a11+a28);
  a28=(a10*a29);
  a33=(a35*a33);
  a33=(a6*a33);
  a28=(a28-a33);
  a11=(a11+a28);
  if (res[1]!=0) res[1][49]=a11;
  if (res[1]!=0) res[1][50]=a29;
  a11=arg[1]? arg[1][54] : 0;
  a28=arg[1]? arg[1][55] : 0;
  a33=(a11+a28);
  a42=(a13*a43);
  a33=(a33-a42);
  a42=(a37*a43);
  a42=(a42/a39);
  a42=(a14*a42);
  a42=(a0*a42);
  a43=(a17*a43);
  a42=(a42+a43);
  a33=(a33-a42);
  a33=(a25*a33);
  if (res[1]!=0) res[1][51]=a33;
  a33=(a12*a41);
  a42=(a38*a41);
  a42=(a42/a40);
  a42=(a20*a42);
  a42=(a4*a42);
  a43=(a16*a41);
  a42=(a42+a43);
  a33=(a33-a42);
  a42=(a22*a29);
  a33=(a33-a42);
  a33=(a25*a33);
  if (res[1]!=0) res[1][52]=a33;
  a28=(a28-a11);
  a28=(a19*a28);
  a11=(a8*a29);
  a28=(a28-a11);
  a11=(a18*a29);
  a11=(a6*a11);
  a33=(a23*a29);
  a11=(a11+a33);
  a11=(a6*a11);
  a29=(a24*a29);
  a11=(a11+a29);
  a28=(a28-a11);
  a41=(a22*a41);
  a28=(a28-a41);
  a28=(a36*a28);
  if (res[1]!=0) res[1][53]=a28;
  if (res[1]!=0) res[1][54]=a27;
  if (res[1]!=0) res[1][55]=a27;
  a28=arg[1]? arg[1][59] : 0;
  a41=(a2*a28);
  a11=arg[1]? arg[1][58] : 0;
  a26=(a26*a11);
  a26=(a0*a26);
  a41=(a41-a26);
  a26=arg[1]? arg[1][60] : 0;
  a29=(a5*a26);
  a30=(a30*a11);
  a30=(a4*a30);
  a29=(a29+a30);
  a41=(a41-a29);
  a29=arg[1]? arg[1][61] : 0;
  a30=(a7*a29);
  a32=(a32*a11);
  a32=(a6*a32);
  a30=(a30+a32);
  a41=(a41-a30);
  if (res[1]!=0) res[1][56]=a41;
  a41=(a3*a28);
  a31=(a31*a11);
  a31=(a0*a31);
  a41=(a41+a31);
  a31=(a9*a26);
  a34=(a34*a11);
  a34=(a4*a34);
  a31=(a31-a34);
  a41=(a41+a31);
  a31=(a10*a29);
  a35=(a35*a11);
  a35=(a6*a35);
  a31=(a31-a35);
  a41=(a41+a31);
  if (res[1]!=0) res[1][57]=a41;
  if (res[1]!=0) res[1][58]=a29;
  a41=arg[1]? arg[1][62] : 0;
  a31=arg[1]? arg[1][63] : 0;
  a35=(a41+a31);
  a11=(a13*a28);
  a35=(a35-a11);
  a37=(a37*a28);
  a37=(a37/a39);
  a37=(a14*a37);
  a37=(a0*a37);
  a28=(a17*a28);
  a37=(a37+a28);
  a35=(a35-a37);
  a35=(a25*a35);
  if (res[1]!=0) res[1][59]=a35;
  a35=(a12*a26);
  a38=(a38*a26);
  a38=(a38/a40);
  a38=(a20*a38);
  a38=(a4*a38);
  a40=(a16*a26);
  a38=(a38+a40);
  a35=(a35-a38);
  a38=(a22*a29);
  a35=(a35-a38);
  a35=(a25*a35);
  if (res[1]!=0) res[1][60]=a35;
  a31=(a31-a41);
  a31=(a19*a31);
  a41=(a8*a29);
  a31=(a31-a41);
  a41=(a18*a29);
  a41=(a6*a41);
  a35=(a23*a29);
  a41=(a41+a35);
  a41=(a6*a41);
  a29=(a24*a29);
  a41=(a41+a29);
  a31=(a31-a41);
  a26=(a22*a26);
  a31=(a31-a26);
  a31=(a36*a31);
  if (res[1]!=0) res[1][61]=a31;
  if (res[1]!=0) res[1][62]=a27;
  if (res[1]!=0) res[1][63]=a27;
  a31=arg[2]? arg[2][3] : 0;
  a26=(a2*a31);
  a41=sin(a1);
  a29=arg[2]? arg[2][2] : 0;
  a35=(a41*a29);
  a35=(a0*a35);
  a26=(a26-a35);
  a35=arg[2]? arg[2][4] : 0;
  a38=(a5*a35);
  a40=cos(a1);
  a37=(a40*a29);
  a37=(a4*a37);
  a38=(a38+a37);
  a26=(a26-a38);
  a38=arg[2]? arg[2][5] : 0;
  a37=(a7*a38);
  a28=cos(a1);
  a39=(a28*a29);
  a39=(a6*a39);
  a37=(a37+a39);
  a26=(a26-a37);
  if (res[2]!=0) res[2][0]=a26;
  a26=(a3*a31);
  a37=cos(a1);
  a39=(a37*a29);
  a39=(a0*a39);
  a26=(a26+a39);
  a39=(a9*a35);
  a11=sin(a1);
  a34=(a11*a29);
  a34=(a4*a34);
  a39=(a39-a34);
  a26=(a26+a39);
  a39=(a10*a38);
  a1=sin(a1);
  a29=(a1*a29);
  a29=(a6*a29);
  a39=(a39-a29);
  a26=(a26+a39);
  if (res[2]!=0) res[2][1]=a26;
  if (res[2]!=0) res[2][2]=a38;
  a26=arg[2]? arg[2][6] : 0;
  a39=arg[2]? arg[2][7] : 0;
  a29=(a26+a39);
  a34=(a13*a31);
  a29=(a29-a34);
  a34=(a0+a0);
  a30=(a34*a31);
  a15=(a15+a15);
  a30=(a30/a15);
  a30=(a14*a30);
  a30=(a0*a30);
  a31=(a17*a31);
  a30=(a30+a31);
  a29=(a29-a30);
  a29=(a25*a29);
  if (res[2]!=0) res[2][3]=a29;
  a29=(a12*a35);
  a30=(a4+a4);
  a31=(a30*a35);
  a21=(a21+a21);
  a31=(a31/a21);
  a31=(a20*a31);
  a31=(a4*a31);
  a32=(a16*a35);
  a31=(a31+a32);
  a29=(a29-a31);
  a31=(a22*a38);
  a29=(a29-a31);
  a29=(a25*a29);
  if (res[2]!=0) res[2][4]=a29;
  a39=(a39-a26);
  a39=(a19*a39);
  a26=(a8*a38);
  a39=(a39-a26);
  a26=(a18*a38);
  a26=(a6*a26);
  a29=(a23*a38);
  a26=(a26+a29);
  a26=(a6*a26);
  a38=(a24*a38);
  a26=(a26+a38);
  a39=(a39-a26);
  a35=(a22*a35);
  a39=(a39-a35);
  a39=(a36*a39);
  if (res[2]!=0) res[2][5]=a39;
  a39=1.;
  if (res[2]!=0) res[2][6]=a39;
  if (res[2]!=0) res[2][7]=a27;
  a35=arg[2]? arg[2][11] : 0;
  a2=(a2*a35);
  a26=arg[2]? arg[2][10] : 0;
  a41=(a41*a26);
  a41=(a0*a41);
  a2=(a2-a41);
  a41=arg[2]? arg[2][12] : 0;
  a5=(a5*a41);
  a40=(a40*a26);
  a40=(a4*a40);
  a5=(a5+a40);
  a2=(a2-a5);
  a5=arg[2]? arg[2][13] : 0;
  a7=(a7*a5);
  a28=(a28*a26);
  a28=(a6*a28);
  a7=(a7+a28);
  a2=(a2-a7);
  if (res[2]!=0) res[2][8]=a2;
  a3=(a3*a35);
  a37=(a37*a26);
  a37=(a0*a37);
  a3=(a3+a37);
  a9=(a9*a41);
  a11=(a11*a26);
  a11=(a4*a11);
  a9=(a9-a11);
  a3=(a3+a9);
  a10=(a10*a5);
  a1=(a1*a26);
  a1=(a6*a1);
  a10=(a10-a1);
  a3=(a3+a10);
  if (res[2]!=0) res[2][9]=a3;
  if (res[2]!=0) res[2][10]=a5;
  a3=arg[2]? arg[2][14] : 0;
  a10=arg[2]? arg[2][15] : 0;
  a1=(a3+a10);
  a13=(a13*a35);
  a1=(a1-a13);
  a34=(a34*a35);
  a34=(a34/a15);
  a14=(a14*a34);
  a0=(a0*a14);
  a17=(a17*a35);
  a0=(a0+a17);
  a1=(a1-a0);
  a1=(a25*a1);
  if (res[2]!=0) res[2][11]=a1;
  a12=(a12*a41);
  a30=(a30*a41);
  a30=(a30/a21);
  a20=(a20*a30);
  a4=(a4*a20);
  a16=(a16*a41);
  a4=(a4+a16);
  a12=(a12-a4);
  a4=(a22*a5);
  a12=(a12-a4);
  a25=(a25*a12);
  if (res[2]!=0) res[2][12]=a25;
  a10=(a10-a3);
  a19=(a19*a10);
  a8=(a8*a5);
  a19=(a19-a8);
  a18=(a18*a5);
  a18=(a6*a18);
  a23=(a23*a5);
  a18=(a18+a23);
  a6=(a6*a18);
  a24=(a24*a5);
  a6=(a6+a24);
  a19=(a19-a6);
  a22=(a22*a41);
  a19=(a19-a22);
  a36=(a36*a19);
  if (res[2]!=0) res[2][13]=a36;
  if (res[2]!=0) res[2][14]=a27;
  if (res[2]!=0) res[2][15]=a39;
  return 0;
}

CASADI_SYMBOL_EXPORT int heron_expl_vde_forw(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int heron_expl_vde_forw_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int heron_expl_vde_forw_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void heron_expl_vde_forw_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int heron_expl_vde_forw_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void heron_expl_vde_forw_release(int mem) {
}

CASADI_SYMBOL_EXPORT void heron_expl_vde_forw_incref(void) {
}

CASADI_SYMBOL_EXPORT void heron_expl_vde_forw_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int heron_expl_vde_forw_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int heron_expl_vde_forw_n_out(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_real heron_expl_vde_forw_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* heron_expl_vde_forw_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* heron_expl_vde_forw_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* heron_expl_vde_forw_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s3;
    case 4: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* heron_expl_vde_forw_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int heron_expl_vde_forw_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 3;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
