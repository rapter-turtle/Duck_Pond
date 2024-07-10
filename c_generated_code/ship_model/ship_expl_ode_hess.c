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
  #define CASADI_PREFIX(ID) ship_expl_ode_hess_ ## ID
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
static const casadi_int casadi_s1[59] = {7, 7, 0, 7, 14, 21, 28, 35, 42, 49, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6};
static const casadi_int casadi_s2[19] = {7, 2, 0, 7, 14, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6};
static const casadi_int casadi_s3[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s4[29] = {25, 1, 0, 25, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
static const casadi_int casadi_s5[13] = {9, 1, 0, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8};
static const casadi_int casadi_s6[49] = {45, 1, 0, 45, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44};

/* ship_expl_ode_hess:(i0[7],i1[7x7],i2[7x2],i3[7],i4[2],i5[25])->(o0[9],o1[45]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a6, a7, a8, a9;
  a0=0.;
  if (res[0]!=0) res[0][0]=a0;
  if (res[0]!=0) res[0][1]=a0;
  a0=arg[0]? arg[0][2] : 0;
  a1=cos(a0);
  a2=arg[0]? arg[0][3] : 0;
  a3=arg[3]? arg[3][1] : 0;
  a4=(a2*a3);
  a5=(a1*a4);
  a6=sin(a0);
  a7=arg[3]? arg[3][0] : 0;
  a8=(a2*a7);
  a9=(a6*a8);
  a5=(a5-a9);
  if (res[0]!=0) res[0][2]=a5;
  a5=sin(a0);
  a5=(a5*a3);
  a9=20.;
  a10=50.;
  a11=casadi_sq(a2);
  a12=1.0000000000000001e-05;
  a11=(a11+a12);
  a11=sqrt(a11);
  a13=(a10*a11);
  a9=(a9+a13);
  a13=1.0000000000000000e-02;
  a14=arg[3]? arg[3][3] : 0;
  a13=(a13*a14);
  a9=(a9*a13);
  a14=(a2+a2);
  a15=(a2*a13);
  a15=(a10*a15);
  a16=(a11+a11);
  a15=(a15/a16);
  a17=(a14*a15);
  a9=(a9+a17);
  a5=(a5-a9);
  a9=cos(a0);
  a9=(a9*a7);
  a5=(a5+a9);
  if (res[0]!=0) res[0][3]=a5;
  a5=arg[3]? arg[3][2] : 0;
  a9=25.;
  a17=100.;
  a18=arg[0]? arg[0][4] : 0;
  a19=casadi_sq(a18);
  a19=(a19+a12);
  a19=sqrt(a19);
  a12=(a17*a19);
  a9=(a9+a12);
  a12=2.0000000000000000e-03;
  a20=arg[3]? arg[3][4] : 0;
  a12=(a12*a20);
  a9=(a9*a12);
  a20=(a18+a18);
  a21=(a18*a12);
  a21=(a17*a21);
  a22=(a19+a19);
  a21=(a21/a22);
  a23=(a20*a21);
  a9=(a9+a23);
  a5=(a5-a9);
  if (res[0]!=0) res[0][4]=a5;
  if (res[0]!=0) res[0][5]=a13;
  if (res[0]!=0) res[0][6]=a12;
  a5=arg[3]? arg[3][5] : 0;
  if (res[0]!=0) res[0][7]=a5;
  a5=arg[3]? arg[3][6] : 0;
  if (res[0]!=0) res[0][8]=a5;
  a5=arg[1]? arg[1][2] : 0;
  a9=arg[1]? arg[1][3] : 0;
  a23=(a3*a9);
  a23=(a1*a23);
  a24=sin(a0);
  a25=(a24*a5);
  a25=(a4*a25);
  a23=(a23-a25);
  a25=cos(a0);
  a26=(a25*a5);
  a26=(a8*a26);
  a27=(a7*a9);
  a27=(a6*a27);
  a26=(a26+a27);
  a23=(a23-a26);
  a26=(a5*a23);
  a27=cos(a0);
  a28=(a27*a5);
  a28=(a3*a28);
  a2=(a2+a2);
  a29=(a2*a9);
  a11=(a11+a11);
  a29=(a29/a11);
  a30=(a10*a29);
  a30=(a13*a30);
  a31=(a9+a9);
  a31=(a15*a31);
  a32=(a13*a9);
  a32=(a10*a32);
  a32=(a32/a16);
  a33=(a15/a16);
  a29=(a29+a29);
  a29=(a33*a29);
  a32=(a32-a29);
  a32=(a14*a32);
  a31=(a31+a32);
  a30=(a30+a31);
  a28=(a28-a30);
  a0=sin(a0);
  a5=(a0*a5);
  a5=(a7*a5);
  a28=(a28-a5);
  a9=(a9*a28);
  a26=(a26+a9);
  a9=arg[1]? arg[1][4] : 0;
  a18=(a18+a18);
  a5=(a18*a9);
  a19=(a19+a19);
  a5=(a5/a19);
  a30=(a17*a5);
  a30=(a12*a30);
  a31=(a9+a9);
  a31=(a21*a31);
  a32=(a12*a9);
  a32=(a17*a32);
  a32=(a32/a22);
  a29=(a21/a22);
  a5=(a5+a5);
  a5=(a29*a5);
  a32=(a32-a5);
  a32=(a20*a32);
  a31=(a31+a32);
  a30=(a30+a31);
  a9=(a9*a30);
  a26=(a26-a9);
  if (res[1]!=0) res[1][0]=a26;
  a26=arg[1]? arg[1][9] : 0;
  a9=(a26*a23);
  a31=arg[1]? arg[1][10] : 0;
  a32=(a31*a28);
  a9=(a9+a32);
  a32=arg[1]? arg[1][11] : 0;
  a5=(a32*a30);
  a9=(a9-a5);
  if (res[1]!=0) res[1][1]=a9;
  a9=arg[1]? arg[1][16] : 0;
  a5=(a9*a23);
  a34=arg[1]? arg[1][17] : 0;
  a35=(a34*a28);
  a5=(a5+a35);
  a35=arg[1]? arg[1][18] : 0;
  a36=(a35*a30);
  a5=(a5-a36);
  if (res[1]!=0) res[1][2]=a5;
  a5=arg[1]? arg[1][23] : 0;
  a36=(a5*a23);
  a37=arg[1]? arg[1][24] : 0;
  a38=(a37*a28);
  a36=(a36+a38);
  a38=arg[1]? arg[1][25] : 0;
  a39=(a38*a30);
  a36=(a36-a39);
  if (res[1]!=0) res[1][3]=a36;
  a36=arg[1]? arg[1][30] : 0;
  a39=(a36*a23);
  a40=arg[1]? arg[1][31] : 0;
  a41=(a40*a28);
  a39=(a39+a41);
  a41=arg[1]? arg[1][32] : 0;
  a42=(a41*a30);
  a39=(a39-a42);
  if (res[1]!=0) res[1][4]=a39;
  a39=arg[1]? arg[1][37] : 0;
  a42=(a39*a23);
  a43=arg[1]? arg[1][38] : 0;
  a44=(a43*a28);
  a42=(a42+a44);
  a44=arg[1]? arg[1][39] : 0;
  a45=(a44*a30);
  a42=(a42-a45);
  if (res[1]!=0) res[1][5]=a42;
  a42=arg[1]? arg[1][44] : 0;
  a45=(a42*a23);
  a46=arg[1]? arg[1][45] : 0;
  a47=(a46*a28);
  a45=(a45+a47);
  a47=arg[1]? arg[1][46] : 0;
  a48=(a47*a30);
  a45=(a45-a48);
  if (res[1]!=0) res[1][6]=a45;
  a45=arg[2]? arg[2][2] : 0;
  a48=(a45*a23);
  a49=arg[2]? arg[2][3] : 0;
  a50=(a49*a28);
  a48=(a48+a50);
  a50=arg[2]? arg[2][4] : 0;
  a51=(a50*a30);
  a48=(a48-a51);
  if (res[1]!=0) res[1][7]=a48;
  a48=arg[2]? arg[2][9] : 0;
  a23=(a48*a23);
  a51=arg[2]? arg[2][10] : 0;
  a28=(a51*a28);
  a23=(a23+a28);
  a28=arg[2]? arg[2][11] : 0;
  a30=(a28*a30);
  a23=(a23-a30);
  if (res[1]!=0) res[1][8]=a23;
  a23=(a3*a31);
  a23=(a1*a23);
  a30=(a24*a26);
  a30=(a4*a30);
  a23=(a23-a30);
  a30=(a25*a26);
  a30=(a8*a30);
  a52=(a7*a31);
  a52=(a6*a52);
  a30=(a30+a52);
  a23=(a23-a30);
  a30=(a26*a23);
  a52=(a27*a26);
  a52=(a3*a52);
  a53=(a2*a31);
  a53=(a53/a11);
  a54=(a10*a53);
  a54=(a13*a54);
  a55=(a31+a31);
  a55=(a15*a55);
  a56=(a13*a31);
  a56=(a10*a56);
  a56=(a56/a16);
  a53=(a53+a53);
  a53=(a33*a53);
  a56=(a56-a53);
  a56=(a14*a56);
  a55=(a55+a56);
  a54=(a54+a55);
  a52=(a52-a54);
  a26=(a0*a26);
  a26=(a7*a26);
  a52=(a52-a26);
  a31=(a31*a52);
  a30=(a30+a31);
  a31=(a18*a32);
  a31=(a31/a19);
  a26=(a17*a31);
  a26=(a12*a26);
  a54=(a32+a32);
  a54=(a21*a54);
  a55=(a12*a32);
  a55=(a17*a55);
  a55=(a55/a22);
  a31=(a31+a31);
  a31=(a29*a31);
  a55=(a55-a31);
  a55=(a20*a55);
  a54=(a54+a55);
  a26=(a26+a54);
  a32=(a32*a26);
  a30=(a30-a32);
  if (res[1]!=0) res[1][9]=a30;
  a30=(a9*a23);
  a32=(a34*a52);
  a30=(a30+a32);
  a32=(a35*a26);
  a30=(a30-a32);
  if (res[1]!=0) res[1][10]=a30;
  a30=(a5*a23);
  a32=(a37*a52);
  a30=(a30+a32);
  a32=(a38*a26);
  a30=(a30-a32);
  if (res[1]!=0) res[1][11]=a30;
  a30=(a36*a23);
  a32=(a40*a52);
  a30=(a30+a32);
  a32=(a41*a26);
  a30=(a30-a32);
  if (res[1]!=0) res[1][12]=a30;
  a30=(a39*a23);
  a32=(a43*a52);
  a30=(a30+a32);
  a32=(a44*a26);
  a30=(a30-a32);
  if (res[1]!=0) res[1][13]=a30;
  a30=(a42*a23);
  a32=(a46*a52);
  a30=(a30+a32);
  a32=(a47*a26);
  a30=(a30-a32);
  if (res[1]!=0) res[1][14]=a30;
  a30=(a45*a23);
  a32=(a49*a52);
  a30=(a30+a32);
  a32=(a50*a26);
  a30=(a30-a32);
  if (res[1]!=0) res[1][15]=a30;
  a23=(a48*a23);
  a52=(a51*a52);
  a23=(a23+a52);
  a26=(a28*a26);
  a23=(a23-a26);
  if (res[1]!=0) res[1][16]=a23;
  a23=(a3*a34);
  a23=(a1*a23);
  a26=(a24*a9);
  a26=(a4*a26);
  a23=(a23-a26);
  a26=(a25*a9);
  a26=(a8*a26);
  a52=(a7*a34);
  a52=(a6*a52);
  a26=(a26+a52);
  a23=(a23-a26);
  a26=(a9*a23);
  a52=(a27*a9);
  a52=(a3*a52);
  a30=(a2*a34);
  a30=(a30/a11);
  a32=(a10*a30);
  a32=(a13*a32);
  a54=(a34+a34);
  a54=(a15*a54);
  a55=(a13*a34);
  a55=(a10*a55);
  a55=(a55/a16);
  a30=(a30+a30);
  a30=(a33*a30);
  a55=(a55-a30);
  a55=(a14*a55);
  a54=(a54+a55);
  a32=(a32+a54);
  a52=(a52-a32);
  a9=(a0*a9);
  a9=(a7*a9);
  a52=(a52-a9);
  a34=(a34*a52);
  a26=(a26+a34);
  a34=(a18*a35);
  a34=(a34/a19);
  a9=(a17*a34);
  a9=(a12*a9);
  a32=(a35+a35);
  a32=(a21*a32);
  a54=(a12*a35);
  a54=(a17*a54);
  a54=(a54/a22);
  a34=(a34+a34);
  a34=(a29*a34);
  a54=(a54-a34);
  a54=(a20*a54);
  a32=(a32+a54);
  a9=(a9+a32);
  a35=(a35*a9);
  a26=(a26-a35);
  if (res[1]!=0) res[1][17]=a26;
  a26=(a5*a23);
  a35=(a37*a52);
  a26=(a26+a35);
  a35=(a38*a9);
  a26=(a26-a35);
  if (res[1]!=0) res[1][18]=a26;
  a26=(a36*a23);
  a35=(a40*a52);
  a26=(a26+a35);
  a35=(a41*a9);
  a26=(a26-a35);
  if (res[1]!=0) res[1][19]=a26;
  a26=(a39*a23);
  a35=(a43*a52);
  a26=(a26+a35);
  a35=(a44*a9);
  a26=(a26-a35);
  if (res[1]!=0) res[1][20]=a26;
  a26=(a42*a23);
  a35=(a46*a52);
  a26=(a26+a35);
  a35=(a47*a9);
  a26=(a26-a35);
  if (res[1]!=0) res[1][21]=a26;
  a26=(a45*a23);
  a35=(a49*a52);
  a26=(a26+a35);
  a35=(a50*a9);
  a26=(a26-a35);
  if (res[1]!=0) res[1][22]=a26;
  a23=(a48*a23);
  a52=(a51*a52);
  a23=(a23+a52);
  a9=(a28*a9);
  a23=(a23-a9);
  if (res[1]!=0) res[1][23]=a23;
  a23=(a3*a37);
  a23=(a1*a23);
  a9=(a24*a5);
  a9=(a4*a9);
  a23=(a23-a9);
  a9=(a25*a5);
  a9=(a8*a9);
  a52=(a7*a37);
  a52=(a6*a52);
  a9=(a9+a52);
  a23=(a23-a9);
  a9=(a5*a23);
  a52=(a27*a5);
  a52=(a3*a52);
  a26=(a2*a37);
  a26=(a26/a11);
  a35=(a10*a26);
  a35=(a13*a35);
  a32=(a37+a37);
  a32=(a15*a32);
  a54=(a13*a37);
  a54=(a10*a54);
  a54=(a54/a16);
  a26=(a26+a26);
  a26=(a33*a26);
  a54=(a54-a26);
  a54=(a14*a54);
  a32=(a32+a54);
  a35=(a35+a32);
  a52=(a52-a35);
  a5=(a0*a5);
  a5=(a7*a5);
  a52=(a52-a5);
  a37=(a37*a52);
  a9=(a9+a37);
  a37=(a18*a38);
  a37=(a37/a19);
  a5=(a17*a37);
  a5=(a12*a5);
  a35=(a38+a38);
  a35=(a21*a35);
  a32=(a12*a38);
  a32=(a17*a32);
  a32=(a32/a22);
  a37=(a37+a37);
  a37=(a29*a37);
  a32=(a32-a37);
  a32=(a20*a32);
  a35=(a35+a32);
  a5=(a5+a35);
  a38=(a38*a5);
  a9=(a9-a38);
  if (res[1]!=0) res[1][24]=a9;
  a9=(a36*a23);
  a38=(a40*a52);
  a9=(a9+a38);
  a38=(a41*a5);
  a9=(a9-a38);
  if (res[1]!=0) res[1][25]=a9;
  a9=(a39*a23);
  a38=(a43*a52);
  a9=(a9+a38);
  a38=(a44*a5);
  a9=(a9-a38);
  if (res[1]!=0) res[1][26]=a9;
  a9=(a42*a23);
  a38=(a46*a52);
  a9=(a9+a38);
  a38=(a47*a5);
  a9=(a9-a38);
  if (res[1]!=0) res[1][27]=a9;
  a9=(a45*a23);
  a38=(a49*a52);
  a9=(a9+a38);
  a38=(a50*a5);
  a9=(a9-a38);
  if (res[1]!=0) res[1][28]=a9;
  a23=(a48*a23);
  a52=(a51*a52);
  a23=(a23+a52);
  a5=(a28*a5);
  a23=(a23-a5);
  if (res[1]!=0) res[1][29]=a23;
  a23=(a3*a40);
  a23=(a1*a23);
  a5=(a24*a36);
  a5=(a4*a5);
  a23=(a23-a5);
  a5=(a25*a36);
  a5=(a8*a5);
  a52=(a7*a40);
  a52=(a6*a52);
  a5=(a5+a52);
  a23=(a23-a5);
  a5=(a36*a23);
  a52=(a27*a36);
  a52=(a3*a52);
  a9=(a2*a40);
  a9=(a9/a11);
  a38=(a10*a9);
  a38=(a13*a38);
  a35=(a40+a40);
  a35=(a15*a35);
  a32=(a13*a40);
  a32=(a10*a32);
  a32=(a32/a16);
  a9=(a9+a9);
  a9=(a33*a9);
  a32=(a32-a9);
  a32=(a14*a32);
  a35=(a35+a32);
  a38=(a38+a35);
  a52=(a52-a38);
  a36=(a0*a36);
  a36=(a7*a36);
  a52=(a52-a36);
  a40=(a40*a52);
  a5=(a5+a40);
  a40=(a18*a41);
  a40=(a40/a19);
  a36=(a17*a40);
  a36=(a12*a36);
  a38=(a41+a41);
  a38=(a21*a38);
  a35=(a12*a41);
  a35=(a17*a35);
  a35=(a35/a22);
  a40=(a40+a40);
  a40=(a29*a40);
  a35=(a35-a40);
  a35=(a20*a35);
  a38=(a38+a35);
  a36=(a36+a38);
  a41=(a41*a36);
  a5=(a5-a41);
  if (res[1]!=0) res[1][30]=a5;
  a5=(a39*a23);
  a41=(a43*a52);
  a5=(a5+a41);
  a41=(a44*a36);
  a5=(a5-a41);
  if (res[1]!=0) res[1][31]=a5;
  a5=(a42*a23);
  a41=(a46*a52);
  a5=(a5+a41);
  a41=(a47*a36);
  a5=(a5-a41);
  if (res[1]!=0) res[1][32]=a5;
  a5=(a45*a23);
  a41=(a49*a52);
  a5=(a5+a41);
  a41=(a50*a36);
  a5=(a5-a41);
  if (res[1]!=0) res[1][33]=a5;
  a23=(a48*a23);
  a52=(a51*a52);
  a23=(a23+a52);
  a36=(a28*a36);
  a23=(a23-a36);
  if (res[1]!=0) res[1][34]=a23;
  a23=(a3*a43);
  a23=(a1*a23);
  a36=(a24*a39);
  a36=(a4*a36);
  a23=(a23-a36);
  a36=(a25*a39);
  a36=(a8*a36);
  a52=(a7*a43);
  a52=(a6*a52);
  a36=(a36+a52);
  a23=(a23-a36);
  a36=(a39*a23);
  a52=(a27*a39);
  a52=(a3*a52);
  a5=(a2*a43);
  a5=(a5/a11);
  a41=(a10*a5);
  a41=(a13*a41);
  a38=(a43+a43);
  a38=(a15*a38);
  a35=(a13*a43);
  a35=(a10*a35);
  a35=(a35/a16);
  a5=(a5+a5);
  a5=(a33*a5);
  a35=(a35-a5);
  a35=(a14*a35);
  a38=(a38+a35);
  a41=(a41+a38);
  a52=(a52-a41);
  a39=(a0*a39);
  a39=(a7*a39);
  a52=(a52-a39);
  a43=(a43*a52);
  a36=(a36+a43);
  a43=(a18*a44);
  a43=(a43/a19);
  a39=(a17*a43);
  a39=(a12*a39);
  a41=(a44+a44);
  a41=(a21*a41);
  a38=(a12*a44);
  a38=(a17*a38);
  a38=(a38/a22);
  a43=(a43+a43);
  a43=(a29*a43);
  a38=(a38-a43);
  a38=(a20*a38);
  a41=(a41+a38);
  a39=(a39+a41);
  a44=(a44*a39);
  a36=(a36-a44);
  if (res[1]!=0) res[1][35]=a36;
  a36=(a42*a23);
  a44=(a46*a52);
  a36=(a36+a44);
  a44=(a47*a39);
  a36=(a36-a44);
  if (res[1]!=0) res[1][36]=a36;
  a36=(a45*a23);
  a44=(a49*a52);
  a36=(a36+a44);
  a44=(a50*a39);
  a36=(a36-a44);
  if (res[1]!=0) res[1][37]=a36;
  a23=(a48*a23);
  a52=(a51*a52);
  a23=(a23+a52);
  a39=(a28*a39);
  a23=(a23-a39);
  if (res[1]!=0) res[1][38]=a23;
  a23=(a3*a46);
  a23=(a1*a23);
  a39=(a24*a42);
  a39=(a4*a39);
  a23=(a23-a39);
  a39=(a25*a42);
  a39=(a8*a39);
  a52=(a7*a46);
  a52=(a6*a52);
  a39=(a39+a52);
  a23=(a23-a39);
  a39=(a42*a23);
  a52=(a27*a42);
  a52=(a3*a52);
  a36=(a2*a46);
  a36=(a36/a11);
  a44=(a10*a36);
  a44=(a13*a44);
  a41=(a46+a46);
  a41=(a15*a41);
  a38=(a13*a46);
  a38=(a10*a38);
  a38=(a38/a16);
  a36=(a36+a36);
  a36=(a33*a36);
  a38=(a38-a36);
  a38=(a14*a38);
  a41=(a41+a38);
  a44=(a44+a41);
  a52=(a52-a44);
  a42=(a0*a42);
  a42=(a7*a42);
  a52=(a52-a42);
  a46=(a46*a52);
  a39=(a39+a46);
  a46=(a18*a47);
  a46=(a46/a19);
  a42=(a17*a46);
  a42=(a12*a42);
  a44=(a47+a47);
  a44=(a21*a44);
  a41=(a12*a47);
  a41=(a17*a41);
  a41=(a41/a22);
  a46=(a46+a46);
  a46=(a29*a46);
  a41=(a41-a46);
  a41=(a20*a41);
  a44=(a44+a41);
  a42=(a42+a44);
  a47=(a47*a42);
  a39=(a39-a47);
  if (res[1]!=0) res[1][39]=a39;
  a39=(a45*a23);
  a47=(a49*a52);
  a39=(a39+a47);
  a47=(a50*a42);
  a39=(a39-a47);
  if (res[1]!=0) res[1][40]=a39;
  a23=(a48*a23);
  a52=(a51*a52);
  a23=(a23+a52);
  a42=(a28*a42);
  a23=(a23-a42);
  if (res[1]!=0) res[1][41]=a23;
  a23=(a3*a49);
  a23=(a1*a23);
  a42=(a24*a45);
  a42=(a4*a42);
  a23=(a23-a42);
  a42=(a25*a45);
  a42=(a8*a42);
  a52=(a7*a49);
  a52=(a6*a52);
  a42=(a42+a52);
  a23=(a23-a42);
  a42=(a45*a23);
  a52=(a27*a45);
  a52=(a3*a52);
  a39=(a2*a49);
  a39=(a39/a11);
  a47=(a10*a39);
  a47=(a13*a47);
  a44=(a49+a49);
  a44=(a15*a44);
  a41=(a13*a49);
  a41=(a10*a41);
  a41=(a41/a16);
  a39=(a39+a39);
  a39=(a33*a39);
  a41=(a41-a39);
  a41=(a14*a41);
  a44=(a44+a41);
  a47=(a47+a44);
  a52=(a52-a47);
  a45=(a0*a45);
  a45=(a7*a45);
  a52=(a52-a45);
  a49=(a49*a52);
  a42=(a42+a49);
  a49=(a18*a50);
  a49=(a49/a19);
  a45=(a17*a49);
  a45=(a12*a45);
  a47=(a50+a50);
  a47=(a21*a47);
  a44=(a12*a50);
  a44=(a17*a44);
  a44=(a44/a22);
  a49=(a49+a49);
  a49=(a29*a49);
  a44=(a44-a49);
  a44=(a20*a44);
  a47=(a47+a44);
  a45=(a45+a47);
  a50=(a50*a45);
  a42=(a42-a50);
  if (res[1]!=0) res[1][42]=a42;
  a23=(a48*a23);
  a52=(a51*a52);
  a23=(a23+a52);
  a45=(a28*a45);
  a23=(a23-a45);
  if (res[1]!=0) res[1][43]=a23;
  a23=(a3*a51);
  a1=(a1*a23);
  a24=(a24*a48);
  a4=(a4*a24);
  a1=(a1-a4);
  a25=(a25*a48);
  a8=(a8*a25);
  a25=(a7*a51);
  a6=(a6*a25);
  a8=(a8+a6);
  a1=(a1-a8);
  a1=(a48*a1);
  a27=(a27*a48);
  a3=(a3*a27);
  a2=(a2*a51);
  a2=(a2/a11);
  a11=(a10*a2);
  a11=(a13*a11);
  a27=(a51+a51);
  a15=(a15*a27);
  a13=(a13*a51);
  a10=(a10*a13);
  a10=(a10/a16);
  a2=(a2+a2);
  a33=(a33*a2);
  a10=(a10-a33);
  a14=(a14*a10);
  a15=(a15+a14);
  a11=(a11+a15);
  a3=(a3-a11);
  a0=(a0*a48);
  a7=(a7*a0);
  a3=(a3-a7);
  a51=(a51*a3);
  a1=(a1+a51);
  a18=(a18*a28);
  a18=(a18/a19);
  a19=(a17*a18);
  a19=(a12*a19);
  a51=(a28+a28);
  a21=(a21*a51);
  a12=(a12*a28);
  a17=(a17*a12);
  a17=(a17/a22);
  a18=(a18+a18);
  a29=(a29*a18);
  a17=(a17-a29);
  a20=(a20*a17);
  a21=(a21+a20);
  a19=(a19+a21);
  a28=(a28*a19);
  a1=(a1-a28);
  if (res[1]!=0) res[1][44]=a1;
  return 0;
}

CASADI_SYMBOL_EXPORT int ship_expl_ode_hess(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int ship_expl_ode_hess_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int ship_expl_ode_hess_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void ship_expl_ode_hess_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int ship_expl_ode_hess_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void ship_expl_ode_hess_release(int mem) {
}

CASADI_SYMBOL_EXPORT void ship_expl_ode_hess_incref(void) {
}

CASADI_SYMBOL_EXPORT void ship_expl_ode_hess_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int ship_expl_ode_hess_n_in(void) { return 6;}

CASADI_SYMBOL_EXPORT casadi_int ship_expl_ode_hess_n_out(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_real ship_expl_ode_hess_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* ship_expl_ode_hess_name_in(casadi_int i) {
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

CASADI_SYMBOL_EXPORT const char* ship_expl_ode_hess_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* ship_expl_ode_hess_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s0;
    case 4: return casadi_s3;
    case 5: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* ship_expl_ode_hess_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s5;
    case 1: return casadi_s6;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int ship_expl_ode_hess_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 6;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif