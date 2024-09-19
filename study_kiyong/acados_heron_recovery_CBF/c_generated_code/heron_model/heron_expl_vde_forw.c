/* This file was automatically generated by CasADi 3.6.3.
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
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a5, a6, a7, a8, a9;
  a0=arg[0]? arg[0][3] : 0;
  a1=arg[0]? arg[0][2] : 0;
  a2=cos(a1);
  a3=(a0*a2);
  a4=arg[0]? arg[0][4] : 0;
  a5=sin(a1);
  a6=(a4*a5);
  a3=(a3-a6);
  if (res[0]!=0) res[0][0]=a3;
  a3=sin(a1);
  a6=(a0*a3);
  a7=cos(a1);
  a8=(a4*a7);
  a6=(a6+a8);
  if (res[0]!=0) res[0][1]=a6;
  a6=arg[0]? arg[0][5] : 0;
  if (res[0]!=0) res[0][2]=a6;
  a8=arg[0]? arg[0][6] : 0;
  a9=arg[0]? arg[0][7] : 0;
  a10=(a8+a9);
  a11=8.9148999999999994e+00;
  a12=(a11*a0);
  a10=(a10-a12);
  a12=1.1210100000000001e+01;
  a13=casadi_sq(a0);
  a14=1.0000000000000001e-05;
  a13=(a13+a14);
  a13=sqrt(a13);
  a15=(a12*a13);
  a16=(a15*a0);
  a10=(a10-a16);
  a16=3.7758000000000003e+01;
  a10=(a10/a16);
  a17=arg[4]? arg[4][0] : 0;
  a10=(a10+a17);
  if (res[0]!=0) res[0][3]=a10;
  a10=-15.;
  a17=(a10*a4);
  a18=3.;
  a19=casadi_sq(a4);
  a19=(a19+a14);
  a19=sqrt(a19);
  a14=(a18*a19);
  a20=(a14*a4);
  a17=(a17-a20);
  a20=6.;
  a21=(a20*a6);
  a17=(a17-a21);
  a17=(a17/a16);
  a16=arg[4]? arg[4][1] : 0;
  a17=(a17+a16);
  if (res[0]!=0) res[0][4]=a17;
  a17=2.9999999999999999e-01;
  a9=(a9-a8);
  a9=(a17*a9);
  a8=1.6954200000000000e+01;
  a16=(a8*a6);
  a9=(a9-a16);
  a16=1.2896599999999999e+01;
  a21=(a16*a6);
  a22=(a21*a6);
  a23=(a22*a6);
  a9=(a9-a23);
  a23=(a20*a4);
  a9=(a9-a23);
  a23=1.8350000000000001e+01;
  a9=(a9/a23);
  a23=arg[4]? arg[4][2] : 0;
  a9=(a9+a23);
  if (res[0]!=0) res[0][5]=a9;
  a9=arg[3]? arg[3][0] : 0;
  if (res[0]!=0) res[0][6]=a9;
  a9=arg[3]? arg[3][1] : 0;
  if (res[0]!=0) res[0][7]=a9;
  a9=arg[1]? arg[1][3] : 0;
  a23=(a2*a9);
  a24=sin(a1);
  a25=arg[1]? arg[1][2] : 0;
  a26=(a24*a25);
  a26=(a0*a26);
  a23=(a23-a26);
  a26=arg[1]? arg[1][4] : 0;
  a27=(a5*a26);
  a28=cos(a1);
  a29=(a28*a25);
  a29=(a4*a29);
  a27=(a27+a29);
  a23=(a23-a27);
  if (res[1]!=0) res[1][0]=a23;
  a23=(a3*a9);
  a27=cos(a1);
  a29=(a27*a25);
  a29=(a0*a29);
  a23=(a23+a29);
  a29=(a7*a26);
  a30=sin(a1);
  a25=(a30*a25);
  a25=(a4*a25);
  a29=(a29-a25);
  a23=(a23+a29);
  if (res[1]!=0) res[1][1]=a23;
  a23=arg[1]? arg[1][5] : 0;
  if (res[1]!=0) res[1][2]=a23;
  a29=2.6484453625721698e-02;
  a25=arg[1]? arg[1][6] : 0;
  a31=arg[1]? arg[1][7] : 0;
  a32=(a25+a31);
  a33=(a11*a9);
  a32=(a32-a33);
  a33=(a0+a0);
  a34=(a33*a9);
  a35=(a13+a13);
  a34=(a34/a35);
  a34=(a12*a34);
  a34=(a0*a34);
  a9=(a15*a9);
  a34=(a34+a9);
  a32=(a32-a34);
  a32=(a29*a32);
  if (res[1]!=0) res[1][3]=a32;
  a32=(a10*a26);
  a34=(a4+a4);
  a9=(a34*a26);
  a36=(a19+a19);
  a9=(a9/a36);
  a9=(a18*a9);
  a9=(a4*a9);
  a37=(a14*a26);
  a9=(a9+a37);
  a32=(a32-a9);
  a9=(a20*a23);
  a32=(a32-a9);
  a32=(a29*a32);
  if (res[1]!=0) res[1][4]=a32;
  a32=5.4495912806539502e-02;
  a31=(a31-a25);
  a31=(a17*a31);
  a25=(a8*a23);
  a31=(a31-a25);
  a25=(a16*a23);
  a25=(a6*a25);
  a9=(a21*a23);
  a25=(a25+a9);
  a25=(a6*a25);
  a23=(a22*a23);
  a25=(a25+a23);
  a31=(a31-a25);
  a26=(a20*a26);
  a31=(a31-a26);
  a31=(a32*a31);
  if (res[1]!=0) res[1][5]=a31;
  a31=0.;
  if (res[1]!=0) res[1][6]=a31;
  if (res[1]!=0) res[1][7]=a31;
  a26=arg[1]? arg[1][11] : 0;
  a25=(a2*a26);
  a23=arg[1]? arg[1][10] : 0;
  a9=(a24*a23);
  a9=(a0*a9);
  a25=(a25-a9);
  a9=arg[1]? arg[1][12] : 0;
  a37=(a5*a9);
  a38=(a28*a23);
  a38=(a4*a38);
  a37=(a37+a38);
  a25=(a25-a37);
  if (res[1]!=0) res[1][8]=a25;
  a25=(a3*a26);
  a37=(a27*a23);
  a37=(a0*a37);
  a25=(a25+a37);
  a37=(a7*a9);
  a23=(a30*a23);
  a23=(a4*a23);
  a37=(a37-a23);
  a25=(a25+a37);
  if (res[1]!=0) res[1][9]=a25;
  a25=arg[1]? arg[1][13] : 0;
  if (res[1]!=0) res[1][10]=a25;
  a37=arg[1]? arg[1][14] : 0;
  a23=arg[1]? arg[1][15] : 0;
  a38=(a37+a23);
  a39=(a11*a26);
  a38=(a38-a39);
  a39=(a33*a26);
  a39=(a39/a35);
  a39=(a12*a39);
  a39=(a0*a39);
  a26=(a15*a26);
  a39=(a39+a26);
  a38=(a38-a39);
  a38=(a29*a38);
  if (res[1]!=0) res[1][11]=a38;
  a38=(a10*a9);
  a39=(a34*a9);
  a39=(a39/a36);
  a39=(a18*a39);
  a39=(a4*a39);
  a26=(a14*a9);
  a39=(a39+a26);
  a38=(a38-a39);
  a39=(a20*a25);
  a38=(a38-a39);
  a38=(a29*a38);
  if (res[1]!=0) res[1][12]=a38;
  a23=(a23-a37);
  a23=(a17*a23);
  a37=(a8*a25);
  a23=(a23-a37);
  a37=(a16*a25);
  a37=(a6*a37);
  a38=(a21*a25);
  a37=(a37+a38);
  a37=(a6*a37);
  a25=(a22*a25);
  a37=(a37+a25);
  a23=(a23-a37);
  a9=(a20*a9);
  a23=(a23-a9);
  a23=(a32*a23);
  if (res[1]!=0) res[1][13]=a23;
  if (res[1]!=0) res[1][14]=a31;
  if (res[1]!=0) res[1][15]=a31;
  a23=arg[1]? arg[1][19] : 0;
  a9=(a2*a23);
  a37=arg[1]? arg[1][18] : 0;
  a25=(a24*a37);
  a25=(a0*a25);
  a9=(a9-a25);
  a25=arg[1]? arg[1][20] : 0;
  a38=(a5*a25);
  a39=(a28*a37);
  a39=(a4*a39);
  a38=(a38+a39);
  a9=(a9-a38);
  if (res[1]!=0) res[1][16]=a9;
  a9=(a3*a23);
  a38=(a27*a37);
  a38=(a0*a38);
  a9=(a9+a38);
  a38=(a7*a25);
  a37=(a30*a37);
  a37=(a4*a37);
  a38=(a38-a37);
  a9=(a9+a38);
  if (res[1]!=0) res[1][17]=a9;
  a9=arg[1]? arg[1][21] : 0;
  if (res[1]!=0) res[1][18]=a9;
  a38=arg[1]? arg[1][22] : 0;
  a37=arg[1]? arg[1][23] : 0;
  a39=(a38+a37);
  a26=(a11*a23);
  a39=(a39-a26);
  a26=(a33*a23);
  a26=(a26/a35);
  a26=(a12*a26);
  a26=(a0*a26);
  a23=(a15*a23);
  a26=(a26+a23);
  a39=(a39-a26);
  a39=(a29*a39);
  if (res[1]!=0) res[1][19]=a39;
  a39=(a10*a25);
  a26=(a34*a25);
  a26=(a26/a36);
  a26=(a18*a26);
  a26=(a4*a26);
  a23=(a14*a25);
  a26=(a26+a23);
  a39=(a39-a26);
  a26=(a20*a9);
  a39=(a39-a26);
  a39=(a29*a39);
  if (res[1]!=0) res[1][20]=a39;
  a37=(a37-a38);
  a37=(a17*a37);
  a38=(a8*a9);
  a37=(a37-a38);
  a38=(a16*a9);
  a38=(a6*a38);
  a39=(a21*a9);
  a38=(a38+a39);
  a38=(a6*a38);
  a9=(a22*a9);
  a38=(a38+a9);
  a37=(a37-a38);
  a25=(a20*a25);
  a37=(a37-a25);
  a37=(a32*a37);
  if (res[1]!=0) res[1][21]=a37;
  if (res[1]!=0) res[1][22]=a31;
  if (res[1]!=0) res[1][23]=a31;
  a37=arg[1]? arg[1][27] : 0;
  a25=(a2*a37);
  a38=arg[1]? arg[1][26] : 0;
  a9=(a24*a38);
  a9=(a0*a9);
  a25=(a25-a9);
  a9=arg[1]? arg[1][28] : 0;
  a39=(a5*a9);
  a26=(a28*a38);
  a26=(a4*a26);
  a39=(a39+a26);
  a25=(a25-a39);
  if (res[1]!=0) res[1][24]=a25;
  a25=(a3*a37);
  a39=(a27*a38);
  a39=(a0*a39);
  a25=(a25+a39);
  a39=(a7*a9);
  a38=(a30*a38);
  a38=(a4*a38);
  a39=(a39-a38);
  a25=(a25+a39);
  if (res[1]!=0) res[1][25]=a25;
  a25=arg[1]? arg[1][29] : 0;
  if (res[1]!=0) res[1][26]=a25;
  a39=arg[1]? arg[1][30] : 0;
  a38=arg[1]? arg[1][31] : 0;
  a26=(a39+a38);
  a23=(a11*a37);
  a26=(a26-a23);
  a23=(a33*a37);
  a23=(a23/a35);
  a23=(a12*a23);
  a23=(a0*a23);
  a37=(a15*a37);
  a23=(a23+a37);
  a26=(a26-a23);
  a26=(a29*a26);
  if (res[1]!=0) res[1][27]=a26;
  a26=(a10*a9);
  a23=(a34*a9);
  a23=(a23/a36);
  a23=(a18*a23);
  a23=(a4*a23);
  a37=(a14*a9);
  a23=(a23+a37);
  a26=(a26-a23);
  a23=(a20*a25);
  a26=(a26-a23);
  a26=(a29*a26);
  if (res[1]!=0) res[1][28]=a26;
  a38=(a38-a39);
  a38=(a17*a38);
  a39=(a8*a25);
  a38=(a38-a39);
  a39=(a16*a25);
  a39=(a6*a39);
  a26=(a21*a25);
  a39=(a39+a26);
  a39=(a6*a39);
  a25=(a22*a25);
  a39=(a39+a25);
  a38=(a38-a39);
  a9=(a20*a9);
  a38=(a38-a9);
  a38=(a32*a38);
  if (res[1]!=0) res[1][29]=a38;
  if (res[1]!=0) res[1][30]=a31;
  if (res[1]!=0) res[1][31]=a31;
  a38=arg[1]? arg[1][35] : 0;
  a9=(a2*a38);
  a39=arg[1]? arg[1][34] : 0;
  a25=(a24*a39);
  a25=(a0*a25);
  a9=(a9-a25);
  a25=arg[1]? arg[1][36] : 0;
  a26=(a5*a25);
  a23=(a28*a39);
  a23=(a4*a23);
  a26=(a26+a23);
  a9=(a9-a26);
  if (res[1]!=0) res[1][32]=a9;
  a9=(a3*a38);
  a26=(a27*a39);
  a26=(a0*a26);
  a9=(a9+a26);
  a26=(a7*a25);
  a39=(a30*a39);
  a39=(a4*a39);
  a26=(a26-a39);
  a9=(a9+a26);
  if (res[1]!=0) res[1][33]=a9;
  a9=arg[1]? arg[1][37] : 0;
  if (res[1]!=0) res[1][34]=a9;
  a26=arg[1]? arg[1][38] : 0;
  a39=arg[1]? arg[1][39] : 0;
  a23=(a26+a39);
  a37=(a11*a38);
  a23=(a23-a37);
  a37=(a33*a38);
  a37=(a37/a35);
  a37=(a12*a37);
  a37=(a0*a37);
  a38=(a15*a38);
  a37=(a37+a38);
  a23=(a23-a37);
  a23=(a29*a23);
  if (res[1]!=0) res[1][35]=a23;
  a23=(a10*a25);
  a37=(a34*a25);
  a37=(a37/a36);
  a37=(a18*a37);
  a37=(a4*a37);
  a38=(a14*a25);
  a37=(a37+a38);
  a23=(a23-a37);
  a37=(a20*a9);
  a23=(a23-a37);
  a23=(a29*a23);
  if (res[1]!=0) res[1][36]=a23;
  a39=(a39-a26);
  a39=(a17*a39);
  a26=(a8*a9);
  a39=(a39-a26);
  a26=(a16*a9);
  a26=(a6*a26);
  a23=(a21*a9);
  a26=(a26+a23);
  a26=(a6*a26);
  a9=(a22*a9);
  a26=(a26+a9);
  a39=(a39-a26);
  a25=(a20*a25);
  a39=(a39-a25);
  a39=(a32*a39);
  if (res[1]!=0) res[1][37]=a39;
  if (res[1]!=0) res[1][38]=a31;
  if (res[1]!=0) res[1][39]=a31;
  a39=arg[1]? arg[1][43] : 0;
  a25=(a2*a39);
  a26=arg[1]? arg[1][42] : 0;
  a9=(a24*a26);
  a9=(a0*a9);
  a25=(a25-a9);
  a9=arg[1]? arg[1][44] : 0;
  a23=(a5*a9);
  a37=(a28*a26);
  a37=(a4*a37);
  a23=(a23+a37);
  a25=(a25-a23);
  if (res[1]!=0) res[1][40]=a25;
  a25=(a3*a39);
  a23=(a27*a26);
  a23=(a0*a23);
  a25=(a25+a23);
  a23=(a7*a9);
  a26=(a30*a26);
  a26=(a4*a26);
  a23=(a23-a26);
  a25=(a25+a23);
  if (res[1]!=0) res[1][41]=a25;
  a25=arg[1]? arg[1][45] : 0;
  if (res[1]!=0) res[1][42]=a25;
  a23=arg[1]? arg[1][46] : 0;
  a26=arg[1]? arg[1][47] : 0;
  a37=(a23+a26);
  a38=(a11*a39);
  a37=(a37-a38);
  a38=(a33*a39);
  a38=(a38/a35);
  a38=(a12*a38);
  a38=(a0*a38);
  a39=(a15*a39);
  a38=(a38+a39);
  a37=(a37-a38);
  a37=(a29*a37);
  if (res[1]!=0) res[1][43]=a37;
  a37=(a10*a9);
  a38=(a34*a9);
  a38=(a38/a36);
  a38=(a18*a38);
  a38=(a4*a38);
  a39=(a14*a9);
  a38=(a38+a39);
  a37=(a37-a38);
  a38=(a20*a25);
  a37=(a37-a38);
  a37=(a29*a37);
  if (res[1]!=0) res[1][44]=a37;
  a26=(a26-a23);
  a26=(a17*a26);
  a23=(a8*a25);
  a26=(a26-a23);
  a23=(a16*a25);
  a23=(a6*a23);
  a37=(a21*a25);
  a23=(a23+a37);
  a23=(a6*a23);
  a25=(a22*a25);
  a23=(a23+a25);
  a26=(a26-a23);
  a9=(a20*a9);
  a26=(a26-a9);
  a26=(a32*a26);
  if (res[1]!=0) res[1][45]=a26;
  if (res[1]!=0) res[1][46]=a31;
  if (res[1]!=0) res[1][47]=a31;
  a26=arg[1]? arg[1][51] : 0;
  a9=(a2*a26);
  a23=arg[1]? arg[1][50] : 0;
  a25=(a24*a23);
  a25=(a0*a25);
  a9=(a9-a25);
  a25=arg[1]? arg[1][52] : 0;
  a37=(a5*a25);
  a38=(a28*a23);
  a38=(a4*a38);
  a37=(a37+a38);
  a9=(a9-a37);
  if (res[1]!=0) res[1][48]=a9;
  a9=(a3*a26);
  a37=(a27*a23);
  a37=(a0*a37);
  a9=(a9+a37);
  a37=(a7*a25);
  a23=(a30*a23);
  a23=(a4*a23);
  a37=(a37-a23);
  a9=(a9+a37);
  if (res[1]!=0) res[1][49]=a9;
  a9=arg[1]? arg[1][53] : 0;
  if (res[1]!=0) res[1][50]=a9;
  a37=arg[1]? arg[1][54] : 0;
  a23=arg[1]? arg[1][55] : 0;
  a38=(a37+a23);
  a39=(a11*a26);
  a38=(a38-a39);
  a39=(a33*a26);
  a39=(a39/a35);
  a39=(a12*a39);
  a39=(a0*a39);
  a26=(a15*a26);
  a39=(a39+a26);
  a38=(a38-a39);
  a38=(a29*a38);
  if (res[1]!=0) res[1][51]=a38;
  a38=(a10*a25);
  a39=(a34*a25);
  a39=(a39/a36);
  a39=(a18*a39);
  a39=(a4*a39);
  a26=(a14*a25);
  a39=(a39+a26);
  a38=(a38-a39);
  a39=(a20*a9);
  a38=(a38-a39);
  a38=(a29*a38);
  if (res[1]!=0) res[1][52]=a38;
  a23=(a23-a37);
  a23=(a17*a23);
  a37=(a8*a9);
  a23=(a23-a37);
  a37=(a16*a9);
  a37=(a6*a37);
  a38=(a21*a9);
  a37=(a37+a38);
  a37=(a6*a37);
  a9=(a22*a9);
  a37=(a37+a9);
  a23=(a23-a37);
  a25=(a20*a25);
  a23=(a23-a25);
  a23=(a32*a23);
  if (res[1]!=0) res[1][53]=a23;
  if (res[1]!=0) res[1][54]=a31;
  if (res[1]!=0) res[1][55]=a31;
  a23=arg[1]? arg[1][59] : 0;
  a25=(a2*a23);
  a37=arg[1]? arg[1][58] : 0;
  a24=(a24*a37);
  a24=(a0*a24);
  a25=(a25-a24);
  a24=arg[1]? arg[1][60] : 0;
  a9=(a5*a24);
  a28=(a28*a37);
  a28=(a4*a28);
  a9=(a9+a28);
  a25=(a25-a9);
  if (res[1]!=0) res[1][56]=a25;
  a25=(a3*a23);
  a27=(a27*a37);
  a27=(a0*a27);
  a25=(a25+a27);
  a27=(a7*a24);
  a30=(a30*a37);
  a30=(a4*a30);
  a27=(a27-a30);
  a25=(a25+a27);
  if (res[1]!=0) res[1][57]=a25;
  a25=arg[1]? arg[1][61] : 0;
  if (res[1]!=0) res[1][58]=a25;
  a27=arg[1]? arg[1][62] : 0;
  a30=arg[1]? arg[1][63] : 0;
  a37=(a27+a30);
  a9=(a11*a23);
  a37=(a37-a9);
  a33=(a33*a23);
  a33=(a33/a35);
  a33=(a12*a33);
  a33=(a0*a33);
  a23=(a15*a23);
  a33=(a33+a23);
  a37=(a37-a33);
  a37=(a29*a37);
  if (res[1]!=0) res[1][59]=a37;
  a37=(a10*a24);
  a34=(a34*a24);
  a34=(a34/a36);
  a34=(a18*a34);
  a34=(a4*a34);
  a36=(a14*a24);
  a34=(a34+a36);
  a37=(a37-a34);
  a34=(a20*a25);
  a37=(a37-a34);
  a37=(a29*a37);
  if (res[1]!=0) res[1][60]=a37;
  a30=(a30-a27);
  a30=(a17*a30);
  a27=(a8*a25);
  a30=(a30-a27);
  a27=(a16*a25);
  a27=(a6*a27);
  a37=(a21*a25);
  a27=(a27+a37);
  a27=(a6*a27);
  a25=(a22*a25);
  a27=(a27+a25);
  a30=(a30-a27);
  a24=(a20*a24);
  a30=(a30-a24);
  a30=(a32*a30);
  if (res[1]!=0) res[1][61]=a30;
  if (res[1]!=0) res[1][62]=a31;
  if (res[1]!=0) res[1][63]=a31;
  a30=arg[2]? arg[2][3] : 0;
  a24=(a2*a30);
  a27=sin(a1);
  a25=arg[2]? arg[2][2] : 0;
  a37=(a27*a25);
  a37=(a0*a37);
  a24=(a24-a37);
  a37=arg[2]? arg[2][4] : 0;
  a34=(a5*a37);
  a36=cos(a1);
  a33=(a36*a25);
  a33=(a4*a33);
  a34=(a34+a33);
  a24=(a24-a34);
  if (res[2]!=0) res[2][0]=a24;
  a24=(a3*a30);
  a34=cos(a1);
  a33=(a34*a25);
  a33=(a0*a33);
  a24=(a24+a33);
  a33=(a7*a37);
  a1=sin(a1);
  a25=(a1*a25);
  a25=(a4*a25);
  a33=(a33-a25);
  a24=(a24+a33);
  if (res[2]!=0) res[2][1]=a24;
  a24=arg[2]? arg[2][5] : 0;
  if (res[2]!=0) res[2][2]=a24;
  a33=arg[2]? arg[2][6] : 0;
  a25=arg[2]? arg[2][7] : 0;
  a23=(a33+a25);
  a35=(a11*a30);
  a23=(a23-a35);
  a35=(a0+a0);
  a9=(a35*a30);
  a13=(a13+a13);
  a9=(a9/a13);
  a9=(a12*a9);
  a9=(a0*a9);
  a30=(a15*a30);
  a9=(a9+a30);
  a23=(a23-a9);
  a23=(a29*a23);
  if (res[2]!=0) res[2][3]=a23;
  a23=(a10*a37);
  a9=(a4+a4);
  a30=(a9*a37);
  a19=(a19+a19);
  a30=(a30/a19);
  a30=(a18*a30);
  a30=(a4*a30);
  a28=(a14*a37);
  a30=(a30+a28);
  a23=(a23-a30);
  a30=(a20*a24);
  a23=(a23-a30);
  a23=(a29*a23);
  if (res[2]!=0) res[2][4]=a23;
  a25=(a25-a33);
  a25=(a17*a25);
  a33=(a8*a24);
  a25=(a25-a33);
  a33=(a16*a24);
  a33=(a6*a33);
  a23=(a21*a24);
  a33=(a33+a23);
  a33=(a6*a33);
  a24=(a22*a24);
  a33=(a33+a24);
  a25=(a25-a33);
  a37=(a20*a37);
  a25=(a25-a37);
  a25=(a32*a25);
  if (res[2]!=0) res[2][5]=a25;
  a25=1.;
  if (res[2]!=0) res[2][6]=a25;
  if (res[2]!=0) res[2][7]=a31;
  a37=arg[2]? arg[2][11] : 0;
  a2=(a2*a37);
  a33=arg[2]? arg[2][10] : 0;
  a27=(a27*a33);
  a27=(a0*a27);
  a2=(a2-a27);
  a27=arg[2]? arg[2][12] : 0;
  a5=(a5*a27);
  a36=(a36*a33);
  a36=(a4*a36);
  a5=(a5+a36);
  a2=(a2-a5);
  if (res[2]!=0) res[2][8]=a2;
  a3=(a3*a37);
  a34=(a34*a33);
  a34=(a0*a34);
  a3=(a3+a34);
  a7=(a7*a27);
  a1=(a1*a33);
  a1=(a4*a1);
  a7=(a7-a1);
  a3=(a3+a7);
  if (res[2]!=0) res[2][9]=a3;
  a3=arg[2]? arg[2][13] : 0;
  if (res[2]!=0) res[2][10]=a3;
  a7=arg[2]? arg[2][14] : 0;
  a1=arg[2]? arg[2][15] : 0;
  a33=(a7+a1);
  a11=(a11*a37);
  a33=(a33-a11);
  a35=(a35*a37);
  a35=(a35/a13);
  a12=(a12*a35);
  a0=(a0*a12);
  a15=(a15*a37);
  a0=(a0+a15);
  a33=(a33-a0);
  a33=(a29*a33);
  if (res[2]!=0) res[2][11]=a33;
  a10=(a10*a27);
  a9=(a9*a27);
  a9=(a9/a19);
  a18=(a18*a9);
  a4=(a4*a18);
  a14=(a14*a27);
  a4=(a4+a14);
  a10=(a10-a4);
  a4=(a20*a3);
  a10=(a10-a4);
  a29=(a29*a10);
  if (res[2]!=0) res[2][12]=a29;
  a1=(a1-a7);
  a17=(a17*a1);
  a8=(a8*a3);
  a17=(a17-a8);
  a16=(a16*a3);
  a16=(a6*a16);
  a21=(a21*a3);
  a16=(a16+a21);
  a6=(a6*a16);
  a22=(a22*a3);
  a6=(a6+a22);
  a17=(a17-a6);
  a20=(a20*a27);
  a17=(a17-a20);
  a32=(a32*a17);
  if (res[2]!=0) res[2][13]=a32;
  if (res[2]!=0) res[2][14]=a31;
  if (res[2]!=0) res[2][15]=a25;
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

CASADI_SYMBOL_EXPORT casadi_real heron_expl_vde_forw_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* heron_expl_vde_forw_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* heron_expl_vde_forw_name_out(casadi_int i) {
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