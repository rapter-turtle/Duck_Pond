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
  #define CASADI_PREFIX(ID) heron_constr_h_fun_jac_uxt_zt_ ## ID
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
static const casadi_int casadi_s1[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[19] = {15, 1, 0, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
static const casadi_int casadi_s4[9] = {5, 1, 0, 5, 0, 1, 2, 3, 4};
static const casadi_int casadi_s5[43] = {9, 5, 0, 7, 14, 21, 28, 35, 2, 3, 4, 5, 6, 7, 8, 2, 3, 4, 5, 6, 7, 8, 2, 3, 4, 5, 6, 7, 8, 2, 3, 4, 5, 6, 7, 8, 2, 3, 4, 5, 6, 7, 8};
static const casadi_int casadi_s6[3] = {5, 0, 0};

/* heron_constr_h_fun_jac_uxt_zt:(i0[7],i1[2],i2[],i3[15])->(o0[5],o1[9x5,35nz],o2[5x0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a100, a101, a102, a103, a104, a105, a106, a107, a108, a109, a11, a110, a111, a112, a113, a114, a115, a116, a117, a118, a119, a12, a120, a121, a122, a123, a124, a125, a126, a127, a128, a129, a13, a130, a131, a132, a133, a134, a135, a136, a137, a138, a139, a14, a140, a141, a142, a143, a144, a145, a146, a147, a148, a149, a15, a150, a151, a152, a153, a154, a155, a156, a157, a158, a159, a16, a160, a161, a162, a163, a164, a165, a166, a167, a168, a169, a17, a170, a171, a172, a173, a174, a175, a176, a177, a178, a179, a18, a180, a181, a182, a183, a184, a185, a186, a187, a188, a189, a19, a190, a191, a192, a193, a194, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a8, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a9, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99;
  a0=arg[3]? arg[3][0] : 0;
  a1=arg[0]? arg[0][0] : 0;
  a2=1.0000000000000001e-01;
  a3=arg[0]? arg[0][3] : 0;
  a4=arg[0]? arg[0][2] : 0;
  a5=cos(a4);
  a6=(a3*a5);
  a6=(a2*a6);
  a6=(a1+a6);
  a7=(a0-a6);
  a8=4.;
  a9=arg[0]? arg[0][5] : 0;
  a10=arg[0]? arg[0][6] : 0;
  a9=(a9+a10);
  a10=10.;
  a11=1.6899999999999999e+01;
  a12=casadi_sq(a3);
  a13=1.0000000000000001e-05;
  a12=(a12+a13);
  a12=sqrt(a12);
  a13=(a11*a12);
  a10=(a10+a13);
  a13=(a10*a3);
  a9=(a9-a13);
  a13=36.;
  a9=(a9/a13);
  a9=(a2*a9);
  a9=(a3+a9);
  a13=5.9999999999999998e-01;
  a14=(a9/a13);
  a14=(a8*a14);
  a15=arg[0]? arg[0][4] : 0;
  a15=(a2*a15);
  a15=(a4+a15);
  a16=1.5707963267948966e+00;
  a17=(a15-a16);
  a18=cos(a17);
  a19=(a14*a18);
  a7=(a7-a19);
  a19=casadi_sq(a7);
  a20=arg[3]? arg[3][1] : 0;
  a21=arg[0]? arg[0][1] : 0;
  a22=sin(a4);
  a23=(a3*a22);
  a23=(a2*a23);
  a23=(a21+a23);
  a24=(a20-a23);
  a25=(a15-a16);
  a26=sin(a25);
  a27=(a14*a26);
  a24=(a24-a27);
  a27=casadi_sq(a24);
  a19=(a19+a27);
  a19=sqrt(a19);
  a27=arg[3]? arg[3][2] : 0;
  a28=(a27+a14);
  a28=(a19-a28);
  a28=exp(a28);
  a29=(a0-a6);
  a30=(a15+a16);
  a31=cos(a30);
  a32=(a14*a31);
  a29=(a29-a32);
  a32=casadi_sq(a29);
  a33=(a20-a23);
  a34=(a15+a16);
  a35=sin(a34);
  a36=(a14*a35);
  a33=(a33-a36);
  a36=casadi_sq(a33);
  a32=(a32+a36);
  a32=sqrt(a32);
  a36=(a27+a14);
  a36=(a32-a36);
  a36=exp(a36);
  a37=(a28+a36);
  a38=1.;
  a37=(a37-a38);
  a39=log(a37);
  a40=9.7499999999999998e-01;
  a41=(a0-a1);
  a42=(a3/a13);
  a42=(a8*a42);
  a43=(a4-a16);
  a44=cos(a43);
  a45=(a42*a44);
  a41=(a41-a45);
  a45=casadi_sq(a41);
  a46=(a20-a21);
  a47=(a4-a16);
  a48=sin(a47);
  a49=(a42*a48);
  a46=(a46-a49);
  a49=casadi_sq(a46);
  a45=(a45+a49);
  a45=sqrt(a45);
  a49=(a27+a42);
  a49=(a45-a49);
  a49=exp(a49);
  a0=(a0-a1);
  a50=(a4+a16);
  a51=cos(a50);
  a52=(a42*a51);
  a0=(a0-a52);
  a52=casadi_sq(a0);
  a20=(a20-a21);
  a53=(a4+a16);
  a54=sin(a53);
  a55=(a42*a54);
  a20=(a20-a55);
  a55=casadi_sq(a20);
  a52=(a52+a55);
  a52=sqrt(a52);
  a27=(a27+a42);
  a27=(a52-a27);
  a27=exp(a27);
  a55=(a49+a27);
  a55=(a55-a38);
  a56=log(a55);
  a56=(a40*a56);
  a39=(a39-a56);
  if (res[0]!=0) res[0][0]=a39;
  a39=arg[3]? arg[3][3] : 0;
  a56=(a39-a6);
  a57=(a9/a13);
  a57=(a8*a57);
  a58=(a15-a16);
  a59=cos(a58);
  a60=(a57*a59);
  a56=(a56-a60);
  a60=casadi_sq(a56);
  a61=arg[3]? arg[3][4] : 0;
  a62=(a61-a23);
  a63=(a15-a16);
  a64=sin(a63);
  a65=(a57*a64);
  a62=(a62-a65);
  a65=casadi_sq(a62);
  a60=(a60+a65);
  a60=sqrt(a60);
  a65=arg[3]? arg[3][5] : 0;
  a66=(a65+a57);
  a66=(a60-a66);
  a66=exp(a66);
  a67=(a39-a6);
  a68=(a15+a16);
  a69=cos(a68);
  a70=(a57*a69);
  a67=(a67-a70);
  a70=casadi_sq(a67);
  a71=(a61-a23);
  a72=(a15+a16);
  a73=sin(a72);
  a74=(a57*a73);
  a71=(a71-a74);
  a74=casadi_sq(a71);
  a70=(a70+a74);
  a70=sqrt(a70);
  a74=(a65+a57);
  a74=(a70-a74);
  a74=exp(a74);
  a75=(a66+a74);
  a75=(a75-a38);
  a76=log(a75);
  a77=(a39-a1);
  a78=(a3/a13);
  a78=(a8*a78);
  a79=(a4-a16);
  a80=cos(a79);
  a81=(a78*a80);
  a77=(a77-a81);
  a81=casadi_sq(a77);
  a82=(a61-a21);
  a83=(a4-a16);
  a84=sin(a83);
  a85=(a78*a84);
  a82=(a82-a85);
  a85=casadi_sq(a82);
  a81=(a81+a85);
  a81=sqrt(a81);
  a85=(a65+a78);
  a85=(a81-a85);
  a85=exp(a85);
  a39=(a39-a1);
  a86=(a4+a16);
  a87=cos(a86);
  a88=(a78*a87);
  a39=(a39-a88);
  a88=casadi_sq(a39);
  a61=(a61-a21);
  a89=(a4+a16);
  a90=sin(a89);
  a91=(a78*a90);
  a61=(a61-a91);
  a91=casadi_sq(a61);
  a88=(a88+a91);
  a88=sqrt(a88);
  a65=(a65+a78);
  a65=(a88-a65);
  a65=exp(a65);
  a91=(a85+a65);
  a91=(a91-a38);
  a92=log(a91);
  a92=(a40*a92);
  a76=(a76-a92);
  if (res[0]!=0) res[0][1]=a76;
  a76=arg[3]? arg[3][6] : 0;
  a92=(a76-a6);
  a93=(a9/a13);
  a93=(a8*a93);
  a94=(a15-a16);
  a95=cos(a94);
  a96=(a93*a95);
  a92=(a92-a96);
  a96=casadi_sq(a92);
  a97=arg[3]? arg[3][7] : 0;
  a98=(a97-a23);
  a99=(a15-a16);
  a100=sin(a99);
  a101=(a93*a100);
  a98=(a98-a101);
  a101=casadi_sq(a98);
  a96=(a96+a101);
  a96=sqrt(a96);
  a101=arg[3]? arg[3][8] : 0;
  a102=(a101+a93);
  a102=(a96-a102);
  a102=exp(a102);
  a103=(a76-a6);
  a104=(a15+a16);
  a105=cos(a104);
  a106=(a93*a105);
  a103=(a103-a106);
  a106=casadi_sq(a103);
  a107=(a97-a23);
  a108=(a15+a16);
  a109=sin(a108);
  a110=(a93*a109);
  a107=(a107-a110);
  a110=casadi_sq(a107);
  a106=(a106+a110);
  a106=sqrt(a106);
  a110=(a101+a93);
  a110=(a106-a110);
  a110=exp(a110);
  a111=(a102+a110);
  a111=(a111-a38);
  a112=log(a111);
  a113=(a76-a1);
  a114=(a3/a13);
  a114=(a8*a114);
  a115=(a4-a16);
  a116=cos(a115);
  a117=(a114*a116);
  a113=(a113-a117);
  a117=casadi_sq(a113);
  a118=(a97-a21);
  a119=(a4-a16);
  a120=sin(a119);
  a121=(a114*a120);
  a118=(a118-a121);
  a121=casadi_sq(a118);
  a117=(a117+a121);
  a117=sqrt(a117);
  a121=(a101+a114);
  a121=(a117-a121);
  a121=exp(a121);
  a76=(a76-a1);
  a122=(a4+a16);
  a123=cos(a122);
  a124=(a114*a123);
  a76=(a76-a124);
  a124=casadi_sq(a76);
  a97=(a97-a21);
  a125=(a4+a16);
  a126=sin(a125);
  a127=(a114*a126);
  a97=(a97-a127);
  a127=casadi_sq(a97);
  a124=(a124+a127);
  a124=sqrt(a124);
  a101=(a101+a114);
  a101=(a124-a101);
  a101=exp(a101);
  a127=(a121+a101);
  a127=(a127-a38);
  a128=log(a127);
  a128=(a40*a128);
  a112=(a112-a128);
  if (res[0]!=0) res[0][2]=a112;
  a112=arg[3]? arg[3][9] : 0;
  a128=(a112-a6);
  a129=(a9/a13);
  a129=(a8*a129);
  a130=(a15-a16);
  a131=cos(a130);
  a132=(a129*a131);
  a128=(a128-a132);
  a132=casadi_sq(a128);
  a133=arg[3]? arg[3][10] : 0;
  a134=(a133-a23);
  a135=(a15-a16);
  a136=sin(a135);
  a137=(a129*a136);
  a134=(a134-a137);
  a137=casadi_sq(a134);
  a132=(a132+a137);
  a132=sqrt(a132);
  a137=arg[3]? arg[3][11] : 0;
  a138=(a137+a129);
  a138=(a132-a138);
  a138=exp(a138);
  a139=(a112-a6);
  a140=(a15+a16);
  a141=cos(a140);
  a142=(a129*a141);
  a139=(a139-a142);
  a142=casadi_sq(a139);
  a143=(a133-a23);
  a144=(a15+a16);
  a145=sin(a144);
  a146=(a129*a145);
  a143=(a143-a146);
  a146=casadi_sq(a143);
  a142=(a142+a146);
  a142=sqrt(a142);
  a146=(a137+a129);
  a146=(a142-a146);
  a146=exp(a146);
  a147=(a138+a146);
  a147=(a147-a38);
  a148=log(a147);
  a149=(a112-a1);
  a150=(a3/a13);
  a150=(a8*a150);
  a151=(a4-a16);
  a152=cos(a151);
  a153=(a150*a152);
  a149=(a149-a153);
  a153=casadi_sq(a149);
  a154=(a133-a21);
  a155=(a4-a16);
  a156=sin(a155);
  a157=(a150*a156);
  a154=(a154-a157);
  a157=casadi_sq(a154);
  a153=(a153+a157);
  a153=sqrt(a153);
  a157=(a137+a150);
  a157=(a153-a157);
  a157=exp(a157);
  a112=(a112-a1);
  a158=(a4+a16);
  a159=cos(a158);
  a160=(a150*a159);
  a112=(a112-a160);
  a160=casadi_sq(a112);
  a133=(a133-a21);
  a161=(a4+a16);
  a162=sin(a161);
  a163=(a150*a162);
  a133=(a133-a163);
  a163=casadi_sq(a133);
  a160=(a160+a163);
  a160=sqrt(a160);
  a137=(a137+a150);
  a137=(a160-a137);
  a137=exp(a137);
  a163=(a157+a137);
  a163=(a163-a38);
  a164=log(a163);
  a164=(a40*a164);
  a148=(a148-a164);
  if (res[0]!=0) res[0][3]=a148;
  a148=arg[3]? arg[3][12] : 0;
  a164=(a148-a6);
  a9=(a9/a13);
  a9=(a8*a9);
  a165=(a15-a16);
  a166=cos(a165);
  a167=(a9*a166);
  a164=(a164-a167);
  a167=casadi_sq(a164);
  a168=arg[3]? arg[3][13] : 0;
  a169=(a168-a23);
  a170=(a15-a16);
  a171=sin(a170);
  a172=(a9*a171);
  a169=(a169-a172);
  a172=casadi_sq(a169);
  a167=(a167+a172);
  a167=sqrt(a167);
  a172=arg[3]? arg[3][14] : 0;
  a173=(a172+a9);
  a173=(a167-a173);
  a173=exp(a173);
  a6=(a148-a6);
  a174=(a15+a16);
  a175=cos(a174);
  a176=(a9*a175);
  a6=(a6-a176);
  a176=casadi_sq(a6);
  a23=(a168-a23);
  a15=(a15+a16);
  a177=sin(a15);
  a178=(a9*a177);
  a23=(a23-a178);
  a178=casadi_sq(a23);
  a176=(a176+a178);
  a176=sqrt(a176);
  a178=(a172+a9);
  a178=(a176-a178);
  a178=exp(a178);
  a179=(a173+a178);
  a179=(a179-a38);
  a180=log(a179);
  a181=(a148-a1);
  a13=(a3/a13);
  a13=(a8*a13);
  a182=(a4-a16);
  a183=cos(a182);
  a184=(a13*a183);
  a181=(a181-a184);
  a184=casadi_sq(a181);
  a185=(a168-a21);
  a186=(a4-a16);
  a187=sin(a186);
  a188=(a13*a187);
  a185=(a185-a188);
  a188=casadi_sq(a185);
  a184=(a184+a188);
  a184=sqrt(a184);
  a188=(a172+a13);
  a188=(a184-a188);
  a188=exp(a188);
  a148=(a148-a1);
  a1=(a4+a16);
  a189=cos(a1);
  a190=(a13*a189);
  a148=(a148-a190);
  a190=casadi_sq(a148);
  a168=(a168-a21);
  a16=(a4+a16);
  a21=sin(a16);
  a191=(a13*a21);
  a168=(a168-a191);
  a191=casadi_sq(a168);
  a190=(a190+a191);
  a190=sqrt(a190);
  a172=(a172+a13);
  a172=(a190-a172);
  a172=exp(a172);
  a191=(a188+a172);
  a191=(a191-a38);
  a192=log(a191);
  a192=(a40*a192);
  a180=(a180-a192);
  if (res[0]!=0) res[0][4]=a180;
  a180=(a41/a45);
  a180=(a49*a180);
  a192=(a0/a52);
  a192=(a27*a192);
  a180=(a180+a192);
  a180=(a180/a55);
  a180=(a40*a180);
  a192=(a7/a19);
  a192=(a28*a192);
  a193=(a29/a32);
  a193=(a36*a193);
  a192=(a192+a193);
  a192=(a192/a37);
  a180=(a180-a192);
  if (res[1]!=0) res[1][0]=a180;
  a180=(a46/a45);
  a180=(a49*a180);
  a192=(a20/a52);
  a192=(a27*a192);
  a180=(a180+a192);
  a180=(a180/a55);
  a180=(a40*a180);
  a192=(a24/a19);
  a192=(a28*a192);
  a193=(a33/a32);
  a193=(a36*a193);
  a192=(a192+a193);
  a192=(a192/a37);
  a180=(a180-a192);
  if (res[1]!=0) res[1][1]=a180;
  a7=(a7+a7);
  a180=sin(a4);
  a180=(a3*a180);
  a180=(a2*a180);
  a17=sin(a17);
  a192=(a14*a17);
  a192=(a180+a192);
  a192=(a7*a192);
  a24=(a24+a24);
  a4=cos(a4);
  a4=(a3*a4);
  a4=(a2*a4);
  a25=cos(a25);
  a193=(a14*a25);
  a193=(a4+a193);
  a193=(a24*a193);
  a192=(a192-a193);
  a19=(a19+a19);
  a192=(a192/a19);
  a192=(a28*a192);
  a29=(a29+a29);
  a30=sin(a30);
  a193=(a14*a30);
  a193=(a180+a193);
  a193=(a29*a193);
  a33=(a33+a33);
  a34=cos(a34);
  a194=(a14*a34);
  a194=(a4+a194);
  a194=(a33*a194);
  a193=(a193-a194);
  a32=(a32+a32);
  a193=(a193/a32);
  a193=(a36*a193);
  a192=(a192+a193);
  a192=(a192/a37);
  a41=(a41+a41);
  a43=sin(a43);
  a43=(a42*a43);
  a43=(a41*a43);
  a46=(a46+a46);
  a47=cos(a47);
  a47=(a42*a47);
  a47=(a46*a47);
  a43=(a43-a47);
  a45=(a45+a45);
  a43=(a43/a45);
  a43=(a49*a43);
  a0=(a0+a0);
  a50=sin(a50);
  a50=(a42*a50);
  a50=(a0*a50);
  a20=(a20+a20);
  a53=cos(a53);
  a42=(a42*a53);
  a42=(a20*a42);
  a50=(a50-a42);
  a52=(a52+a52);
  a50=(a50/a52);
  a50=(a27*a50);
  a43=(a43+a50);
  a43=(a43/a55);
  a43=(a40*a43);
  a192=(a192-a43);
  if (res[1]!=0) res[1][2]=a192;
  a5=(a2*a5);
  a192=1.6666666666666667e+00;
  a43=2.7777777777777776e-02;
  a12=(a3/a12);
  a11=(a11*a12);
  a3=(a3*a11);
  a3=(a3+a10);
  a43=(a43*a3);
  a43=(a2*a43);
  a38=(a38-a43);
  a43=(a192*a38);
  a43=(a8*a43);
  a3=(a18*a43);
  a3=(a5+a3);
  a3=(a7*a3);
  a22=(a2*a22);
  a10=(a26*a43);
  a10=(a22+a10);
  a10=(a24*a10);
  a3=(a3+a10);
  a3=(a3/a19);
  a3=(a3+a43);
  a3=(a28*a3);
  a10=(a31*a43);
  a10=(a5+a10);
  a10=(a29*a10);
  a11=(a35*a43);
  a11=(a22+a11);
  a11=(a33*a11);
  a10=(a10+a11);
  a10=(a10/a32);
  a10=(a10+a43);
  a10=(a36*a10);
  a3=(a3+a10);
  a3=(a3/a37);
  a10=-6.6666666666666670e+00;
  a43=6.6666666666666670e+00;
  a44=(a43*a44);
  a41=(a41*a44);
  a48=(a43*a48);
  a46=(a46*a48);
  a41=(a41+a46);
  a41=(a41/a45);
  a41=(a10-a41);
  a49=(a49*a41);
  a51=(a43*a51);
  a0=(a0*a51);
  a54=(a43*a54);
  a20=(a20*a54);
  a0=(a0+a20);
  a0=(a0/a52);
  a0=(a10-a0);
  a27=(a27*a0);
  a49=(a49+a27);
  a49=(a49/a55);
  a49=(a40*a49);
  a3=(a3+a49);
  a3=(-a3);
  if (res[1]!=0) res[1][3]=a3;
  a17=(a2*a17);
  a17=(a14*a17);
  a17=(a7*a17);
  a25=(a2*a25);
  a25=(a14*a25);
  a25=(a24*a25);
  a17=(a17-a25);
  a17=(a17/a19);
  a17=(a28*a17);
  a30=(a2*a30);
  a30=(a14*a30);
  a30=(a29*a30);
  a34=(a2*a34);
  a14=(a14*a34);
  a14=(a33*a14);
  a30=(a30-a14);
  a30=(a30/a32);
  a30=(a36*a30);
  a17=(a17+a30);
  a17=(a17/a37);
  if (res[1]!=0) res[1][4]=a17;
  a17=-1.8518518518518521e-02;
  a30=1.8518518518518521e-02;
  a14=(a30*a18);
  a14=(a7*a14);
  a34=(a30*a26);
  a34=(a24*a34);
  a14=(a14+a34);
  a14=(a14/a19);
  a14=(a17-a14);
  a14=(a28*a14);
  a34=(a30*a31);
  a34=(a29*a34);
  a25=(a30*a35);
  a25=(a33*a25);
  a34=(a34+a25);
  a34=(a34/a32);
  a34=(a17-a34);
  a34=(a36*a34);
  a14=(a14+a34);
  a14=(a14/a37);
  if (res[1]!=0) res[1][5]=a14;
  a18=(a30*a18);
  a7=(a7*a18);
  a26=(a30*a26);
  a24=(a24*a26);
  a7=(a7+a24);
  a7=(a7/a19);
  a7=(a17-a7);
  a28=(a28*a7);
  a31=(a30*a31);
  a29=(a29*a31);
  a35=(a30*a35);
  a33=(a33*a35);
  a29=(a29+a33);
  a29=(a29/a32);
  a29=(a17-a29);
  a36=(a36*a29);
  a28=(a28+a36);
  a28=(a28/a37);
  if (res[1]!=0) res[1][6]=a28;
  a28=(a77/a81);
  a28=(a85*a28);
  a37=(a39/a88);
  a37=(a65*a37);
  a28=(a28+a37);
  a28=(a28/a91);
  a28=(a40*a28);
  a37=(a56/a60);
  a37=(a66*a37);
  a36=(a67/a70);
  a36=(a74*a36);
  a37=(a37+a36);
  a37=(a37/a75);
  a28=(a28-a37);
  if (res[1]!=0) res[1][7]=a28;
  a28=(a82/a81);
  a28=(a85*a28);
  a37=(a61/a88);
  a37=(a65*a37);
  a28=(a28+a37);
  a28=(a28/a91);
  a28=(a40*a28);
  a37=(a62/a60);
  a37=(a66*a37);
  a36=(a71/a70);
  a36=(a74*a36);
  a37=(a37+a36);
  a37=(a37/a75);
  a28=(a28-a37);
  if (res[1]!=0) res[1][8]=a28;
  a56=(a56+a56);
  a58=sin(a58);
  a28=(a57*a58);
  a28=(a180+a28);
  a28=(a56*a28);
  a62=(a62+a62);
  a63=cos(a63);
  a37=(a57*a63);
  a37=(a4+a37);
  a37=(a62*a37);
  a28=(a28-a37);
  a60=(a60+a60);
  a28=(a28/a60);
  a28=(a66*a28);
  a67=(a67+a67);
  a68=sin(a68);
  a37=(a57*a68);
  a37=(a180+a37);
  a37=(a67*a37);
  a71=(a71+a71);
  a72=cos(a72);
  a36=(a57*a72);
  a36=(a4+a36);
  a36=(a71*a36);
  a37=(a37-a36);
  a70=(a70+a70);
  a37=(a37/a70);
  a37=(a74*a37);
  a28=(a28+a37);
  a28=(a28/a75);
  a77=(a77+a77);
  a79=sin(a79);
  a79=(a78*a79);
  a79=(a77*a79);
  a82=(a82+a82);
  a83=cos(a83);
  a83=(a78*a83);
  a83=(a82*a83);
  a79=(a79-a83);
  a81=(a81+a81);
  a79=(a79/a81);
  a79=(a85*a79);
  a39=(a39+a39);
  a86=sin(a86);
  a86=(a78*a86);
  a86=(a39*a86);
  a61=(a61+a61);
  a89=cos(a89);
  a78=(a78*a89);
  a78=(a61*a78);
  a86=(a86-a78);
  a88=(a88+a88);
  a86=(a86/a88);
  a86=(a65*a86);
  a79=(a79+a86);
  a79=(a79/a91);
  a79=(a40*a79);
  a28=(a28-a79);
  if (res[1]!=0) res[1][9]=a28;
  a28=(a192*a38);
  a28=(a8*a28);
  a79=(a59*a28);
  a79=(a5+a79);
  a79=(a56*a79);
  a86=(a64*a28);
  a86=(a22+a86);
  a86=(a62*a86);
  a79=(a79+a86);
  a79=(a79/a60);
  a79=(a79+a28);
  a79=(a66*a79);
  a86=(a69*a28);
  a86=(a5+a86);
  a86=(a67*a86);
  a78=(a73*a28);
  a78=(a22+a78);
  a78=(a71*a78);
  a86=(a86+a78);
  a86=(a86/a70);
  a86=(a86+a28);
  a86=(a74*a86);
  a79=(a79+a86);
  a79=(a79/a75);
  a80=(a43*a80);
  a77=(a77*a80);
  a84=(a43*a84);
  a82=(a82*a84);
  a77=(a77+a82);
  a77=(a77/a81);
  a77=(a10-a77);
  a85=(a85*a77);
  a87=(a43*a87);
  a39=(a39*a87);
  a90=(a43*a90);
  a61=(a61*a90);
  a39=(a39+a61);
  a39=(a39/a88);
  a39=(a10-a39);
  a65=(a65*a39);
  a85=(a85+a65);
  a85=(a85/a91);
  a85=(a40*a85);
  a79=(a79+a85);
  a79=(-a79);
  if (res[1]!=0) res[1][10]=a79;
  a58=(a2*a58);
  a58=(a57*a58);
  a58=(a56*a58);
  a63=(a2*a63);
  a63=(a57*a63);
  a63=(a62*a63);
  a58=(a58-a63);
  a58=(a58/a60);
  a58=(a66*a58);
  a68=(a2*a68);
  a68=(a57*a68);
  a68=(a67*a68);
  a72=(a2*a72);
  a57=(a57*a72);
  a57=(a71*a57);
  a68=(a68-a57);
  a68=(a68/a70);
  a68=(a74*a68);
  a58=(a58+a68);
  a58=(a58/a75);
  if (res[1]!=0) res[1][11]=a58;
  a58=(a30*a59);
  a58=(a56*a58);
  a68=(a30*a64);
  a68=(a62*a68);
  a58=(a58+a68);
  a58=(a58/a60);
  a58=(a17-a58);
  a58=(a66*a58);
  a68=(a30*a69);
  a68=(a67*a68);
  a57=(a30*a73);
  a57=(a71*a57);
  a68=(a68+a57);
  a68=(a68/a70);
  a68=(a17-a68);
  a68=(a74*a68);
  a58=(a58+a68);
  a58=(a58/a75);
  if (res[1]!=0) res[1][12]=a58;
  a59=(a30*a59);
  a56=(a56*a59);
  a64=(a30*a64);
  a62=(a62*a64);
  a56=(a56+a62);
  a56=(a56/a60);
  a56=(a17-a56);
  a66=(a66*a56);
  a69=(a30*a69);
  a67=(a67*a69);
  a73=(a30*a73);
  a71=(a71*a73);
  a67=(a67+a71);
  a67=(a67/a70);
  a67=(a17-a67);
  a74=(a74*a67);
  a66=(a66+a74);
  a66=(a66/a75);
  if (res[1]!=0) res[1][13]=a66;
  a66=(a113/a117);
  a66=(a121*a66);
  a75=(a76/a124);
  a75=(a101*a75);
  a66=(a66+a75);
  a66=(a66/a127);
  a66=(a40*a66);
  a75=(a92/a96);
  a75=(a102*a75);
  a74=(a103/a106);
  a74=(a110*a74);
  a75=(a75+a74);
  a75=(a75/a111);
  a66=(a66-a75);
  if (res[1]!=0) res[1][14]=a66;
  a66=(a118/a117);
  a66=(a121*a66);
  a75=(a97/a124);
  a75=(a101*a75);
  a66=(a66+a75);
  a66=(a66/a127);
  a66=(a40*a66);
  a75=(a98/a96);
  a75=(a102*a75);
  a74=(a107/a106);
  a74=(a110*a74);
  a75=(a75+a74);
  a75=(a75/a111);
  a66=(a66-a75);
  if (res[1]!=0) res[1][15]=a66;
  a92=(a92+a92);
  a94=sin(a94);
  a66=(a93*a94);
  a66=(a180+a66);
  a66=(a92*a66);
  a98=(a98+a98);
  a99=cos(a99);
  a75=(a93*a99);
  a75=(a4+a75);
  a75=(a98*a75);
  a66=(a66-a75);
  a96=(a96+a96);
  a66=(a66/a96);
  a66=(a102*a66);
  a103=(a103+a103);
  a104=sin(a104);
  a75=(a93*a104);
  a75=(a180+a75);
  a75=(a103*a75);
  a107=(a107+a107);
  a108=cos(a108);
  a74=(a93*a108);
  a74=(a4+a74);
  a74=(a107*a74);
  a75=(a75-a74);
  a106=(a106+a106);
  a75=(a75/a106);
  a75=(a110*a75);
  a66=(a66+a75);
  a66=(a66/a111);
  a113=(a113+a113);
  a115=sin(a115);
  a115=(a114*a115);
  a115=(a113*a115);
  a118=(a118+a118);
  a119=cos(a119);
  a119=(a114*a119);
  a119=(a118*a119);
  a115=(a115-a119);
  a117=(a117+a117);
  a115=(a115/a117);
  a115=(a121*a115);
  a76=(a76+a76);
  a122=sin(a122);
  a122=(a114*a122);
  a122=(a76*a122);
  a97=(a97+a97);
  a125=cos(a125);
  a114=(a114*a125);
  a114=(a97*a114);
  a122=(a122-a114);
  a124=(a124+a124);
  a122=(a122/a124);
  a122=(a101*a122);
  a115=(a115+a122);
  a115=(a115/a127);
  a115=(a40*a115);
  a66=(a66-a115);
  if (res[1]!=0) res[1][16]=a66;
  a66=(a192*a38);
  a66=(a8*a66);
  a115=(a95*a66);
  a115=(a5+a115);
  a115=(a92*a115);
  a122=(a100*a66);
  a122=(a22+a122);
  a122=(a98*a122);
  a115=(a115+a122);
  a115=(a115/a96);
  a115=(a115+a66);
  a115=(a102*a115);
  a122=(a105*a66);
  a122=(a5+a122);
  a122=(a103*a122);
  a114=(a109*a66);
  a114=(a22+a114);
  a114=(a107*a114);
  a122=(a122+a114);
  a122=(a122/a106);
  a122=(a122+a66);
  a122=(a110*a122);
  a115=(a115+a122);
  a115=(a115/a111);
  a116=(a43*a116);
  a113=(a113*a116);
  a120=(a43*a120);
  a118=(a118*a120);
  a113=(a113+a118);
  a113=(a113/a117);
  a113=(a10-a113);
  a121=(a121*a113);
  a123=(a43*a123);
  a76=(a76*a123);
  a126=(a43*a126);
  a97=(a97*a126);
  a76=(a76+a97);
  a76=(a76/a124);
  a76=(a10-a76);
  a101=(a101*a76);
  a121=(a121+a101);
  a121=(a121/a127);
  a121=(a40*a121);
  a115=(a115+a121);
  a115=(-a115);
  if (res[1]!=0) res[1][17]=a115;
  a94=(a2*a94);
  a94=(a93*a94);
  a94=(a92*a94);
  a99=(a2*a99);
  a99=(a93*a99);
  a99=(a98*a99);
  a94=(a94-a99);
  a94=(a94/a96);
  a94=(a102*a94);
  a104=(a2*a104);
  a104=(a93*a104);
  a104=(a103*a104);
  a108=(a2*a108);
  a93=(a93*a108);
  a93=(a107*a93);
  a104=(a104-a93);
  a104=(a104/a106);
  a104=(a110*a104);
  a94=(a94+a104);
  a94=(a94/a111);
  if (res[1]!=0) res[1][18]=a94;
  a94=(a30*a95);
  a94=(a92*a94);
  a104=(a30*a100);
  a104=(a98*a104);
  a94=(a94+a104);
  a94=(a94/a96);
  a94=(a17-a94);
  a94=(a102*a94);
  a104=(a30*a105);
  a104=(a103*a104);
  a93=(a30*a109);
  a93=(a107*a93);
  a104=(a104+a93);
  a104=(a104/a106);
  a104=(a17-a104);
  a104=(a110*a104);
  a94=(a94+a104);
  a94=(a94/a111);
  if (res[1]!=0) res[1][19]=a94;
  a95=(a30*a95);
  a92=(a92*a95);
  a100=(a30*a100);
  a98=(a98*a100);
  a92=(a92+a98);
  a92=(a92/a96);
  a92=(a17-a92);
  a102=(a102*a92);
  a105=(a30*a105);
  a103=(a103*a105);
  a109=(a30*a109);
  a107=(a107*a109);
  a103=(a103+a107);
  a103=(a103/a106);
  a103=(a17-a103);
  a110=(a110*a103);
  a102=(a102+a110);
  a102=(a102/a111);
  if (res[1]!=0) res[1][20]=a102;
  a102=(a149/a153);
  a102=(a157*a102);
  a111=(a112/a160);
  a111=(a137*a111);
  a102=(a102+a111);
  a102=(a102/a163);
  a102=(a40*a102);
  a111=(a128/a132);
  a111=(a138*a111);
  a110=(a139/a142);
  a110=(a146*a110);
  a111=(a111+a110);
  a111=(a111/a147);
  a102=(a102-a111);
  if (res[1]!=0) res[1][21]=a102;
  a102=(a154/a153);
  a102=(a157*a102);
  a111=(a133/a160);
  a111=(a137*a111);
  a102=(a102+a111);
  a102=(a102/a163);
  a102=(a40*a102);
  a111=(a134/a132);
  a111=(a138*a111);
  a110=(a143/a142);
  a110=(a146*a110);
  a111=(a111+a110);
  a111=(a111/a147);
  a102=(a102-a111);
  if (res[1]!=0) res[1][22]=a102;
  a128=(a128+a128);
  a130=sin(a130);
  a102=(a129*a130);
  a102=(a180+a102);
  a102=(a128*a102);
  a134=(a134+a134);
  a135=cos(a135);
  a111=(a129*a135);
  a111=(a4+a111);
  a111=(a134*a111);
  a102=(a102-a111);
  a132=(a132+a132);
  a102=(a102/a132);
  a102=(a138*a102);
  a139=(a139+a139);
  a140=sin(a140);
  a111=(a129*a140);
  a111=(a180+a111);
  a111=(a139*a111);
  a143=(a143+a143);
  a144=cos(a144);
  a110=(a129*a144);
  a110=(a4+a110);
  a110=(a143*a110);
  a111=(a111-a110);
  a142=(a142+a142);
  a111=(a111/a142);
  a111=(a146*a111);
  a102=(a102+a111);
  a102=(a102/a147);
  a149=(a149+a149);
  a151=sin(a151);
  a151=(a150*a151);
  a151=(a149*a151);
  a154=(a154+a154);
  a155=cos(a155);
  a155=(a150*a155);
  a155=(a154*a155);
  a151=(a151-a155);
  a153=(a153+a153);
  a151=(a151/a153);
  a151=(a157*a151);
  a112=(a112+a112);
  a158=sin(a158);
  a158=(a150*a158);
  a158=(a112*a158);
  a133=(a133+a133);
  a161=cos(a161);
  a150=(a150*a161);
  a150=(a133*a150);
  a158=(a158-a150);
  a160=(a160+a160);
  a158=(a158/a160);
  a158=(a137*a158);
  a151=(a151+a158);
  a151=(a151/a163);
  a151=(a40*a151);
  a102=(a102-a151);
  if (res[1]!=0) res[1][23]=a102;
  a102=(a192*a38);
  a102=(a8*a102);
  a151=(a131*a102);
  a151=(a5+a151);
  a151=(a128*a151);
  a158=(a136*a102);
  a158=(a22+a158);
  a158=(a134*a158);
  a151=(a151+a158);
  a151=(a151/a132);
  a151=(a151+a102);
  a151=(a138*a151);
  a158=(a141*a102);
  a158=(a5+a158);
  a158=(a139*a158);
  a150=(a145*a102);
  a150=(a22+a150);
  a150=(a143*a150);
  a158=(a158+a150);
  a158=(a158/a142);
  a158=(a158+a102);
  a158=(a146*a158);
  a151=(a151+a158);
  a151=(a151/a147);
  a152=(a43*a152);
  a149=(a149*a152);
  a156=(a43*a156);
  a154=(a154*a156);
  a149=(a149+a154);
  a149=(a149/a153);
  a149=(a10-a149);
  a157=(a157*a149);
  a159=(a43*a159);
  a112=(a112*a159);
  a162=(a43*a162);
  a133=(a133*a162);
  a112=(a112+a133);
  a112=(a112/a160);
  a112=(a10-a112);
  a137=(a137*a112);
  a157=(a157+a137);
  a157=(a157/a163);
  a157=(a40*a157);
  a151=(a151+a157);
  a151=(-a151);
  if (res[1]!=0) res[1][24]=a151;
  a130=(a2*a130);
  a130=(a129*a130);
  a130=(a128*a130);
  a135=(a2*a135);
  a135=(a129*a135);
  a135=(a134*a135);
  a130=(a130-a135);
  a130=(a130/a132);
  a130=(a138*a130);
  a140=(a2*a140);
  a140=(a129*a140);
  a140=(a139*a140);
  a144=(a2*a144);
  a129=(a129*a144);
  a129=(a143*a129);
  a140=(a140-a129);
  a140=(a140/a142);
  a140=(a146*a140);
  a130=(a130+a140);
  a130=(a130/a147);
  if (res[1]!=0) res[1][25]=a130;
  a130=(a30*a131);
  a130=(a128*a130);
  a140=(a30*a136);
  a140=(a134*a140);
  a130=(a130+a140);
  a130=(a130/a132);
  a130=(a17-a130);
  a130=(a138*a130);
  a140=(a30*a141);
  a140=(a139*a140);
  a129=(a30*a145);
  a129=(a143*a129);
  a140=(a140+a129);
  a140=(a140/a142);
  a140=(a17-a140);
  a140=(a146*a140);
  a130=(a130+a140);
  a130=(a130/a147);
  if (res[1]!=0) res[1][26]=a130;
  a131=(a30*a131);
  a128=(a128*a131);
  a136=(a30*a136);
  a134=(a134*a136);
  a128=(a128+a134);
  a128=(a128/a132);
  a128=(a17-a128);
  a138=(a138*a128);
  a141=(a30*a141);
  a139=(a139*a141);
  a145=(a30*a145);
  a143=(a143*a145);
  a139=(a139+a143);
  a139=(a139/a142);
  a139=(a17-a139);
  a146=(a146*a139);
  a138=(a138+a146);
  a138=(a138/a147);
  if (res[1]!=0) res[1][27]=a138;
  a138=(a181/a184);
  a138=(a188*a138);
  a147=(a148/a190);
  a147=(a172*a147);
  a138=(a138+a147);
  a138=(a138/a191);
  a138=(a40*a138);
  a147=(a164/a167);
  a147=(a173*a147);
  a146=(a6/a176);
  a146=(a178*a146);
  a147=(a147+a146);
  a147=(a147/a179);
  a138=(a138-a147);
  if (res[1]!=0) res[1][28]=a138;
  a138=(a185/a184);
  a138=(a188*a138);
  a147=(a168/a190);
  a147=(a172*a147);
  a138=(a138+a147);
  a138=(a138/a191);
  a138=(a40*a138);
  a147=(a169/a167);
  a147=(a173*a147);
  a146=(a23/a176);
  a146=(a178*a146);
  a147=(a147+a146);
  a147=(a147/a179);
  a138=(a138-a147);
  if (res[1]!=0) res[1][29]=a138;
  a164=(a164+a164);
  a165=sin(a165);
  a138=(a9*a165);
  a138=(a180+a138);
  a138=(a164*a138);
  a169=(a169+a169);
  a170=cos(a170);
  a147=(a9*a170);
  a147=(a4+a147);
  a147=(a169*a147);
  a138=(a138-a147);
  a167=(a167+a167);
  a138=(a138/a167);
  a138=(a173*a138);
  a6=(a6+a6);
  a174=sin(a174);
  a147=(a9*a174);
  a180=(a180+a147);
  a180=(a6*a180);
  a23=(a23+a23);
  a15=cos(a15);
  a147=(a9*a15);
  a4=(a4+a147);
  a4=(a23*a4);
  a180=(a180-a4);
  a176=(a176+a176);
  a180=(a180/a176);
  a180=(a178*a180);
  a138=(a138+a180);
  a138=(a138/a179);
  a181=(a181+a181);
  a182=sin(a182);
  a182=(a13*a182);
  a182=(a181*a182);
  a185=(a185+a185);
  a186=cos(a186);
  a186=(a13*a186);
  a186=(a185*a186);
  a182=(a182-a186);
  a184=(a184+a184);
  a182=(a182/a184);
  a182=(a188*a182);
  a148=(a148+a148);
  a1=sin(a1);
  a1=(a13*a1);
  a1=(a148*a1);
  a168=(a168+a168);
  a16=cos(a16);
  a13=(a13*a16);
  a13=(a168*a13);
  a1=(a1-a13);
  a190=(a190+a190);
  a1=(a1/a190);
  a1=(a172*a1);
  a182=(a182+a1);
  a182=(a182/a191);
  a182=(a40*a182);
  a138=(a138-a182);
  if (res[1]!=0) res[1][30]=a138;
  a192=(a192*a38);
  a8=(a8*a192);
  a192=(a166*a8);
  a192=(a5+a192);
  a192=(a164*a192);
  a38=(a171*a8);
  a38=(a22+a38);
  a38=(a169*a38);
  a192=(a192+a38);
  a192=(a192/a167);
  a192=(a192+a8);
  a192=(a173*a192);
  a38=(a175*a8);
  a5=(a5+a38);
  a5=(a6*a5);
  a38=(a177*a8);
  a22=(a22+a38);
  a22=(a23*a22);
  a5=(a5+a22);
  a5=(a5/a176);
  a5=(a5+a8);
  a5=(a178*a5);
  a192=(a192+a5);
  a192=(a192/a179);
  a183=(a43*a183);
  a181=(a181*a183);
  a187=(a43*a187);
  a185=(a185*a187);
  a181=(a181+a185);
  a181=(a181/a184);
  a181=(a10-a181);
  a188=(a188*a181);
  a189=(a43*a189);
  a148=(a148*a189);
  a43=(a43*a21);
  a168=(a168*a43);
  a148=(a148+a168);
  a148=(a148/a190);
  a10=(a10-a148);
  a172=(a172*a10);
  a188=(a188+a172);
  a188=(a188/a191);
  a40=(a40*a188);
  a192=(a192+a40);
  a192=(-a192);
  if (res[1]!=0) res[1][31]=a192;
  a165=(a2*a165);
  a165=(a9*a165);
  a165=(a164*a165);
  a170=(a2*a170);
  a170=(a9*a170);
  a170=(a169*a170);
  a165=(a165-a170);
  a165=(a165/a167);
  a165=(a173*a165);
  a174=(a2*a174);
  a174=(a9*a174);
  a174=(a6*a174);
  a2=(a2*a15);
  a9=(a9*a2);
  a9=(a23*a9);
  a174=(a174-a9);
  a174=(a174/a176);
  a174=(a178*a174);
  a165=(a165+a174);
  a165=(a165/a179);
  if (res[1]!=0) res[1][32]=a165;
  a165=(a30*a166);
  a165=(a164*a165);
  a174=(a30*a171);
  a174=(a169*a174);
  a165=(a165+a174);
  a165=(a165/a167);
  a165=(a17-a165);
  a165=(a173*a165);
  a174=(a30*a175);
  a174=(a6*a174);
  a9=(a30*a177);
  a9=(a23*a9);
  a174=(a174+a9);
  a174=(a174/a176);
  a174=(a17-a174);
  a174=(a178*a174);
  a165=(a165+a174);
  a165=(a165/a179);
  if (res[1]!=0) res[1][33]=a165;
  a166=(a30*a166);
  a164=(a164*a166);
  a171=(a30*a171);
  a169=(a169*a171);
  a164=(a164+a169);
  a164=(a164/a167);
  a164=(a17-a164);
  a173=(a173*a164);
  a175=(a30*a175);
  a6=(a6*a175);
  a30=(a30*a177);
  a23=(a23*a30);
  a6=(a6+a23);
  a6=(a6/a176);
  a17=(a17-a6);
  a178=(a178*a17);
  a173=(a173+a178);
  a173=(a173/a179);
  if (res[1]!=0) res[1][34]=a173;
  return 0;
}

CASADI_SYMBOL_EXPORT int heron_constr_h_fun_jac_uxt_zt(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int heron_constr_h_fun_jac_uxt_zt_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int heron_constr_h_fun_jac_uxt_zt_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void heron_constr_h_fun_jac_uxt_zt_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int heron_constr_h_fun_jac_uxt_zt_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void heron_constr_h_fun_jac_uxt_zt_release(int mem) {
}

CASADI_SYMBOL_EXPORT void heron_constr_h_fun_jac_uxt_zt_incref(void) {
}

CASADI_SYMBOL_EXPORT void heron_constr_h_fun_jac_uxt_zt_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int heron_constr_h_fun_jac_uxt_zt_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int heron_constr_h_fun_jac_uxt_zt_n_out(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_real heron_constr_h_fun_jac_uxt_zt_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* heron_constr_h_fun_jac_uxt_zt_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* heron_constr_h_fun_jac_uxt_zt_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* heron_constr_h_fun_jac_uxt_zt_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* heron_constr_h_fun_jac_uxt_zt_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    case 1: return casadi_s5;
    case 2: return casadi_s6;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int heron_constr_h_fun_jac_uxt_zt_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 3;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
