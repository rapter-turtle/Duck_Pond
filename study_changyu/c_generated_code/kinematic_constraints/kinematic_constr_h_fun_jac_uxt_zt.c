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
  #define CASADI_PREFIX(ID) kinematic_constr_h_fun_jac_uxt_zt_ ## ID
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

static const casadi_int casadi_s0[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s1[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[29] = {25, 1, 0, 25, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
static const casadi_int casadi_s4[9] = {5, 1, 0, 5, 0, 1, 2, 3, 4};
static const casadi_int casadi_s5[38] = {8, 5, 0, 6, 12, 18, 24, 30, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7};
static const casadi_int casadi_s6[3] = {5, 0, 0};

/* kinematic_constr_h_fun_jac_uxt_zt:(i0[6],i1[2],i2[],i3[25])->(o0[5],o1[8x5,30nz],o2[5x0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a100, a101, a102, a103, a104, a105, a106, a107, a108, a109, a11, a110, a111, a112, a113, a114, a115, a116, a117, a118, a119, a12, a120, a121, a122, a123, a124, a125, a126, a127, a128, a129, a13, a130, a131, a132, a133, a134, a135, a136, a137, a138, a139, a14, a140, a141, a142, a143, a144, a145, a146, a147, a148, a149, a15, a150, a151, a152, a153, a154, a155, a156, a157, a158, a159, a16, a160, a161, a162, a163, a164, a165, a166, a167, a168, a169, a17, a170, a171, a172, a173, a174, a175, a176, a177, a178, a179, a18, a180, a181, a182, a183, a184, a185, a186, a187, a188, a189, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a8, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a9, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99;
  a0=arg[3]? arg[3][0] : 0;
  a1=1.0000000000000001e-01;
  a2=arg[3]? arg[3][3] : 0;
  a2=(a1*a2);
  a2=(a0+a2);
  a3=arg[0]? arg[0][0] : 0;
  a4=arg[0]? arg[0][3] : 0;
  a5=arg[0]? arg[0][2] : 0;
  a6=cos(a5);
  a7=(a4*a6);
  a7=(a1*a7);
  a7=(a3+a7);
  a8=(a2-a7);
  a9=3.;
  a10=arg[0]? arg[0][5] : 0;
  a10=(a1*a10);
  a10=(a4+a10);
  a11=2.0000000000000001e-01;
  a12=(a10/a11);
  a12=(a9*a12);
  a13=arg[0]? arg[0][4] : 0;
  a13=(a1*a13);
  a13=(a5+a13);
  a14=1.5707963267948966e+00;
  a15=(a13-a14);
  a16=cos(a15);
  a17=(a12*a16);
  a8=(a8-a17);
  a17=casadi_sq(a8);
  a18=arg[3]? arg[3][1] : 0;
  a19=arg[3]? arg[3][4] : 0;
  a19=(a1*a19);
  a19=(a18+a19);
  a20=arg[0]? arg[0][1] : 0;
  a21=sin(a5);
  a22=(a4*a21);
  a22=(a1*a22);
  a22=(a20+a22);
  a23=(a19-a22);
  a24=(a13-a14);
  a25=sin(a24);
  a26=(a12*a25);
  a23=(a23-a26);
  a26=casadi_sq(a23);
  a17=(a17+a26);
  a17=sqrt(a17);
  a26=arg[3]? arg[3][2] : 0;
  a27=(a26+a12);
  a27=(a17-a27);
  a27=exp(a27);
  a2=(a2-a7);
  a28=(a13+a14);
  a29=cos(a28);
  a30=(a12*a29);
  a2=(a2-a30);
  a30=casadi_sq(a2);
  a19=(a19-a22);
  a31=(a13+a14);
  a32=sin(a31);
  a33=(a12*a32);
  a19=(a19-a33);
  a33=casadi_sq(a19);
  a30=(a30+a33);
  a30=sqrt(a30);
  a33=(a26+a12);
  a33=(a30-a33);
  a33=exp(a33);
  a34=(a27+a33);
  a35=2.;
  a34=(a34/a35);
  a36=log(a34);
  a37=9.6999999999999997e-01;
  a38=(a0-a3);
  a39=(a4/a11);
  a39=(a9*a39);
  a40=(a5-a14);
  a41=cos(a40);
  a42=(a39*a41);
  a38=(a38-a42);
  a42=casadi_sq(a38);
  a43=(a18-a20);
  a44=(a5-a14);
  a45=sin(a44);
  a46=(a39*a45);
  a43=(a43-a46);
  a46=casadi_sq(a43);
  a42=(a42+a46);
  a42=sqrt(a42);
  a46=(a26+a39);
  a46=(a42-a46);
  a46=exp(a46);
  a0=(a0-a3);
  a47=(a5+a14);
  a48=cos(a47);
  a49=(a39*a48);
  a0=(a0-a49);
  a49=casadi_sq(a0);
  a18=(a18-a20);
  a50=(a5+a14);
  a51=sin(a50);
  a52=(a39*a51);
  a18=(a18-a52);
  a52=casadi_sq(a18);
  a49=(a49+a52);
  a49=sqrt(a49);
  a26=(a26+a39);
  a26=(a49-a26);
  a26=exp(a26);
  a52=(a46+a26);
  a52=(a52/a35);
  a53=log(a52);
  a53=(a37*a53);
  a36=(a36-a53);
  if (res[0]!=0) res[0][0]=a36;
  a36=arg[3]? arg[3][5] : 0;
  a53=arg[3]? arg[3][8] : 0;
  a53=(a1*a53);
  a53=(a36+a53);
  a54=(a53-a7);
  a55=(a10/a11);
  a55=(a9*a55);
  a56=(a13-a14);
  a57=cos(a56);
  a58=(a55*a57);
  a54=(a54-a58);
  a58=casadi_sq(a54);
  a59=arg[3]? arg[3][6] : 0;
  a60=arg[3]? arg[3][9] : 0;
  a60=(a1*a60);
  a60=(a59+a60);
  a61=(a60-a22);
  a62=(a13-a14);
  a63=sin(a62);
  a64=(a55*a63);
  a61=(a61-a64);
  a64=casadi_sq(a61);
  a58=(a58+a64);
  a58=sqrt(a58);
  a64=arg[3]? arg[3][7] : 0;
  a65=(a64+a55);
  a65=(a58-a65);
  a65=exp(a65);
  a53=(a53-a7);
  a66=(a13+a14);
  a67=cos(a66);
  a68=(a55*a67);
  a53=(a53-a68);
  a68=casadi_sq(a53);
  a60=(a60-a22);
  a69=(a13+a14);
  a70=sin(a69);
  a71=(a55*a70);
  a60=(a60-a71);
  a71=casadi_sq(a60);
  a68=(a68+a71);
  a68=sqrt(a68);
  a71=(a64+a55);
  a71=(a68-a71);
  a71=exp(a71);
  a72=(a65+a71);
  a72=(a72/a35);
  a73=log(a72);
  a74=(a36-a3);
  a75=(a4/a11);
  a75=(a9*a75);
  a76=(a5-a14);
  a77=cos(a76);
  a78=(a75*a77);
  a74=(a74-a78);
  a78=casadi_sq(a74);
  a79=(a59-a20);
  a80=(a5-a14);
  a81=sin(a80);
  a82=(a75*a81);
  a79=(a79-a82);
  a82=casadi_sq(a79);
  a78=(a78+a82);
  a78=sqrt(a78);
  a82=(a64+a75);
  a82=(a78-a82);
  a82=exp(a82);
  a36=(a36-a3);
  a83=(a5+a14);
  a84=cos(a83);
  a85=(a75*a84);
  a36=(a36-a85);
  a85=casadi_sq(a36);
  a59=(a59-a20);
  a86=(a5+a14);
  a87=sin(a86);
  a88=(a75*a87);
  a59=(a59-a88);
  a88=casadi_sq(a59);
  a85=(a85+a88);
  a85=sqrt(a85);
  a64=(a64+a75);
  a64=(a85-a64);
  a64=exp(a64);
  a88=(a82+a64);
  a88=(a88/a35);
  a89=log(a88);
  a89=(a37*a89);
  a73=(a73-a89);
  if (res[0]!=0) res[0][1]=a73;
  a73=arg[3]? arg[3][10] : 0;
  a89=arg[3]? arg[3][13] : 0;
  a89=(a1*a89);
  a89=(a73+a89);
  a90=(a89-a7);
  a91=(a10/a11);
  a91=(a9*a91);
  a92=(a13-a14);
  a93=cos(a92);
  a94=(a91*a93);
  a90=(a90-a94);
  a94=casadi_sq(a90);
  a95=arg[3]? arg[3][11] : 0;
  a96=arg[3]? arg[3][14] : 0;
  a96=(a1*a96);
  a96=(a95+a96);
  a97=(a96-a22);
  a98=(a13-a14);
  a99=sin(a98);
  a100=(a91*a99);
  a97=(a97-a100);
  a100=casadi_sq(a97);
  a94=(a94+a100);
  a94=sqrt(a94);
  a100=arg[3]? arg[3][12] : 0;
  a101=(a100+a91);
  a101=(a94-a101);
  a101=exp(a101);
  a89=(a89-a7);
  a102=(a13+a14);
  a103=cos(a102);
  a104=(a91*a103);
  a89=(a89-a104);
  a104=casadi_sq(a89);
  a96=(a96-a22);
  a105=(a13+a14);
  a106=sin(a105);
  a107=(a91*a106);
  a96=(a96-a107);
  a107=casadi_sq(a96);
  a104=(a104+a107);
  a104=sqrt(a104);
  a107=(a100+a91);
  a107=(a104-a107);
  a107=exp(a107);
  a108=(a101+a107);
  a108=(a108/a35);
  a109=log(a108);
  a110=(a73-a3);
  a111=(a4/a11);
  a111=(a9*a111);
  a112=(a5-a14);
  a113=cos(a112);
  a114=(a111*a113);
  a110=(a110-a114);
  a114=casadi_sq(a110);
  a115=(a95-a20);
  a116=(a5-a14);
  a117=sin(a116);
  a118=(a111*a117);
  a115=(a115-a118);
  a118=casadi_sq(a115);
  a114=(a114+a118);
  a114=sqrt(a114);
  a118=(a100+a111);
  a118=(a114-a118);
  a118=exp(a118);
  a73=(a73-a3);
  a119=(a5+a14);
  a120=cos(a119);
  a121=(a111*a120);
  a73=(a73-a121);
  a121=casadi_sq(a73);
  a95=(a95-a20);
  a122=(a5+a14);
  a123=sin(a122);
  a124=(a111*a123);
  a95=(a95-a124);
  a124=casadi_sq(a95);
  a121=(a121+a124);
  a121=sqrt(a121);
  a100=(a100+a111);
  a100=(a121-a100);
  a100=exp(a100);
  a124=(a118+a100);
  a124=(a124/a35);
  a125=log(a124);
  a125=(a37*a125);
  a109=(a109-a125);
  if (res[0]!=0) res[0][2]=a109;
  a109=arg[3]? arg[3][15] : 0;
  a125=arg[3]? arg[3][18] : 0;
  a125=(a1*a125);
  a125=(a109+a125);
  a126=(a125-a7);
  a127=(a10/a11);
  a127=(a9*a127);
  a128=(a13-a14);
  a129=cos(a128);
  a130=(a127*a129);
  a126=(a126-a130);
  a130=casadi_sq(a126);
  a131=arg[3]? arg[3][16] : 0;
  a132=arg[3]? arg[3][19] : 0;
  a132=(a1*a132);
  a132=(a131+a132);
  a133=(a132-a22);
  a134=(a13-a14);
  a135=sin(a134);
  a136=(a127*a135);
  a133=(a133-a136);
  a136=casadi_sq(a133);
  a130=(a130+a136);
  a130=sqrt(a130);
  a136=arg[3]? arg[3][17] : 0;
  a137=(a136+a127);
  a137=(a130-a137);
  a137=exp(a137);
  a125=(a125-a7);
  a138=(a13+a14);
  a139=cos(a138);
  a140=(a127*a139);
  a125=(a125-a140);
  a140=casadi_sq(a125);
  a132=(a132-a22);
  a141=(a13+a14);
  a142=sin(a141);
  a143=(a127*a142);
  a132=(a132-a143);
  a143=casadi_sq(a132);
  a140=(a140+a143);
  a140=sqrt(a140);
  a143=(a136+a127);
  a143=(a140-a143);
  a143=exp(a143);
  a144=(a137+a143);
  a144=(a144/a35);
  a145=log(a144);
  a146=(a109-a3);
  a147=(a4/a11);
  a147=(a9*a147);
  a148=(a5-a14);
  a149=cos(a148);
  a150=(a147*a149);
  a146=(a146-a150);
  a150=casadi_sq(a146);
  a151=(a131-a20);
  a152=(a5-a14);
  a153=sin(a152);
  a154=(a147*a153);
  a151=(a151-a154);
  a154=casadi_sq(a151);
  a150=(a150+a154);
  a150=sqrt(a150);
  a154=(a136+a147);
  a154=(a150-a154);
  a154=exp(a154);
  a109=(a109-a3);
  a155=(a5+a14);
  a156=cos(a155);
  a157=(a147*a156);
  a109=(a109-a157);
  a157=casadi_sq(a109);
  a131=(a131-a20);
  a158=(a5+a14);
  a159=sin(a158);
  a160=(a147*a159);
  a131=(a131-a160);
  a160=casadi_sq(a131);
  a157=(a157+a160);
  a157=sqrt(a157);
  a136=(a136+a147);
  a136=(a157-a136);
  a136=exp(a136);
  a160=(a154+a136);
  a160=(a160/a35);
  a161=log(a160);
  a161=(a37*a161);
  a145=(a145-a161);
  if (res[0]!=0) res[0][3]=a145;
  a145=arg[3]? arg[3][20] : 0;
  a161=arg[3]? arg[3][23] : 0;
  a161=(a1*a161);
  a161=(a145+a161);
  a162=(a161-a7);
  a10=(a10/a11);
  a10=(a9*a10);
  a163=(a13-a14);
  a164=cos(a163);
  a165=(a10*a164);
  a162=(a162-a165);
  a165=casadi_sq(a162);
  a166=arg[3]? arg[3][21] : 0;
  a167=arg[3]? arg[3][24] : 0;
  a167=(a1*a167);
  a167=(a166+a167);
  a168=(a167-a22);
  a169=(a13-a14);
  a170=sin(a169);
  a171=(a10*a170);
  a168=(a168-a171);
  a171=casadi_sq(a168);
  a165=(a165+a171);
  a165=sqrt(a165);
  a171=arg[3]? arg[3][22] : 0;
  a172=(a171+a10);
  a172=(a165-a172);
  a172=exp(a172);
  a161=(a161-a7);
  a7=(a13+a14);
  a173=cos(a7);
  a174=(a10*a173);
  a161=(a161-a174);
  a174=casadi_sq(a161);
  a167=(a167-a22);
  a13=(a13+a14);
  a22=sin(a13);
  a175=(a10*a22);
  a167=(a167-a175);
  a175=casadi_sq(a167);
  a174=(a174+a175);
  a174=sqrt(a174);
  a175=(a171+a10);
  a175=(a174-a175);
  a175=exp(a175);
  a176=(a172+a175);
  a176=(a176/a35);
  a177=log(a176);
  a178=(a145-a3);
  a11=(a4/a11);
  a9=(a9*a11);
  a11=(a5-a14);
  a179=cos(a11);
  a180=(a9*a179);
  a178=(a178-a180);
  a180=casadi_sq(a178);
  a181=(a166-a20);
  a182=(a5-a14);
  a183=sin(a182);
  a184=(a9*a183);
  a181=(a181-a184);
  a184=casadi_sq(a181);
  a180=(a180+a184);
  a180=sqrt(a180);
  a184=(a171+a9);
  a184=(a180-a184);
  a184=exp(a184);
  a145=(a145-a3);
  a3=(a5+a14);
  a185=cos(a3);
  a186=(a9*a185);
  a145=(a145-a186);
  a186=casadi_sq(a145);
  a166=(a166-a20);
  a14=(a5+a14);
  a20=sin(a14);
  a187=(a9*a20);
  a166=(a166-a187);
  a187=casadi_sq(a166);
  a186=(a186+a187);
  a186=sqrt(a186);
  a171=(a171+a9);
  a171=(a186-a171);
  a171=exp(a171);
  a187=(a184+a171);
  a187=(a187/a35);
  a35=log(a187);
  a35=(a37*a35);
  a177=(a177-a35);
  if (res[0]!=0) res[0][4]=a177;
  a177=5.0000000000000000e-01;
  a35=(a38/a42);
  a35=(a46*a35);
  a188=(a0/a49);
  a188=(a26*a188);
  a35=(a35+a188);
  a35=(a177*a35);
  a35=(a35/a52);
  a35=(a37*a35);
  a188=(a8/a17);
  a188=(a27*a188);
  a189=(a2/a30);
  a189=(a33*a189);
  a188=(a188+a189);
  a188=(a177*a188);
  a188=(a188/a34);
  a35=(a35-a188);
  if (res[1]!=0) res[1][0]=a35;
  a35=(a43/a42);
  a35=(a46*a35);
  a188=(a18/a49);
  a188=(a26*a188);
  a35=(a35+a188);
  a35=(a177*a35);
  a35=(a35/a52);
  a35=(a37*a35);
  a188=(a23/a17);
  a188=(a27*a188);
  a189=(a19/a30);
  a189=(a33*a189);
  a188=(a188+a189);
  a188=(a177*a188);
  a188=(a188/a34);
  a35=(a35-a188);
  if (res[1]!=0) res[1][1]=a35;
  a8=(a8+a8);
  a35=sin(a5);
  a35=(a4*a35);
  a35=(a1*a35);
  a15=sin(a15);
  a188=(a12*a15);
  a188=(a35+a188);
  a188=(a8*a188);
  a23=(a23+a23);
  a5=cos(a5);
  a4=(a4*a5);
  a4=(a1*a4);
  a24=cos(a24);
  a5=(a12*a24);
  a5=(a4+a5);
  a5=(a23*a5);
  a188=(a188-a5);
  a17=(a17+a17);
  a188=(a188/a17);
  a188=(a27*a188);
  a2=(a2+a2);
  a28=sin(a28);
  a5=(a12*a28);
  a5=(a35+a5);
  a5=(a2*a5);
  a19=(a19+a19);
  a31=cos(a31);
  a189=(a12*a31);
  a189=(a4+a189);
  a189=(a19*a189);
  a5=(a5-a189);
  a30=(a30+a30);
  a5=(a5/a30);
  a5=(a33*a5);
  a188=(a188+a5);
  a188=(a177*a188);
  a188=(a188/a34);
  a38=(a38+a38);
  a40=sin(a40);
  a40=(a39*a40);
  a40=(a38*a40);
  a43=(a43+a43);
  a44=cos(a44);
  a44=(a39*a44);
  a44=(a43*a44);
  a40=(a40-a44);
  a42=(a42+a42);
  a40=(a40/a42);
  a40=(a46*a40);
  a0=(a0+a0);
  a47=sin(a47);
  a47=(a39*a47);
  a47=(a0*a47);
  a18=(a18+a18);
  a50=cos(a50);
  a39=(a39*a50);
  a39=(a18*a39);
  a47=(a47-a39);
  a49=(a49+a49);
  a47=(a47/a49);
  a47=(a26*a47);
  a40=(a40+a47);
  a40=(a177*a40);
  a40=(a40/a52);
  a40=(a37*a40);
  a188=(a188-a40);
  if (res[1]!=0) res[1][2]=a188;
  a188=-15.;
  a6=(a1*a6);
  a40=15.;
  a47=(a40*a16);
  a47=(a6+a47);
  a47=(a8*a47);
  a21=(a1*a21);
  a39=(a40*a25);
  a39=(a21+a39);
  a39=(a23*a39);
  a47=(a47+a39);
  a47=(a47/a17);
  a47=(a188-a47);
  a47=(a27*a47);
  a39=(a40*a29);
  a39=(a6+a39);
  a39=(a2*a39);
  a50=(a40*a32);
  a50=(a21+a50);
  a50=(a19*a50);
  a39=(a39+a50);
  a39=(a39/a30);
  a39=(a188-a39);
  a39=(a33*a39);
  a47=(a47+a39);
  a47=(a177*a47);
  a47=(a47/a34);
  a41=(a40*a41);
  a38=(a38*a41);
  a45=(a40*a45);
  a43=(a43*a45);
  a38=(a38+a43);
  a38=(a38/a42);
  a38=(a188-a38);
  a46=(a46*a38);
  a48=(a40*a48);
  a0=(a0*a48);
  a51=(a40*a51);
  a18=(a18*a51);
  a0=(a0+a18);
  a0=(a0/a49);
  a0=(a188-a0);
  a26=(a26*a0);
  a46=(a46+a26);
  a46=(a177*a46);
  a46=(a46/a52);
  a46=(a37*a46);
  a47=(a47-a46);
  if (res[1]!=0) res[1][3]=a47;
  a15=(a1*a15);
  a15=(a12*a15);
  a15=(a8*a15);
  a24=(a1*a24);
  a24=(a12*a24);
  a24=(a23*a24);
  a15=(a15-a24);
  a15=(a15/a17);
  a15=(a27*a15);
  a28=(a1*a28);
  a28=(a12*a28);
  a28=(a2*a28);
  a31=(a1*a31);
  a12=(a12*a31);
  a12=(a19*a12);
  a28=(a28-a12);
  a28=(a28/a30);
  a28=(a33*a28);
  a15=(a15+a28);
  a15=(a177*a15);
  a15=(a15/a34);
  if (res[1]!=0) res[1][4]=a15;
  a15=-1.5000000000000000e+00;
  a28=1.5000000000000000e+00;
  a16=(a28*a16);
  a8=(a8*a16);
  a25=(a28*a25);
  a23=(a23*a25);
  a8=(a8+a23);
  a8=(a8/a17);
  a8=(a15-a8);
  a27=(a27*a8);
  a29=(a28*a29);
  a2=(a2*a29);
  a32=(a28*a32);
  a19=(a19*a32);
  a2=(a2+a19);
  a2=(a2/a30);
  a2=(a15-a2);
  a33=(a33*a2);
  a27=(a27+a33);
  a27=(a177*a27);
  a27=(a27/a34);
  if (res[1]!=0) res[1][5]=a27;
  a27=(a74/a78);
  a27=(a82*a27);
  a34=(a36/a85);
  a34=(a64*a34);
  a27=(a27+a34);
  a27=(a177*a27);
  a27=(a27/a88);
  a27=(a37*a27);
  a34=(a54/a58);
  a34=(a65*a34);
  a33=(a53/a68);
  a33=(a71*a33);
  a34=(a34+a33);
  a34=(a177*a34);
  a34=(a34/a72);
  a27=(a27-a34);
  if (res[1]!=0) res[1][6]=a27;
  a27=(a79/a78);
  a27=(a82*a27);
  a34=(a59/a85);
  a34=(a64*a34);
  a27=(a27+a34);
  a27=(a177*a27);
  a27=(a27/a88);
  a27=(a37*a27);
  a34=(a61/a58);
  a34=(a65*a34);
  a33=(a60/a68);
  a33=(a71*a33);
  a34=(a34+a33);
  a34=(a177*a34);
  a34=(a34/a72);
  a27=(a27-a34);
  if (res[1]!=0) res[1][7]=a27;
  a54=(a54+a54);
  a56=sin(a56);
  a27=(a55*a56);
  a27=(a35+a27);
  a27=(a54*a27);
  a61=(a61+a61);
  a62=cos(a62);
  a34=(a55*a62);
  a34=(a4+a34);
  a34=(a61*a34);
  a27=(a27-a34);
  a58=(a58+a58);
  a27=(a27/a58);
  a27=(a65*a27);
  a53=(a53+a53);
  a66=sin(a66);
  a34=(a55*a66);
  a34=(a35+a34);
  a34=(a53*a34);
  a60=(a60+a60);
  a69=cos(a69);
  a33=(a55*a69);
  a33=(a4+a33);
  a33=(a60*a33);
  a34=(a34-a33);
  a68=(a68+a68);
  a34=(a34/a68);
  a34=(a71*a34);
  a27=(a27+a34);
  a27=(a177*a27);
  a27=(a27/a72);
  a74=(a74+a74);
  a76=sin(a76);
  a76=(a75*a76);
  a76=(a74*a76);
  a79=(a79+a79);
  a80=cos(a80);
  a80=(a75*a80);
  a80=(a79*a80);
  a76=(a76-a80);
  a78=(a78+a78);
  a76=(a76/a78);
  a76=(a82*a76);
  a36=(a36+a36);
  a83=sin(a83);
  a83=(a75*a83);
  a83=(a36*a83);
  a59=(a59+a59);
  a86=cos(a86);
  a75=(a75*a86);
  a75=(a59*a75);
  a83=(a83-a75);
  a85=(a85+a85);
  a83=(a83/a85);
  a83=(a64*a83);
  a76=(a76+a83);
  a76=(a177*a76);
  a76=(a76/a88);
  a76=(a37*a76);
  a27=(a27-a76);
  if (res[1]!=0) res[1][8]=a27;
  a27=(a40*a57);
  a27=(a6+a27);
  a27=(a54*a27);
  a76=(a40*a63);
  a76=(a21+a76);
  a76=(a61*a76);
  a27=(a27+a76);
  a27=(a27/a58);
  a27=(a188-a27);
  a27=(a65*a27);
  a76=(a40*a67);
  a76=(a6+a76);
  a76=(a53*a76);
  a83=(a40*a70);
  a83=(a21+a83);
  a83=(a60*a83);
  a76=(a76+a83);
  a76=(a76/a68);
  a76=(a188-a76);
  a76=(a71*a76);
  a27=(a27+a76);
  a27=(a177*a27);
  a27=(a27/a72);
  a77=(a40*a77);
  a74=(a74*a77);
  a81=(a40*a81);
  a79=(a79*a81);
  a74=(a74+a79);
  a74=(a74/a78);
  a74=(a188-a74);
  a82=(a82*a74);
  a84=(a40*a84);
  a36=(a36*a84);
  a87=(a40*a87);
  a59=(a59*a87);
  a36=(a36+a59);
  a36=(a36/a85);
  a36=(a188-a36);
  a64=(a64*a36);
  a82=(a82+a64);
  a82=(a177*a82);
  a82=(a82/a88);
  a82=(a37*a82);
  a27=(a27-a82);
  if (res[1]!=0) res[1][9]=a27;
  a56=(a1*a56);
  a56=(a55*a56);
  a56=(a54*a56);
  a62=(a1*a62);
  a62=(a55*a62);
  a62=(a61*a62);
  a56=(a56-a62);
  a56=(a56/a58);
  a56=(a65*a56);
  a66=(a1*a66);
  a66=(a55*a66);
  a66=(a53*a66);
  a69=(a1*a69);
  a55=(a55*a69);
  a55=(a60*a55);
  a66=(a66-a55);
  a66=(a66/a68);
  a66=(a71*a66);
  a56=(a56+a66);
  a56=(a177*a56);
  a56=(a56/a72);
  if (res[1]!=0) res[1][10]=a56;
  a57=(a28*a57);
  a54=(a54*a57);
  a63=(a28*a63);
  a61=(a61*a63);
  a54=(a54+a61);
  a54=(a54/a58);
  a54=(a15-a54);
  a65=(a65*a54);
  a67=(a28*a67);
  a53=(a53*a67);
  a70=(a28*a70);
  a60=(a60*a70);
  a53=(a53+a60);
  a53=(a53/a68);
  a53=(a15-a53);
  a71=(a71*a53);
  a65=(a65+a71);
  a65=(a177*a65);
  a65=(a65/a72);
  if (res[1]!=0) res[1][11]=a65;
  a65=(a110/a114);
  a65=(a118*a65);
  a72=(a73/a121);
  a72=(a100*a72);
  a65=(a65+a72);
  a65=(a177*a65);
  a65=(a65/a124);
  a65=(a37*a65);
  a72=(a90/a94);
  a72=(a101*a72);
  a71=(a89/a104);
  a71=(a107*a71);
  a72=(a72+a71);
  a72=(a177*a72);
  a72=(a72/a108);
  a65=(a65-a72);
  if (res[1]!=0) res[1][12]=a65;
  a65=(a115/a114);
  a65=(a118*a65);
  a72=(a95/a121);
  a72=(a100*a72);
  a65=(a65+a72);
  a65=(a177*a65);
  a65=(a65/a124);
  a65=(a37*a65);
  a72=(a97/a94);
  a72=(a101*a72);
  a71=(a96/a104);
  a71=(a107*a71);
  a72=(a72+a71);
  a72=(a177*a72);
  a72=(a72/a108);
  a65=(a65-a72);
  if (res[1]!=0) res[1][13]=a65;
  a90=(a90+a90);
  a92=sin(a92);
  a65=(a91*a92);
  a65=(a35+a65);
  a65=(a90*a65);
  a97=(a97+a97);
  a98=cos(a98);
  a72=(a91*a98);
  a72=(a4+a72);
  a72=(a97*a72);
  a65=(a65-a72);
  a94=(a94+a94);
  a65=(a65/a94);
  a65=(a101*a65);
  a89=(a89+a89);
  a102=sin(a102);
  a72=(a91*a102);
  a72=(a35+a72);
  a72=(a89*a72);
  a96=(a96+a96);
  a105=cos(a105);
  a71=(a91*a105);
  a71=(a4+a71);
  a71=(a96*a71);
  a72=(a72-a71);
  a104=(a104+a104);
  a72=(a72/a104);
  a72=(a107*a72);
  a65=(a65+a72);
  a65=(a177*a65);
  a65=(a65/a108);
  a110=(a110+a110);
  a112=sin(a112);
  a112=(a111*a112);
  a112=(a110*a112);
  a115=(a115+a115);
  a116=cos(a116);
  a116=(a111*a116);
  a116=(a115*a116);
  a112=(a112-a116);
  a114=(a114+a114);
  a112=(a112/a114);
  a112=(a118*a112);
  a73=(a73+a73);
  a119=sin(a119);
  a119=(a111*a119);
  a119=(a73*a119);
  a95=(a95+a95);
  a122=cos(a122);
  a111=(a111*a122);
  a111=(a95*a111);
  a119=(a119-a111);
  a121=(a121+a121);
  a119=(a119/a121);
  a119=(a100*a119);
  a112=(a112+a119);
  a112=(a177*a112);
  a112=(a112/a124);
  a112=(a37*a112);
  a65=(a65-a112);
  if (res[1]!=0) res[1][14]=a65;
  a65=(a40*a93);
  a65=(a6+a65);
  a65=(a90*a65);
  a112=(a40*a99);
  a112=(a21+a112);
  a112=(a97*a112);
  a65=(a65+a112);
  a65=(a65/a94);
  a65=(a188-a65);
  a65=(a101*a65);
  a112=(a40*a103);
  a112=(a6+a112);
  a112=(a89*a112);
  a119=(a40*a106);
  a119=(a21+a119);
  a119=(a96*a119);
  a112=(a112+a119);
  a112=(a112/a104);
  a112=(a188-a112);
  a112=(a107*a112);
  a65=(a65+a112);
  a65=(a177*a65);
  a65=(a65/a108);
  a113=(a40*a113);
  a110=(a110*a113);
  a117=(a40*a117);
  a115=(a115*a117);
  a110=(a110+a115);
  a110=(a110/a114);
  a110=(a188-a110);
  a118=(a118*a110);
  a120=(a40*a120);
  a73=(a73*a120);
  a123=(a40*a123);
  a95=(a95*a123);
  a73=(a73+a95);
  a73=(a73/a121);
  a73=(a188-a73);
  a100=(a100*a73);
  a118=(a118+a100);
  a118=(a177*a118);
  a118=(a118/a124);
  a118=(a37*a118);
  a65=(a65-a118);
  if (res[1]!=0) res[1][15]=a65;
  a92=(a1*a92);
  a92=(a91*a92);
  a92=(a90*a92);
  a98=(a1*a98);
  a98=(a91*a98);
  a98=(a97*a98);
  a92=(a92-a98);
  a92=(a92/a94);
  a92=(a101*a92);
  a102=(a1*a102);
  a102=(a91*a102);
  a102=(a89*a102);
  a105=(a1*a105);
  a91=(a91*a105);
  a91=(a96*a91);
  a102=(a102-a91);
  a102=(a102/a104);
  a102=(a107*a102);
  a92=(a92+a102);
  a92=(a177*a92);
  a92=(a92/a108);
  if (res[1]!=0) res[1][16]=a92;
  a93=(a28*a93);
  a90=(a90*a93);
  a99=(a28*a99);
  a97=(a97*a99);
  a90=(a90+a97);
  a90=(a90/a94);
  a90=(a15-a90);
  a101=(a101*a90);
  a103=(a28*a103);
  a89=(a89*a103);
  a106=(a28*a106);
  a96=(a96*a106);
  a89=(a89+a96);
  a89=(a89/a104);
  a89=(a15-a89);
  a107=(a107*a89);
  a101=(a101+a107);
  a101=(a177*a101);
  a101=(a101/a108);
  if (res[1]!=0) res[1][17]=a101;
  a101=(a146/a150);
  a101=(a154*a101);
  a108=(a109/a157);
  a108=(a136*a108);
  a101=(a101+a108);
  a101=(a177*a101);
  a101=(a101/a160);
  a101=(a37*a101);
  a108=(a126/a130);
  a108=(a137*a108);
  a107=(a125/a140);
  a107=(a143*a107);
  a108=(a108+a107);
  a108=(a177*a108);
  a108=(a108/a144);
  a101=(a101-a108);
  if (res[1]!=0) res[1][18]=a101;
  a101=(a151/a150);
  a101=(a154*a101);
  a108=(a131/a157);
  a108=(a136*a108);
  a101=(a101+a108);
  a101=(a177*a101);
  a101=(a101/a160);
  a101=(a37*a101);
  a108=(a133/a130);
  a108=(a137*a108);
  a107=(a132/a140);
  a107=(a143*a107);
  a108=(a108+a107);
  a108=(a177*a108);
  a108=(a108/a144);
  a101=(a101-a108);
  if (res[1]!=0) res[1][19]=a101;
  a126=(a126+a126);
  a128=sin(a128);
  a101=(a127*a128);
  a101=(a35+a101);
  a101=(a126*a101);
  a133=(a133+a133);
  a134=cos(a134);
  a108=(a127*a134);
  a108=(a4+a108);
  a108=(a133*a108);
  a101=(a101-a108);
  a130=(a130+a130);
  a101=(a101/a130);
  a101=(a137*a101);
  a125=(a125+a125);
  a138=sin(a138);
  a108=(a127*a138);
  a108=(a35+a108);
  a108=(a125*a108);
  a132=(a132+a132);
  a141=cos(a141);
  a107=(a127*a141);
  a107=(a4+a107);
  a107=(a132*a107);
  a108=(a108-a107);
  a140=(a140+a140);
  a108=(a108/a140);
  a108=(a143*a108);
  a101=(a101+a108);
  a101=(a177*a101);
  a101=(a101/a144);
  a146=(a146+a146);
  a148=sin(a148);
  a148=(a147*a148);
  a148=(a146*a148);
  a151=(a151+a151);
  a152=cos(a152);
  a152=(a147*a152);
  a152=(a151*a152);
  a148=(a148-a152);
  a150=(a150+a150);
  a148=(a148/a150);
  a148=(a154*a148);
  a109=(a109+a109);
  a155=sin(a155);
  a155=(a147*a155);
  a155=(a109*a155);
  a131=(a131+a131);
  a158=cos(a158);
  a147=(a147*a158);
  a147=(a131*a147);
  a155=(a155-a147);
  a157=(a157+a157);
  a155=(a155/a157);
  a155=(a136*a155);
  a148=(a148+a155);
  a148=(a177*a148);
  a148=(a148/a160);
  a148=(a37*a148);
  a101=(a101-a148);
  if (res[1]!=0) res[1][20]=a101;
  a101=(a40*a129);
  a101=(a6+a101);
  a101=(a126*a101);
  a148=(a40*a135);
  a148=(a21+a148);
  a148=(a133*a148);
  a101=(a101+a148);
  a101=(a101/a130);
  a101=(a188-a101);
  a101=(a137*a101);
  a148=(a40*a139);
  a148=(a6+a148);
  a148=(a125*a148);
  a155=(a40*a142);
  a155=(a21+a155);
  a155=(a132*a155);
  a148=(a148+a155);
  a148=(a148/a140);
  a148=(a188-a148);
  a148=(a143*a148);
  a101=(a101+a148);
  a101=(a177*a101);
  a101=(a101/a144);
  a149=(a40*a149);
  a146=(a146*a149);
  a153=(a40*a153);
  a151=(a151*a153);
  a146=(a146+a151);
  a146=(a146/a150);
  a146=(a188-a146);
  a154=(a154*a146);
  a156=(a40*a156);
  a109=(a109*a156);
  a159=(a40*a159);
  a131=(a131*a159);
  a109=(a109+a131);
  a109=(a109/a157);
  a109=(a188-a109);
  a136=(a136*a109);
  a154=(a154+a136);
  a154=(a177*a154);
  a154=(a154/a160);
  a154=(a37*a154);
  a101=(a101-a154);
  if (res[1]!=0) res[1][21]=a101;
  a128=(a1*a128);
  a128=(a127*a128);
  a128=(a126*a128);
  a134=(a1*a134);
  a134=(a127*a134);
  a134=(a133*a134);
  a128=(a128-a134);
  a128=(a128/a130);
  a128=(a137*a128);
  a138=(a1*a138);
  a138=(a127*a138);
  a138=(a125*a138);
  a141=(a1*a141);
  a127=(a127*a141);
  a127=(a132*a127);
  a138=(a138-a127);
  a138=(a138/a140);
  a138=(a143*a138);
  a128=(a128+a138);
  a128=(a177*a128);
  a128=(a128/a144);
  if (res[1]!=0) res[1][22]=a128;
  a129=(a28*a129);
  a126=(a126*a129);
  a135=(a28*a135);
  a133=(a133*a135);
  a126=(a126+a133);
  a126=(a126/a130);
  a126=(a15-a126);
  a137=(a137*a126);
  a139=(a28*a139);
  a125=(a125*a139);
  a142=(a28*a142);
  a132=(a132*a142);
  a125=(a125+a132);
  a125=(a125/a140);
  a125=(a15-a125);
  a143=(a143*a125);
  a137=(a137+a143);
  a137=(a177*a137);
  a137=(a137/a144);
  if (res[1]!=0) res[1][23]=a137;
  a137=(a178/a180);
  a137=(a184*a137);
  a144=(a145/a186);
  a144=(a171*a144);
  a137=(a137+a144);
  a137=(a177*a137);
  a137=(a137/a187);
  a137=(a37*a137);
  a144=(a162/a165);
  a144=(a172*a144);
  a143=(a161/a174);
  a143=(a175*a143);
  a144=(a144+a143);
  a144=(a177*a144);
  a144=(a144/a176);
  a137=(a137-a144);
  if (res[1]!=0) res[1][24]=a137;
  a137=(a181/a180);
  a137=(a184*a137);
  a144=(a166/a186);
  a144=(a171*a144);
  a137=(a137+a144);
  a137=(a177*a137);
  a137=(a137/a187);
  a137=(a37*a137);
  a144=(a168/a165);
  a144=(a172*a144);
  a143=(a167/a174);
  a143=(a175*a143);
  a144=(a144+a143);
  a144=(a177*a144);
  a144=(a144/a176);
  a137=(a137-a144);
  if (res[1]!=0) res[1][25]=a137;
  a162=(a162+a162);
  a163=sin(a163);
  a137=(a10*a163);
  a137=(a35+a137);
  a137=(a162*a137);
  a168=(a168+a168);
  a169=cos(a169);
  a144=(a10*a169);
  a144=(a4+a144);
  a144=(a168*a144);
  a137=(a137-a144);
  a165=(a165+a165);
  a137=(a137/a165);
  a137=(a172*a137);
  a161=(a161+a161);
  a7=sin(a7);
  a144=(a10*a7);
  a35=(a35+a144);
  a35=(a161*a35);
  a167=(a167+a167);
  a13=cos(a13);
  a144=(a10*a13);
  a4=(a4+a144);
  a4=(a167*a4);
  a35=(a35-a4);
  a174=(a174+a174);
  a35=(a35/a174);
  a35=(a175*a35);
  a137=(a137+a35);
  a137=(a177*a137);
  a137=(a137/a176);
  a178=(a178+a178);
  a11=sin(a11);
  a11=(a9*a11);
  a11=(a178*a11);
  a181=(a181+a181);
  a182=cos(a182);
  a182=(a9*a182);
  a182=(a181*a182);
  a11=(a11-a182);
  a180=(a180+a180);
  a11=(a11/a180);
  a11=(a184*a11);
  a145=(a145+a145);
  a3=sin(a3);
  a3=(a9*a3);
  a3=(a145*a3);
  a166=(a166+a166);
  a14=cos(a14);
  a9=(a9*a14);
  a9=(a166*a9);
  a3=(a3-a9);
  a186=(a186+a186);
  a3=(a3/a186);
  a3=(a171*a3);
  a11=(a11+a3);
  a11=(a177*a11);
  a11=(a11/a187);
  a11=(a37*a11);
  a137=(a137-a11);
  if (res[1]!=0) res[1][26]=a137;
  a137=(a40*a164);
  a137=(a6+a137);
  a137=(a162*a137);
  a11=(a40*a170);
  a11=(a21+a11);
  a11=(a168*a11);
  a137=(a137+a11);
  a137=(a137/a165);
  a137=(a188-a137);
  a137=(a172*a137);
  a11=(a40*a173);
  a6=(a6+a11);
  a6=(a161*a6);
  a11=(a40*a22);
  a21=(a21+a11);
  a21=(a167*a21);
  a6=(a6+a21);
  a6=(a6/a174);
  a6=(a188-a6);
  a6=(a175*a6);
  a137=(a137+a6);
  a137=(a177*a137);
  a137=(a137/a176);
  a179=(a40*a179);
  a178=(a178*a179);
  a183=(a40*a183);
  a181=(a181*a183);
  a178=(a178+a181);
  a178=(a178/a180);
  a178=(a188-a178);
  a184=(a184*a178);
  a185=(a40*a185);
  a145=(a145*a185);
  a40=(a40*a20);
  a166=(a166*a40);
  a145=(a145+a166);
  a145=(a145/a186);
  a188=(a188-a145);
  a171=(a171*a188);
  a184=(a184+a171);
  a184=(a177*a184);
  a184=(a184/a187);
  a37=(a37*a184);
  a137=(a137-a37);
  if (res[1]!=0) res[1][27]=a137;
  a163=(a1*a163);
  a163=(a10*a163);
  a163=(a162*a163);
  a169=(a1*a169);
  a169=(a10*a169);
  a169=(a168*a169);
  a163=(a163-a169);
  a163=(a163/a165);
  a163=(a172*a163);
  a7=(a1*a7);
  a7=(a10*a7);
  a7=(a161*a7);
  a1=(a1*a13);
  a10=(a10*a1);
  a10=(a167*a10);
  a7=(a7-a10);
  a7=(a7/a174);
  a7=(a175*a7);
  a163=(a163+a7);
  a163=(a177*a163);
  a163=(a163/a176);
  if (res[1]!=0) res[1][28]=a163;
  a164=(a28*a164);
  a162=(a162*a164);
  a170=(a28*a170);
  a168=(a168*a170);
  a162=(a162+a168);
  a162=(a162/a165);
  a162=(a15-a162);
  a172=(a172*a162);
  a173=(a28*a173);
  a161=(a161*a173);
  a28=(a28*a22);
  a167=(a167*a28);
  a161=(a161+a167);
  a161=(a161/a174);
  a15=(a15-a161);
  a175=(a175*a15);
  a172=(a172+a175);
  a177=(a177*a172);
  a177=(a177/a176);
  if (res[1]!=0) res[1][29]=a177;
  return 0;
}

CASADI_SYMBOL_EXPORT int kinematic_constr_h_fun_jac_uxt_zt(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int kinematic_constr_h_fun_jac_uxt_zt_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int kinematic_constr_h_fun_jac_uxt_zt_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void kinematic_constr_h_fun_jac_uxt_zt_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int kinematic_constr_h_fun_jac_uxt_zt_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void kinematic_constr_h_fun_jac_uxt_zt_release(int mem) {
}

CASADI_SYMBOL_EXPORT void kinematic_constr_h_fun_jac_uxt_zt_incref(void) {
}

CASADI_SYMBOL_EXPORT void kinematic_constr_h_fun_jac_uxt_zt_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int kinematic_constr_h_fun_jac_uxt_zt_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int kinematic_constr_h_fun_jac_uxt_zt_n_out(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_real kinematic_constr_h_fun_jac_uxt_zt_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* kinematic_constr_h_fun_jac_uxt_zt_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* kinematic_constr_h_fun_jac_uxt_zt_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* kinematic_constr_h_fun_jac_uxt_zt_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* kinematic_constr_h_fun_jac_uxt_zt_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    case 1: return casadi_s5;
    case 2: return casadi_s6;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int kinematic_constr_h_fun_jac_uxt_zt_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 3;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
