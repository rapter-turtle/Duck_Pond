/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

#ifndef ACADOS_SOLVER_heron_H_
#define ACADOS_SOLVER_heron_H_

#include "acados/utils/types.h"

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

#define HERON_NX     8
#define HERON_NZ     0
#define HERON_NU     2
#define HERON_NP     6
#define HERON_NBX    2
#define HERON_NBX0   8
#define HERON_NBU    2
#define HERON_NSBX   0
#define HERON_NSBU   0
#define HERON_NSH    2
#define HERON_NSH0   0
#define HERON_NSG    0
#define HERON_NSPHI  0
#define HERON_NSHN   2
#define HERON_NSGN   0
#define HERON_NSPHIN 0
#define HERON_NSPHI0 0
#define HERON_NSBXN  0
#define HERON_NS     2
#define HERON_NS0    0
#define HERON_NSN    2
#define HERON_NG     0
#define HERON_NBXN   0
#define HERON_NGN    0
#define HERON_NY0    10
#define HERON_NY     10
#define HERON_NYN    8
#define HERON_N      40
#define HERON_NH     2
#define HERON_NHN    2
#define HERON_NH0    0
#define HERON_NPHI0  0
#define HERON_NPHI   0
#define HERON_NPHIN  0
#define HERON_NR     0

#ifdef __cplusplus
extern "C" {
#endif


// ** capsule for solver data **
typedef struct heron_solver_capsule
{
    // acados objects
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_out *sens_out;
    ocp_nlp_solver *nlp_solver;
    void *nlp_opts;
    ocp_nlp_plan_t *nlp_solver_plan;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;

    // number of expected runtime parameters
    unsigned int nlp_np;

    /* external functions */
    // dynamics

    external_function_param_casadi *impl_dae_fun;
    external_function_param_casadi *impl_dae_fun_jac_x_xdot_z;
    external_function_param_casadi *impl_dae_jac_x_xdot_u_z;




    // cost

    external_function_param_casadi *cost_y_fun;
    external_function_param_casadi *cost_y_fun_jac_ut_xt;
    external_function_param_casadi *cost_y_hess;



    external_function_param_casadi cost_y_0_fun;
    external_function_param_casadi cost_y_0_fun_jac_ut_xt;
    external_function_param_casadi cost_y_0_hess;



    external_function_param_casadi cost_y_e_fun;
    external_function_param_casadi cost_y_e_fun_jac_ut_xt;
    external_function_param_casadi cost_y_e_hess;


    // constraints
    external_function_param_casadi *nl_constr_h_fun_jac;
    external_function_param_casadi *nl_constr_h_fun;






    external_function_param_casadi nl_constr_h_e_fun_jac;
    external_function_param_casadi nl_constr_h_e_fun;

} heron_solver_capsule;

ACADOS_SYMBOL_EXPORT heron_solver_capsule * heron_acados_create_capsule(void);
ACADOS_SYMBOL_EXPORT int heron_acados_free_capsule(heron_solver_capsule *capsule);

ACADOS_SYMBOL_EXPORT int heron_acados_create(heron_solver_capsule * capsule);

ACADOS_SYMBOL_EXPORT int heron_acados_reset(heron_solver_capsule* capsule, int reset_qp_solver_mem);

/**
 * Generic version of heron_acados_create which allows to use a different number of shooting intervals than
 * the number used for code generation. If new_time_steps=NULL and n_time_steps matches the number used for code
 * generation, the time-steps from code generation is used.
 */
ACADOS_SYMBOL_EXPORT int heron_acados_create_with_discretization(heron_solver_capsule * capsule, int n_time_steps, double* new_time_steps);
/**
 * Update the time step vector. Number N must be identical to the currently set number of shooting nodes in the
 * nlp_solver_plan. Returns 0 if no error occurred and a otherwise a value other than 0.
 */
ACADOS_SYMBOL_EXPORT int heron_acados_update_time_steps(heron_solver_capsule * capsule, int N, double* new_time_steps);
/**
 * This function is used for updating an already initialized solver with a different number of qp_cond_N.
 */
ACADOS_SYMBOL_EXPORT int heron_acados_update_qp_solver_cond_N(heron_solver_capsule * capsule, int qp_solver_cond_N);
ACADOS_SYMBOL_EXPORT int heron_acados_update_params(heron_solver_capsule * capsule, int stage, double *value, int np);
ACADOS_SYMBOL_EXPORT int heron_acados_update_params_sparse(heron_solver_capsule * capsule, int stage, int *idx, double *p, int n_update);

ACADOS_SYMBOL_EXPORT int heron_acados_solve(heron_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int heron_acados_free(heron_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void heron_acados_print_stats(heron_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int heron_acados_custom_update(heron_solver_capsule* capsule, double* data, int data_len);


ACADOS_SYMBOL_EXPORT ocp_nlp_in *heron_acados_get_nlp_in(heron_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *heron_acados_get_nlp_out(heron_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *heron_acados_get_sens_out(heron_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_solver *heron_acados_get_nlp_solver(heron_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_config *heron_acados_get_nlp_config(heron_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void *heron_acados_get_nlp_opts(heron_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_dims *heron_acados_get_nlp_dims(heron_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_plan_t *heron_acados_get_nlp_plan(heron_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_heron_H_
