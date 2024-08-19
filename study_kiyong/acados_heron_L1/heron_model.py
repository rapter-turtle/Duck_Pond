#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, sqrt

def export_heron_model() -> AcadosModel:

    model_name = 'heron'

    # constants
    M = 37.758 # Mass [kg]
    I = 18.35 # Inertial tensor [kg m^2]    
    Xu = 8.9149
    Xuu = 11.2101
    Nr = 16.9542
    Nrrr = 12.8966
    Yv = 15
    Yvv = 3
    Yr = 6
    Nv = 6
    dist = 0.3 # 30cm

    # set up states & controls
    xn   = SX.sym('xn')
    yn   = SX.sym('yn')
    psi  = SX.sym('psi')
    un    = SX.sym('un')
    vn    = SX.sym('vn')
    r    = SX.sym('r')

    n1  = SX.sym('n1')
    n2  = SX.sym('n2')

    x = vertcat(xn, yn, psi, un, vn, r, n1, n2)

    n1d  = SX.sym('n1d')
    n2d  = SX.sym('n2d')
    u   = vertcat(n1d, n2d)

    # xdot
    xn_dot  = SX.sym('xn_dot')
    yn_dot  = SX.sym('yn_dot')
    psi_dot = SX.sym('psi_dot')
    u_dot   = SX.sym('u_dot')
    v_dot   = SX.sym('v_dot')
    r_dot   = SX.sym('r_dot')
    n1_dot   = SX.sym('n1_dot')
    n2_dot   = SX.sym('n2_dot')

    xdot = vertcat(xn_dot, yn_dot, psi_dot, u_dot, v_dot, r_dot, n1_dot, n2_dot)

    eps = 0.00001

    oa = SX.sym('oa') 
    ob = SX.sym('ob') 
    oc = SX.sym('oc')
    disturbance_u = SX.sym('disturbance_u')
    disturbance_r = SX.sym('disturbance_r')
     
    p = vertcat(oa, ob, oc, disturbance_u, disturbance_r)


    # dynamics
    f_expl = vertcat(un*cos(psi) + vn*sin(psi),
                     un*sin(psi) - vn*cos(psi),
                     r,
                     ( (n1+n2) - Xu*un - Xuu*sqrt(un*un+eps)*un )/M,
                     ( -Yv*vn - Yvv*sqrt(vn*vn+eps)*vn - Yr*r )/M,
                     ( (-n1+n2)*dist - Nr*r - Nrrr*r*r - Nv*vn)/I,
                     n1d,
                     n2d
                     )

    f_impl = xdot - f_expl


    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name
    model.p = p

    # store meta information
    model.x_labels = ['$x$ [m]', '$y$ [m]',  '$psi$ [rad]',  '$u$ [m/s]', '$r$ [rad/s]', '$n_1$ [N]', '$n_2$ [N]']
    model.u_labels = ['$n_1_d$ [N/s]', '$n_2_d$ [N/s]']
    model.t_label = '$t$ [s]'

    return model

