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


    # set up states & controls
    xn   = SX.sym('xn')
    yn   = SX.sym('yn')
    xn_dot  = SX.sym('xn_dot')
    yn_dot  = SX.sym('yn_dot')

    x = vertcat(xn, yn, xn_dot, yn_dot)

    n1  = SX.sym('n1')
    n2  = SX.sym('n2')

    u   = vertcat(n1, n2)

    # xdot
    xn_dotdot  = SX.sym('xn_dotdot')
    yn_dotdot  = SX.sym('yn_dotdot')

    xdot = vertcat(xn_dot, yn_dot, xn_dotdot, yn_dotdot)

    eps = 0.00001

    oa = SX.sym('oa') 
    ob = SX.sym('ob') 
    oc = SX.sym('oc')
    disturbance_u = SX.sym('disturbance_u')
    disturbance_r = SX.sym('disturbance_r')
     
    p = vertcat(oa, ob, oc, disturbance_u, disturbance_r)


    # dynamics
    # f_expl = vertcat(xn_dot,
    #                  yn_dot,
    #                  - xn - 1.73*xn_dot + n1,
    #                  - yn - 1.73*yn_dot + n2
    #                  )
    f_expl = vertcat(xn_dot,
                     yn_dot,
                     n1,
                     n2
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
    model.x_labels = ['$x$ [m]', '$y$ [m]',  '$x_dot$ [rad]',  '$y_dot$ [m/s]']
    model.u_labels = ['$n_1$ [N/s]', '$n_2$ [N/s]']
    model.t_label = '$t$ [s]'

    return model

