import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.general_utils import set_pyoptsparse_opt, printoptions
from dymos import Trajectory, GaussLobatto, Phase, Radau
import numpy as np
import os
import unittest
import dymos as dm

from dymos.examples.crtbp.crtbp_ode import crtbp_ode, richardson_approximation


def make_problem(transcription=dm.GaussLobatto(num_segments=10, order=3, compressed=True), optimizer='IPOPT',
                 orbit_options=None):
    if orbit_options is None:
        orbit_options = {'system': 'earth-moon', 'point': 'L1', 'init_state': np.zeros(6), 'period': 1}
    p = om.Problem(model=om.Group())
    p.driver = om.pyOptSparseDriver()
    p.driver.declare_coloring()
    p.driver.options['optimizer'] = optimizer

    if optimizer == 'SNOPT':
        p.driver.opt_settings['Major iterations limit'] = 100
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
    elif optimizer == 'IPOPT':
        p.driver.opt_settings['hessian_approximation'] = 'limited-memory'
        # p.driver.opt_settings['nlp_scaling_method'] = 'user-scaling'
        p.driver.opt_settings['print_level'] = 5
        p.driver.opt_settings['linear_solver'] = 'mumps'

    traj = p.model.add_subsystem('traj', Trajectory())
    phase = traj.add_phase('phase', Phase(ode_class=crtbp_ode, transcription=transcription))

    phase.set_time_options(fix_initial=True, fix_duration=True)
    phase.add_state('x', rate_source='x_dot')
    phase.add_state('y', rate_source='y_dot', fix_initial=True, fix_final=True)
    phase.add_state('z', rate_source='z_dot')
    phase.add_state('x_dot', rate_source='vx_dot', fix_initial=True, fix_final=True)
    phase.add_state('y_dot', rate_source='vy_dot')
    phase.add_state('z_dot', rate_source='vz_dot', fix_initial=True, fix_final=True)

    p.model.add_subsystem('x_periodic_bc', om.ExecComp('bc_defect=final-initial'))
    p.model.connect('traj.phase.timeseries.states:x', 'x_periodic_bc.initial', src_indices=0)
    p.model.connect('traj.phase.timeseries.states:x', 'x_periodic_bc.final', src_indices=-1)

    p.model.add_constraint('x_periodic_bc.bc_defect', equals=0)

    p.model.add_subsystem('z_periodic_bc', om.ExecComp('bc_defect=final-initial'))
    p.model.connect('traj.phase.timeseries.states:z', 'z_periodic_bc.initial', src_indices=0)
    p.model.connect('traj.phase.timeseries.states:z', 'z_periodic_bc.final', src_indices=-1)

    p.model.add_constraint('z_periodic_bc.bc_defect', equals=0)

    p.model.add_subsystem('vy_periodic_bc', om.ExecComp('bc_defect=final-initial'))
    p.model.connect('traj.phase.timeseries.states:y_dot', 'vy_periodic_bc.initial', src_indices=0)
    p.model.connect('traj.phase.timeseries.states:y_dot', 'vy_periodic_bc.final', src_indices=-1)

    p.model.add_constraint('vy_periodic_bc.bc_defect', equals=0)

    phase.add_objective('time', loc='final')

    p.setup(check=True)
    t_guess, state_guess = richardson_approximation(orbit_options)

    p.set_val('traj.phase.t_initial', 0)
    p.set_val('traj.phase.t_duration', orbit_options['period'])
    p.set_val('traj.phase.states:x', phase.interpolate(xs=t_guess, ys=state_guess[:, 0], nodes='state_input'))
    p.set_val('traj.phase.states:y', phase.interpolate(xs=t_guess, ys=state_guess[:, 1], nodes='state_input'))
    p.set_val('traj.phase.states:z', phase.interpolate(xs=t_guess, ys=state_guess[:, 2], nodes='state_input'))
    p.set_val('traj.phase.states:x_dot', phase.interpolate(xs=t_guess, ys=state_guess[:, 3], nodes='state_input'))
    p.set_val('traj.phase.states:y_dot', phase.interpolate(xs=t_guess, ys=state_guess[:, 4], nodes='state_input'))
    p.set_val('traj.phase.states:z_dot', phase.interpolate(xs=t_guess, ys=state_guess[:, 5], nodes='state_input'))

    return p


class TestGeneratePeriodicOrbits(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['total_coloring.pkl', 'SLSQP.out', 'SNOPT_print.out']:
            if os.path.exists(filename):
                os.remove(filename)

    def test_partials(self):
        p = make_problem(transcription=Radau, optimizer='SLSQP')
        p.run_model()
        with printoptions(linewidth=1024, edgeitems=100):
            cpd = p.check_partials(method='fd', compact_print=True, out_stream=None)

    def test_gen_L1_lyap_gl(self):
        initial_state = np.array([0.8089, 0.0, 0.0, 0.0, 0.2838, 0.0])
        T = 3.0224
        p = make_problem(transcription=GaussLobatto(num_segments=10, order=3, compressed=True), optimizer='IPOPT',
                         orbit_options={'system': 'earth-moon', 'point': 'L1', 'init_state': initial_state,
                                        'period': T})
        dm.run_problem(p, refine=True)
