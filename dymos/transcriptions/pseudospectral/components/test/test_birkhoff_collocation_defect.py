import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import openmdao.api as om
from dymos.utils.testing_utils import assert_check_partials

import dymos as dm
from dymos.transcriptions.pseudospectral.components.birkhoff_collocation_comp import BirkhoffCollocationComp
from dymos.transcriptions.grid_data import GridData

# Modify class so we can run it standalone.
from dymos.utils.misc import CompWrapperConfig
from dymos.utils.lgl import lgl
from dymos.utils.lgr import lgr
CollocationComp = CompWrapperConfig(BirkhoffCollocationComp)


class TestCollocationComp(unittest.TestCase):

    def make_problem(self, transcription='radau-ps'):
        dm.options['include_check_partials'] = True

        gd = GridData(
            num_segments=1, segment_ends=np.array([0., 10.]),
            transcription=transcription, transcription_order=20)
        n = gd.subset_num_nodes['col']
        tau = gd.node_stau
        t = 5 * tau + 5

        self.p = om.Problem(model=om.Group())

        # state_options = {'x': {'units': 'm', 'shape': (1,), 'fix_initial': True,
        #                        'fix_final': False, 'solve_segments': False,
        #                        'input_initial': False},
        #                  'v': {'units': 'm/s', 'shape': (3, 2), 'fix_initial': False,
        #                        'fix_final': True, 'solve_segments': False,
        #                        'input_initial': False}}

        state_options = {'x': {'units': 'm', 'shape': (1,), 'fix_initial': True,
                               'fix_final': False, 'solve_segments': False,
                               'input_initial': False}}

        indep_comp = om.IndepVarComp()
        self.p.model.add_subsystem('indep', indep_comp, promotes_outputs=['*'])

        # Testing the basic ODE xdot = -x, x(0) = 10
        # Solution is x(t) = 10*exp(-t)

        x_val = 10*np.exp(-t)

        indep_comp.add_output(
            'dt_dstau',
            val=np.ones(n)*5, units='s')
        indep_comp.add_output(
            'state_value:x',
            val=x_val, units='m')
        indep_comp.add_output(
            'f_value:x',
            val=-x_val, units='m/s')
        indep_comp.add_output(
            'f_computed:x',
            val=-x_val, units='m/s')

        # indep_comp.add_output(
        #     'f_value:v',
        #     val=np.zeros((n-1, 3, 2)), units='m/s')
        # indep_comp.add_output(
        #     'f_computed:v',
        #     val=np.zeros((n, 3, 2)), units='m/s')

        self.p.model.add_subsystem('defect_comp',
                                   subsys=CollocationComp(grid_data=gd,
                                                          state_options=state_options,
                                                          time_units='s'))

        if transcription == 'radau-ps':
            src_indices = om.slicer[:-1]
        else:
            src_indices = om.slicer[:]

        self.p.model.connect('state_value:x', 'defect_comp.states:x', src_indices=src_indices)
        self.p.model.connect('f_value:x', 'defect_comp.state_rates:x', src_indices=src_indices)
        # self.p.model.connect('f_value:v', 'defect_comp.f_value:v')
        self.p.model.connect('f_computed:x', 'defect_comp.f_computed:x', src_indices=src_indices)
        # self.p.model.connect('f_computed:v', 'defect_comp.f_computed:v')
        self.p.model.connect('dt_dstau', 'defect_comp.dt_dstau')

        self.p.setup(force_alloc_complex=True)

        self.p.set_val('defect_comp.initial_states:x', 10.0)
        self.p.set_val('defect_comp.final_states:x', x_val[-1])

        # self.p['f_value:v'] = np.random.random((n-1, 3, 2))

        # self.p['f_computed:v'] = np.random.random((n, 3, 2))

        self.p.run_model()

    def tearDown(self):
        dm.options['include_check_partials'] = False

    def test_results_radau_grid(self):
        self.make_problem(transcription='radau-ps')
        assert_almost_equal(self.p['defect_comp.state_defects:x'], 0.0)
        assert_almost_equal(self.p['defect_comp.state_rate_defects:x'], 0.0)
        assert_almost_equal(self.p['defect_comp.final_state_defects:x'], 0.0)

    def test_results_gl_grid(self):
        self.make_problem(transcription='birkhoff-gauss-lobatto')
        assert_almost_equal(self.p['defect_comp.state_defects:x'], 0.0)
        assert_almost_equal(self.p['defect_comp.state_rate_defects:x'], 0.0)
        assert_almost_equal(self.p['defect_comp.final_state_defects:x'], 0.0)

    def test_partials(self):
        self.make_problem(transcription='birkhoff-gauss-lobatto')
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=False, method='fd')
        assert_check_partials(cpd)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
