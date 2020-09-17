import openmdao.api as om
import numpy as np
import unittest
import dymos as dm

from dymos.examples.crtbp.crtbp_ode import crtbp_ode


def richardson_approx(amplitude=1, point='L1'):
    return


def make_problem(transcription=dm.GaussLobatto(num_segments=10), compressed=True):

    p = om.Problem(model=om.Group())
    p.driver = om.pyOptSparseDriver()
    phase = dm.Phase(ode_class=crtbp_ode, transcription=transcription)

    p.model.add_subsystem('phase0', phase)



