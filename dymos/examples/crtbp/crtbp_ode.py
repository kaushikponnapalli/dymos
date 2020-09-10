import numpy as np
import openmdao.api as om

mu_dict = {'earth-moon': 1,
           'sun-earth': 1,
           'jupiter-europa': 1}


class crtbp_ode(om.ExplicitComponent):
    """
    ODE for the circular restricted three-body problem (CRTBP)
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('system', default='earth-moon', values=['earth-moon', 'sun-earth', 'jupiter-europa'],
                             desc='primary and secondary masses to be considered')

    def setup(self):
        nn = self.options['num_nodes']
        mu_val = mu_dict[self.options['system']]*np.ones(nn)

        self.add_input('mu', val=mu_val, desc='gravitational parameter for the specified CRTBP system')
        self.add_input('x', val=np.ones(nn), desc='x-position in rotating frame')
        self.add_input('y', val=np.ones(nn), desc='y-position in rotating frame')
        self.add_input('z', val=np.ones(nn), desc='z-position in rotating frame')
        self.add_input('x_dot', val=np.ones(nn), desc='x-velocity in rotating frame')
        self.add_input('y_dot', val=np.ones(nn), desc='y-velocity in rotating frame')
        self.add_input('z_dot', val=np.ones(nn), desc='z-velocity in rotating frame')

        self.add_output('vx_dot', val=np.ones(nn), desc='computed acceleration in the rotating frame')
        self.add_output('vy_dot', val=np.ones(nn), desc='computed acceleration in the rotating frame')
        self.add_output('vz_dot', val=np.ones(nn), desc='computed acceleration in the rotating frame')
