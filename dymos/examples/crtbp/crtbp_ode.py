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

        # Setup partials
        ar = np.arange(self.options['num_nodes'])

        self.declare_partials(of='vx_dot', wrt='x', rows=ar, cols=ar)
        self.declare_partials(of='vx_dot', wrt='y', rows=ar, cols=ar)
        self.declare_partials(of='vx_dot', wrt='z', rows=ar, cols=ar)
        self.declare_partials(of='vx_dot', wrt='y_dot', rows=ar, cols=ar, val=2.0)
        self.declare_partials(of='vx_dot', wrt='mu', rows=ar, cols=ar)

        self.declare_partials(of='vy_dot', wrt='x', rows=ar, cols=ar)
        self.declare_partials(of='vy_dot', wrt='y', rows=ar, cols=ar)
        self.declare_partials(of='vy_dot', wrt='z', rows=ar, cols=ar)
        self.declare_partials(of='vy_dot', wrt='x_dot', rows=ar, cols=ar, val=-2.0)
        self.declare_partials(of='vy_dot', wrt='mu', rows=ar, cols=ar)

        self.declare_partials(of='vz_dot', wrt='x', rows=ar, cols=ar)
        self.declare_partials(of='vz_dot', wrt='y', rows=ar, cols=ar)
        self.declare_partials(of='vz_dot', wrt='z', rows=ar, cols=ar)
        self.declare_partials(of='vz_dot', wrt='mu', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        mu = inputs['mu']
        x = inputs['x']
        y = inputs['y']
        z = inputs['z']
        x_dot = inputs['x_dot']
        y_dot = inputs['y_dot']

        r1 = np.sqrt((x+mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x+mu-1)**2 + y**2 + z**2)

        outputs['vx_dot'] = x + 2*y_dot - (1 - mu)*(x + mu)/(r1 ** 3) - mu*(x + mu - 1)/(r2 ** 3)
        outputs['vy_dot'] = y - 2*x_dot - (1 - mu)*y/(r1 ** 3) - mu*y/(r2 ** 3)
        outputs['vz_dot'] = - (1 - mu)*z/(r1 ** 3) - mu*z/(r2 ** 3)

    def compute_partials(self, inputs, partials):
        mu = inputs['mu']
        x = inputs['x']
        y = inputs['y']
        z = inputs['z']

        r1 = np.sqrt((x+mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x+mu-1)**2 + y**2 + z**2)

        partials['vx_dot', 'mu'] = (1 + x - 2 * mu)/(r1 ** 3) + (1 - x + 2 * mu)/(r2 ** 3) + 3 * (1 - mu) * (x-mu) * \
                                   (x + mu) / (r1 ** 5) + 3 * mu * (mu + x - 1)**2/(r2 ** 5)
        partials['vx_dot', 'x'] = 1 - (1 - mu) * (1 / (r1 ** 3) - 3 * (x + mu) ** 2 / (r1 ** 5)) - mu * (1 / (r2 ** 3) - 3 * (x + mu - 1) ** 2 / (r2 ** 5))
        partials['vx_dot', 'y'] = 3 * (1 - mu) * y * (x + mu) / (r1 ** 5) + 3 * mu * y * (x + mu - 1)/(r2 ** 5)
        partials['vx_dot', 'z'] = 3 * (1 - mu) * z * (x + mu) / (r1 ** 5) + 3 * mu * z * (x + mu - 1)/(r2 ** 5)

        partials['vy_dot', 'mu'] = y / (r1 ** 3) + 3 * y * (1 - mu) * (x + mu) / (r1 ** 5) - y / (r2 ** 3) + 3 * mu * y * (x + mu - 1) / (r2 ** 5)
        partials['vy_dot', 'x'] = 3 * (1 - mu) * y * (x + mu) / (r1 ** 5) + 3 * mu * y * (x + mu - 1)/(r2 ** 5)
        partials['vy_dot', 'y'] = 1 - (1 - mu)*(1/(r1 ** 3) - 3 * y ** 2/(r1 ** 5)) - mu * (1/(r2 ** 3) - 3 * y ** 2 /
                                                                                            (r2 ** 5))
        partials['vy_dot', 'z'] = 3 * (1 - mu) * y * z / (r1 ** 5) + 3 * mu * y * z / (r2 ** 5)

        partials['vz_dot', 'mu'] = - z / (r2 ** 3) + 3 * mu * z * (x + mu - 1) / (r2 ** 5)
        partials['vz_dot', 'x'] = 3 * (1 - mu) * z * (x + mu) / (r1 ** 5) + 3 * mu * z * (x + mu - 1)/(r2 ** 5)
        partials['vz_dot', 'y'] = 3 * (1 - mu) * y * z / (r1 ** 5) + 3 * mu * y * z / (r2 ** 5)
        partials['vz_dot', 'z'] = -(1 - mu)*(1 / (r1 ** 3) - 3 * z ** 2 / (r1 ** 5)) - mu * (
                1 / (r2 ** 3) - 3 * z ** 2 / (r2 ** 5))
