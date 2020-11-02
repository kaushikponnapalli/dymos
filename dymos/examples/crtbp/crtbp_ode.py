import numpy as np
import openmdao.api as om

mu_dict = {'earth-moon': 0.0121505856,
           'sun-earth': 3.04042385e-6,
           'jupiter-europa': 0.002521721}


def richardson_approximation(orbit_options=None):
    # Initial guess generation for periodic orbits about collinear equilibrium points

    if orbit_options is None:
        orbit_options = {'system': 'earth-moon', 'point': 'L1', 'init_state': np.zeros(6),
                         'period': 1}
    num_points = 1000
    data = np.zeros((num_points, 6))
    mu = mu_dict[orbit_options['system']]
    t = np.linspace(0, orbit_options['period'], num=num_points)

    y0 = (mu * (1 - mu) ** (1 / 3)) / 3
    y = y0 + 1

    while abs(y - y0) > 1e-9:
        y0 = y
        if orbit_options['point'] == 'L1':
            y = (mu * (y0 - 1)**2 / (3 - 2 * mu - y0 * (3 - mu - y0))) ** (1 / 3)
        elif orbit_options['point'] == 'L2':
            y = (mu * (y0 + 1)**2 / (3 - 2 * mu + y0 * (3 - mu + y0))) ** (1 / 3)
        elif orbit_options['point'] == 'L3':
            y = ((1 - mu) * (y0 + 1)**2 / (1 + 2 * mu + y0 * (2 + mu + y0))) ** (1 / 3)

    point_location = {'L1': 1 - mu - y, 'L2': 1 - mu + y, 'L3': -mu - y}

    c2 = {'L1': (mu + (1 - mu) * y ** 3 / ((1 - y) ** 3)) / (y ** 3),
          'L2': (mu + (1 - mu) * y ** 3 / ((1 + y) ** 3)) / (y ** 3),
          'L3': (1 - mu + mu*(y/(y + 1)) ** 3) / (y ** 3)}

    L = np.sqrt(0.5 * (2 - c2[orbit_options['point']] + np.sqrt(9 * c2[orbit_options['point']] ** 2 -
                                                                8 * c2[orbit_options['point']])))
    k = 2 * L / (L ** 2 + 1 - c2[orbit_options['point']])

    Ax = orbit_options['init_state'][0]
    Az = orbit_options['init_state'][2]

    data[:, 0] = point_location[orbit_options['point']] - Ax * np.cos(L*t)
    data[:, 1] = k * Ax * np.sin(L*t)
    data[:, 2] = Az * np.sin(L*t)
    data[:, 3] = L * Ax * np.sin(L*t)
    data[:, 4] = k * L * Ax * np.cos(L*t)
    data[:, 5] = L * Az * np.cos(L*t)

    return t, data


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
        mu_val = mu_dict[self.options['system']]

        self.add_input('mu', val=mu_val, desc='gravitational parameter for the specified CRTBP system')
        self.add_input('x', val=np.ones(nn), desc='x-position in rotating frame')
        self.add_input('y', val=np.ones(nn), desc='y-position in rotating frame')
        self.add_input('z', val=np.ones(nn), desc='z-position in rotating frame')
        self.add_input('x_dot', val=np.ones(nn), desc='x-velocity in rotating frame')
        self.add_input('y_dot', val=np.ones(nn), desc='y-velocity in rotating frame')
        self.add_input('z_dot', val=np.ones(nn), desc='z-velocity in rotating frame')

        self.add_output('vx', val=np.ones(nn), desc='computed velocity in the rotating frame', units='1.0/s')
        self.add_output('vy', val=np.ones(nn), desc='computed velocity in the rotating frame', units='1.0/s')
        self.add_output('vz', val=np.ones(nn), desc='computed velocity in the rotating frame', units='1.0/s')
        self.add_output('vx_dot', val=np.ones(nn), desc='computed acceleration in the rotating frame', units='1.0/s')
        self.add_output('vy_dot', val=np.ones(nn), desc='computed acceleration in the rotating frame', units='1.0/s')
        self.add_output('vz_dot', val=np.ones(nn), desc='computed acceleration in the rotating frame', units='1.0/s')

        # Setup partials
        ar = np.arange(nn)
        c = np.zeros(nn)

        self.declare_partials(of='vx', wrt='x_dot', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='vy', wrt='y_dot', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='vz', wrt='z_dot', rows=ar, cols=ar, val=1.0)

        self.declare_partials(of='vx_dot', wrt='x', rows=ar, cols=ar)
        self.declare_partials(of='vx_dot', wrt='y', rows=ar, cols=ar)
        self.declare_partials(of='vx_dot', wrt='z', rows=ar, cols=ar)
        self.declare_partials(of='vx_dot', wrt='y_dot', rows=ar, cols=ar, val=2.0)
        self.declare_partials(of='vx_dot', wrt='mu', rows=ar, cols=c)

        self.declare_partials(of='vy_dot', wrt='x', rows=ar, cols=ar)
        self.declare_partials(of='vy_dot', wrt='y', rows=ar, cols=ar)
        self.declare_partials(of='vy_dot', wrt='z', rows=ar, cols=ar)
        self.declare_partials(of='vy_dot', wrt='x_dot', rows=ar, cols=ar, val=-2.0)
        self.declare_partials(of='vy_dot', wrt='mu', rows=ar, cols=c)

        self.declare_partials(of='vz_dot', wrt='x', rows=ar, cols=ar)
        self.declare_partials(of='vz_dot', wrt='y', rows=ar, cols=ar)
        self.declare_partials(of='vz_dot', wrt='z', rows=ar, cols=ar)
        self.declare_partials(of='vz_dot', wrt='mu', rows=ar, cols=c)

    def compute(self, inputs, outputs):
        mu = inputs['mu']
        x = inputs['x']
        y = inputs['y']
        z = inputs['z']
        x_dot = inputs['x_dot']
        y_dot = inputs['y_dot']
        z_dot = inputs['z_dot']

        r1 = np.sqrt((x+mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x+mu-1)**2 + y**2 + z**2)

        outputs['vx'] = x_dot
        outputs['vy'] = y_dot
        outputs['vz'] = z_dot

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

        partials['vx_dot', 'mu'] = (2 * mu + x - 1)/(r1 ** 3) + (1 - x - 2 * mu)/(r2 ** 3) + 3 * (1 - mu) * (x + mu) * \
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
