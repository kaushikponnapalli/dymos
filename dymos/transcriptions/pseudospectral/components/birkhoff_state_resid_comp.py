import numpy as np
import openmdao.api as om

from openmdao.utils.general_utils import ensure_compatible

from ...._options import options as dymos_options


class BirkhoffStateResidComp(om.ImplicitComponent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_check_partials = not dymos_options['include_check_partials']
        self._io_pairs = []

    def add_residual_from_input(self, name, **kwargs):
        """
        Adds a residual whose value is given by resid_input.

        Parameters
        ----------
        name : str
            The name of the input providing the residuals for the given output
        kwargs
            Additional keyword arguments for super.add_input and super.add_output.
        """
        val = kwargs['val'] if 'val' in kwargs else 1.0
        shape = kwargs['shape'] if 'shape' in kwargs else None
        val, shape = ensure_compatible(name, value=val, shape=shape)
        resid_name = 'resid_' + name

        self._io_pairs.append((resid_name, name))

        size = np.prod(shape, dtype=int)
        ar = np.arange(size, dtype=int)

        self.add_input(name, **kwargs)
        self.add_residual(resid_name, **kwargs)
        self.declare_partials(of=resid_name, wrt=name, rows=ar, cols=ar, val=1.0)

    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        Compute residuals given inputs and outputs.

        The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        residuals : Vector
            Unscaled, dimensional residuals written to via residuals[key].
        """
        residuals.set_val(inputs.asarray())
        # print(residuals)
        # for name, resid_input in self._io_pairs:
        #     residuals[name] = inputs[resid_input]
