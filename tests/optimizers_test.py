# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the optimizers module."""

import functools

from absl.testing import absltest
import numpy as onp

import jax.numpy as np
import jax.test_util as jtu
from jax import jit, grad, jacfwd, jacrev
from jax import core, tree_util
from jax import lax
from jax.experimental import optimizers
from jax.experimental.lbfgs import lbfgs
from jax.interpreters import xla

from jax.config import config
config.parse_flags_with_absl()


class OptimizerTests(jtu.JaxTestCase):

  def _CheckOptimizer(self, optimizer, loss, x0, num_steps, *args, **kwargs):
    self._CheckFuns(optimizer, loss, x0, *args)
    self._CheckRun(optimizer, loss, x0, num_steps, *args, **kwargs)

  def _CheckFuns(self, optimizer, loss, x0, *args):
    init_fun, update_fun, get_params = optimizer(*args)
    opt_state = init_fun(x0)
    self.assertAllClose(x0, get_params(opt_state), check_dtypes=True)
    opt_state2 = update_fun(0, grad(loss)(x0), opt_state)  # doesn't crash
    self.assertEqual(tree_util.tree_structure(opt_state),
                     tree_util.tree_structure(opt_state2))

  @jtu.skip_on_devices('gpu')
  def _CheckRun(self, optimizer, loss, x0, num_steps, *args, **kwargs):
    init_fun, update_fun, get_params = optimizer(*args)

    opt_state = init_fun(x0)
    for i in range(num_steps):
      x = get_params(opt_state)
      g = grad(loss)(x)
      opt_state = update_fun(i, g, opt_state)
    xstar = get_params(opt_state)
    self.assertLess(loss(xstar), 1e-2)

    update_fun_jitted = jit(update_fun)
    opt_state = init_fun(x0)
    for i in range(num_steps):
      x = get_params(opt_state)
      g = grad(loss)(x)
      opt_state = update_fun_jitted(i, g, opt_state)
    xstar = get_params(opt_state)
    self.assertLess(loss(xstar), 1e-2)

  def testSgdScalar(self):
    def loss(x): return x**2
    x0 = 1.
    num_iters = 100
    step_size = 0.1
    self._CheckOptimizer(optimizers.sgd, loss, x0, num_iters, step_size)

  def testSgdVector(self):
    def loss(x): return np.dot(x, x)
    x0 = np.ones(2)
    num_iters = 100
    step_size = 0.1
    self._CheckOptimizer(optimizers.sgd, loss, x0, num_iters, step_size)

  def testSgdNestedTuple(self):
    def loss(xyz):
      x, (y, z) = xyz
      return sum(np.dot(a, a) for a in [x, y, z])
    x0 = (np.ones(2), (np.ones(2), np.ones(2)))
    num_iters = 100
    step_size = 0.1
    self._CheckOptimizer(optimizers.sgd, loss, x0, num_iters, step_size)

  def testMomentumVector(self):
    def loss(x): return np.dot(x, x)
    x0 = np.ones(2)
    num_iters = 100
    step_size = 0.1
    mass = 0.
    self._CheckOptimizer(optimizers.momentum, loss, x0, num_iters, step_size, mass)

  def testMomentumDict(self):
    def loss(dct): return np.dot(dct['x'], dct['x'])
    x0 = {'x': np.ones(2)}
    num_iters = 100
    step_size = 0.1
    mass = 0.
    self._CheckOptimizer(optimizers.momentum, loss, x0, num_iters, step_size, mass)

  def testRmspropVector(self):
    def loss(x): return np.dot(x, x)
    x0 = np.ones(2)
    num_iters = 100
    step_size = 0.1
    self._CheckOptimizer(optimizers.rmsprop, loss, x0, num_iters, step_size)

  def testAdamVector(self):
    def loss(x): return np.dot(x, x)
    x0 = np.ones(2)
    num_iters = 100
    step_size = 0.1
    self._CheckOptimizer(optimizers.adam, loss, x0, num_iters, step_size)

  def testSgdClosure(self):
    def loss(y, x): return y**2 * x**2
    x0 = 1.
    y = 1.
    num_iters = 20
    step_size = 0.1
    partial_loss = functools.partial(loss, y)
    self._CheckRun(optimizers.sgd, partial_loss, x0, num_iters, step_size)

  def testAdagrad(self):

    def loss(xs):
      x1, x2 = xs
      return np.sum(x1**2) + np.sum(x2**2)

    num_iters = 100
    step_size = 0.1
    x0 = (np.ones(2), np.ones((2, 2)))
    self._CheckOptimizer(optimizers.adagrad, loss, x0, num_iters, step_size)

  def testSM3(self):
    def loss(xs):
      x1, x2 = xs
      return np.sum(x1 ** 2) + np.sum(x2 ** 2)

    num_iters = 100
    step_size = 0.1
    x0 = (np.ones(2), np.ones((2, 2)))
    self._CheckOptimizer(optimizers.sm3, loss, x0, num_iters, step_size)

  def testSgdVectorExponentialDecaySchedule(self):
    def loss(x): return np.dot(x, x)
    x0 = np.ones(2)
    step_schedule = optimizers.exponential_decay(0.1, 3, 2.)
    self._CheckFuns(optimizers.sgd, loss, x0, step_schedule)

  def testSgdVectorInverseTimeDecaySchedule(self):
    def loss(x): return np.dot(x, x)
    x0 = np.ones(2)
    step_schedule = optimizers.inverse_time_decay(0.1, 3, 2.)
    self._CheckFuns(optimizers.sgd, loss, x0, step_schedule)

  def testAdamVectorInverseTimeDecaySchedule(self):
    def loss(x): return np.dot(x, x)
    x0 = np.ones(2)
    step_schedule = optimizers.inverse_time_decay(0.1, 3, 2.)
    self._CheckFuns(optimizers.adam, loss, x0, step_schedule)

  def testMomentumVectorInverseTimeDecayStaircaseSchedule(self):
    def loss(x): return np.dot(x, x)
    x0 = np.ones(2)
    step_sched = optimizers.inverse_time_decay(0.1, 3, 2., staircase=True)
    mass = 0.9
    self._CheckFuns(optimizers.momentum, loss, x0, step_sched, mass)

  def testRmspropmomentumVectorPolynomialDecaySchedule(self):
    def loss(x): return np.dot(x, x)
    x0 = np.ones(2)
    step_schedule = optimizers.polynomial_decay(1.0, 50, 0.1)
    self._CheckFuns(optimizers.rmsprop_momentum, loss, x0, step_schedule)

  def testRmspropVectorPiecewiseConstantSchedule(self):
    def loss(x): return np.dot(x, x)
    x0 = np.ones(2)
    step_schedule = optimizers.piecewise_constant([25, 75], [1.0, 0.5, 0.1])
    self._CheckFuns(optimizers.rmsprop, loss, x0, step_schedule)

  def testTracedStepSize(self):
    def loss(x): return np.dot(x, x)
    x0 = np.ones(2)
    step_size = 0.1

    init_fun, _, _ = optimizers.sgd(step_size)
    opt_state = init_fun(x0)

    @jit
    def update(opt_state, step_size):
      _, update_fun, get_params = optimizers.sgd(step_size)
      x = get_params(opt_state)
      g = grad(loss)(x)
      return update_fun(0, g, opt_state)

    update(opt_state, 0.9)  # doesn't crash

  # TODO(mattjj): re-enable
  # def testDeviceTupleState(self):
  #   init_fun, update_fun, _ = optimizers.sgd(0.1)
  #   opt_state = init_fun(np.zeros(3))
  #   self.assertIsInstance(opt_state, optimizers.OptimizerState)
  #   self.assertIsInstance(opt_state.packed_state, core.JaxTuple)
  #   opt_state = jit(update_fun)(0, np.zeros(3), opt_state)
  #   self.assertIsInstance(opt_state, optimizers.OptimizerState)
  #   self.assertIsInstance(opt_state.packed_state, xla.DeviceTuple)

  def testUpdateFunStructureMismatchErrorMessage(self):
    @optimizers.optimizer
    def opt_maker():
      def init_fun(x0):
        return {'x': x0}
      def update_fun(i, g, opt_state):
        x = opt_state['x']
        return {'x': x - 0.1 * g, 'v': g}  # bug!
      def get_params(opt_state):
        return opt_state['x']
      return init_fun, update_fun, get_params

    init_fun, update_fun, get_params = opt_maker()
    opt_state = init_fun(np.zeros(3))
    self.assertRaises(TypeError, lambda: update_fun(opt_state))

  def testUtilityNorm(self):
    x0 = (np.ones(2), (np.ones(3), np.ones(4)))
    norm = optimizers.l2_norm(x0)
    expected = onp.sqrt(onp.sum(onp.ones(2+3+4)**2))
    self.assertAllClose(norm, expected, check_dtypes=False)

  def testUtilityClipGrads(self):
    g = (np.ones(2), (np.ones(3), np.ones(4)))
    norm = optimizers.l2_norm(g)

    ans = optimizers.clip_grads(g, 1.1 * norm)
    expected = g
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = optimizers.l2_norm(optimizers.clip_grads(g, 0.9 * norm))
    expected = 0.9 * norm
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testIssue758(self):
    # code from https://github.com/google/jax/issues/758
    # this is more of a scan + jacfwd/jacrev test, but it lives here to use the
    # optimizers.py code

    def harmonic_bond(conf, params):
      return np.sum(conf * params)

    opt_init, opt_update, get_params = optimizers.sgd(5e-2)

    x0 = onp.array([0.5], dtype=onp.float64)
    params = onp.array([0.3], dtype=onp.float64)

    def minimize_structure(test_params):
      energy_fn = functools.partial(harmonic_bond, params=test_params)
      grad_fn = grad(energy_fn, argnums=(0,))
      opt_state = opt_init(x0)

      def apply_carry(carry, _):
        i, x = carry
        g = grad_fn(get_params(x))[0]
        new_state = opt_update(i, g, x)
        new_carry = (i+1, new_state)
        return new_carry, _

      carry_final, _ = lax.scan(apply_carry, (0, opt_state), np.zeros((75, 0)))
      trip, opt_final = carry_final
      assert trip == 75
      return opt_final

    initial_params = np.float64(0.5)
    minimize_structure(initial_params)

    def loss(test_params):
      opt_final = minimize_structure(test_params)
      return 1.0 - get_params(opt_final)[0]

    loss_opt_init, loss_opt_update, loss_get_params = optimizers.sgd(5e-2)

    J1 = jacrev(loss, argnums=(0,))(initial_params)
    J2 = jacfwd(loss, argnums=(0,))(initial_params)
    self.assertAllClose(J1, J2, check_dtypes=True, rtol=1e-6)

  def testUnpackPackRoundTrip(self):
    opt_init, _, _ = optimizers.momentum(0.1, mass=0.9)
    params = [{'w': onp.random.randn(1, 2), 'bias': onp.random.randn(2)}]
    expected = opt_init(params)
    ans = optimizers.pack_optimizer_state(
        optimizers.unpack_optimizer_state(expected))
    self.assertEqual(ans, expected)

  def testLBFGSpytrees(self):
    def loss(x):
      return np.sum(np.real(x * np.conj(x)))

    def loss_iterable(xs):
      return np.sum([loss(x) for x in xs])

    def loss_dict(x_dict):
      return loss(x_dict['x']) + loss(x_dict['y'])

    def loss_mixed(x_tree):
      scalar, inner_list, xy_dict = x_tree
      return loss(scalar) + loss_iterable(inner_list) + loss_dict(xy_dict)

    # with scalar arg
    x_min, loss_min = lbfgs(loss, 3.)
    self.assertAllClose(x_min, 0., check_dtypes=False)
    self.assertAllClose(loss_min, 0., check_dtypes=False)

    # with vector arg
    x_min, loss_min = lbfgs(loss, np.array([3., 2., 1.]))
    self.assertAllClose(x_min, np.zeros([3]), check_dtypes=False)
    self.assertAllClose(loss_min, 0., check_dtypes=False)

    # with nd arg
    x_min, loss_min = lbfgs(loss, np.array([[3., 2., 1.], [2., 3., 1.]]))
    self.assertAllClose(x_min, np.zeros([2, 3]), check_dtypes=False)
    self.assertAllClose(loss_min, 0., check_dtypes=False)

    # with tuple arg
    x_min, loss_min = lbfgs(loss_iterable, (2., 3., 4.))
    self.assertAllClose(x_min, (0., 0., 0.), check_dtypes=False)
    self.assertAllClose(loss_min, 0., check_dtypes=False)

    # with dict arg
    x_min, loss_min = lbfgs(loss_dict, {'x': 1., 'y': 2.})
    self.assertAllClose(x_min, {'x': 0., 'y': 0.}, check_dtypes=False)
    self.assertAllClose(loss_min, 0., check_dtypes=False)

    # with mixed pytree arg
    x0 = [3., [np.array([1., 2.]), 4., np.array([[1., 2.], [3., 4.]])], {'x': 1., 'y': np.array([1., 2.])}]
    x_expect = [0., [np.zeros([2]), 0., np.zeros([2, 2])], {'x': 0., 'y': np.zeros([2])}]
    x_min, loss_min = lbfgs(loss_mixed, x0)
    self.assertAllClose(x_min, x_expect, check_dtypes=False)
    self.assertAllClose(loss_min, 0., check_dtypes=False)

  def testLBFGSdtypes(self):
    # TODO
    #  real function with complex intermediate values
    #  function of mixed variables
    #  all-complex
    #  float16 vs float32
    pass

  def testLbfgsArgs(self):
    # TODO trivial args
    # TODO non-trivial python type args
    # TODO jax type args
    # TODO use all options as documented
    # TODO use only some options
    # TODO use illegal option
    pass

  def testLbfgsConvergence(self):
    # TODO list of functions from wikipedia
    pass



if __name__ == '__main__':
  absltest.main()
