import jax.numpy as np
from jax.tree_util import tree_map, tree_multimap, tree_flatten
from jax.experimental.optimizers import make_schedule
from jax.api import value_and_grad


_lbfgs_defaults = {'step_size': 1.,
                   'max_iter': 200,
                   'max_eval': 'default',
                   'tolerance_grad': 1e-5,
                   'tolerance_change': 1e-9,
                   'line_search': 'strong_wolfe',
                   'history_size': 100,
                   'wolfe_c1': 1e-4,
                   'wolfe_c2': 0.9,
                   'max_iter_ls': 20}


def lbfgs(fun, x0, args=(), return_value=True, options=None):
    """
    Minimize a scalar function of one or more (real or complex) variables using the L-BFGS algorithm with optional
    Strong Wolfe line search.

    The L-BFGS algorithm approximates the inverse Hessian and gives a descend direction. Optionally, the step
    size in this direction is determine to satisfy the Strong Wolfe criteria.
    (If disabled, the step size is predetermined)

    Note that the dtype structure of `x_min` is determined by `x0` (it will be exactly the same).
    In particular, this means that if `fun` is to be understood as a function of *complex* variables, the corresponding
    values in `x0` need to have a complex dtype

    Parameters
    ----------
    fun : callable
        The function to be minimised

            ``fun(x, *args) -> float``

        where `x` is an arbitrary pytree of ndarrays or python scalars and `args` is a tuple of additional arguments
        that are considered fixed during the optimisation
    x0 : pytree
        Initial guess. Must be a valid first argument for `fun`.
    args : tuple, optional
        Extra arguments passed to `fun`
    return_value : bool, optional
        If True, `lbfgs` returns both the minimising argument `x_min` as well as the minimal value `fun(x_min, *args)`
    options : dict, optional
        A dictionary of options:

            step_size : float or jax.experimental.optimizers schedule
                If not using line search: the step size at each iteration
                If using line search: the initial guess for the step size
            max_iter : int
                Maximum number of iterations
            max_eval : int or 'default'
                Maximum number of times that `fun` is called via `value_and_grad`
                `default` results in `1.25 * max_iter`
            tolerance_grad : float
                The tolerance value, below which gradients are considered negligible
            tolerance_change : float
                The tolerance value, below which changes in parameter- or value-space are considered negligible
            history_size : int
                The number of gradients and values from previous iterations that are stored to approximate the inverse
                Hessian. Note that if this is larger than `max_iter`, the L-BFGS becomes BFGS.
            line_search : None or 'strong_wolfe'
                Which line-search algorithm to use.
                If None, the step size is determined solely by `step_size`
                If `strong_wolfe`, uses a line search with the rest of the options and taking `step_size`
                as initial step sizes
            wolfe_c1 : float
                The parameter c1 in the Armijo condition
            wolfe_c2 : float
                The parameter c2 in the (2nd) Wolfe condition
            max_iter_ls : int
                The maximum number of line search iterations per L-BFGS iteration

    Returns
    -------
    x_min : pytree
        The (first) argument of `fun` that minimizes it.
        The dtype will be inherited from `x0`
    fun_min : float, optional
        The minimal value `fun(x_min, *args)`

    """
    # merge options with defaults
    # TODO (Jakob-Unfried): Issue a warning when `options` contains an invalid key?
    options = {**_lbfgs_defaults, **options} if options is not None else _lbfgs_defaults
    if options['max_eval'] == 'default':
        options['max_eval'] = round(1.25 * options['max_iter'])
    step_size = make_schedule(options['step_size'])
    if options['line_search'] not in ['strong_wolfe', None]:
        raise ValueError(f'Expected line_search "strong_wolfe" or None. got {options["line_search"]}')

    # function that evaluates cost and the complex gradient ∂* and keeps track of the number of evaluations
    def evaluate_cost(_x, _num_evals):
        _cost, _grad = value_and_grad(fun)(_x, *args)
        _grad = tree_map(np.conj, _grad)  # complex gradient ∂*
        return _cost, _grad, _num_evals + 1

    # initial evaluation
    initial_cost, grad, current_evals = evaluate_cost(x0, 0)

    # already optimal?
    opt_cond = _max_abs(grad) <= options['tolerance_grad']
    if opt_cond:
        return x0, initial_cost if return_value else x0

    # initialise variables for main loop
    x = x0
    cost = initial_cost
    y_history = []
    s_history = []
    rho_history = []
    gamma = 1.
    last_grad = last_x = None

    # main loop
    for n_iter in range(options['max_iter']):
        # --------------------------------
        #    compute descend direction
        # --------------------------------
        if n_iter == 0:  # History is empty. No Hessian approximation yet. Just step in (negative) gradient direction
            direction = tree_map(lambda arr: -arr, grad)
        else:
            # Update history
            y = _vec_add_prefactor(grad, -1, last_grad)  # y_k = p_k - p_{k-1}
            s = _vec_add_prefactor(x, -1, last_x)  # s_k = x_k - x_{k-1}
            rho_inv = np.real(_vec_scalar_prod(y, s))  # rho = 1 / Real(y.conj().dot(s))
            if rho_inv > 1e-10:  # else this update is skipped. TODO shouldn't hardcode this. depend on precision/dtype
                if len(y_history) >= options['history_size']:
                    y_history.pop(0)
                    s_history.pop(0)
                    rho_history.pop(0)

                y_history.append(y)
                s_history.append(s)
                rho_history.append(1. / rho_inv)

                gamma = rho_inv / np.real(_vec_scalar_prod(y, y))  # scale of diagonal Hessian approximation

            # Compute approximate product of inverse Hessian with gradient ('two-loop recursion')
            current_history_size = len(y_history)
            alpha = [np.nan] * current_history_size
            direction = tree_map(lambda arr: -arr, grad)

            for i in range(current_history_size - 1, -1, -1):  # newest to oldest
                alpha[i] = rho_history[i] * np.real(_vec_scalar_prod(s_history[i], direction))
                direction = _vec_add_prefactor(direction, -alpha[i], y_history[i])

            direction = tree_map(lambda arr: gamma * arr, direction)  # scale with gamma

            for i in range(current_history_size):  # oldest to newest
                beta_i = rho_history[i] * np.real(_vec_scalar_prod(y_history[i], direction))
                direction = _vec_add_prefactor(direction, alpha[i] - beta_i, s_history[i])

        # save for next iteration
        last_grad = grad
        last_cost = cost
        last_x = x

        # --------------------------------
        #       compute step length
        # --------------------------------
        # reset initial guess
        if n_iter == 1:
            t = min(1., 1./_sum_abs(grad)) * step_size(n_iter)
        else:
            t = step_size(n_iter)

        # compute directional derivative
        dir_deriv = np.real(_vec_scalar_prod(grad, direction))

        # check for significant change
        if dir_deriv > - options['tolerance_change']:
            break

        # optional line search
        if options['line_search'] == 'strong_wolfe':
            def evaluate_cost_line(_t, _num_evals):
                _x = _vec_add_prefactor(x, _t, direction)
                _cost, _grad, _num_evals = evaluate_cost(_x, _num_evals)
                return _cost, _grad, _num_evals

            cost, grad, t, _evals = _strong_wolfe(evaluate_cost_line, t, direction, cost, grad, dir_deriv,
                                                  options['wolfe_c1'], options['wolfe_c2'], options['tolerance_change'],
                                                  options['max_iter_ls'])
            current_evals += _evals
            x = _vec_add_prefactor(x, t, direction)
        else:  # step with scheduled length
            x = _vec_add_prefactor(x, t, direction)
            cost, grad, current_evals = evaluate_cost(x, current_evals)

        opt_cond = _max_abs(grad) <= options['tolerance_grad']

        # --------------------------------
        #       check conditions
        # --------------------------------
        # optimality reached?
        if opt_cond:
            break

        # alotted resources exceeded?
        if n_iter >= options['max_iter']:
            break
        if current_evals >= options['max_eval']:
            break

        # lack of progress
        if abs(t) * _max_abs(direction) <= options['tolerance_change']:
            break
        if abs(cost - last_cost) <= options['tolerance_change']:
            break

    return x, cost if return_value else x


def _max_abs(tree):
    maxs = tree_map(lambda arr: np.max(np.abs(arr)), tree)
    return np.max(tree_flatten(maxs)[0])


def _vec_scalar_prod(tree1, tree2):
    prods = tree_multimap(lambda arr1, arr2: np.sum(np.conj(arr1) * arr2), tree1, tree2)
    return np.sum(tree_flatten(prods)[0])


def _vec_add_prefactor(tree1, prefactor, tree2):
    return tree_multimap(lambda arr1, arr2: arr1 + prefactor * arr2, tree1, tree2)


def _sum_abs(tree):
    sums = tree_map(lambda arr: np.sum(np.abs(arr)), tree)
    return np.sum(tree_flatten(sums)[0])


def _cubic_interpolate(x1: float, f1: float, g1: float, x2: float, f2: float, g2: float, bounds=None) -> float:
    """
    returns the minimiser of the cubic polynomial that matches a function f(x) in two points in value and derivative

    Parameters
    ----------
    x1
        the first point
    f1
        f(x1)
    g1
        f'(x1)
    x2
        the second point
    f2
        f(x2)
    g2
        f'(x2)
    bounds
        bounds that restrict the output, defaults to ( min(x1,x2), max(x1,x2) )

    Returns
    -------

    """
    if bounds is not None:
        x_min, x_max = bounds
    else:
        x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)

    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_squared = d1 ** 2 - g1 * g2
    if d2_squared >= 0:
        d2 = np.sqrt(d2_squared)
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, x_min), x_max)
    else:
        return (x_min + x_max) / 2


def _strong_wolfe(evaluate_cost_line, t, direction, cost_init, grad_init, dir_deriv_init, c1, c2, tol_change, max_iter):
    """
    Performs a line search to find a step length that fulfils the Wolfe conditions

    Parameters
    ----------
    evaluate_cost_line
        function that evaluates the cost function on a line
        Parameters: t (step length), n_evals
        Returns: cost (value of cost function), grad (cost gradient), n_evals
    t
        the initial guess for the step length
    direction
        the search direction from L-BFGS
    cost_init
        the value of the cost function at the initial point
    grad_init
        the gradient of the cost function at the initial point
    dir_deriv_init
        the directional derivative of the cost function at the initial point
    c1
        The Armijo parameter. Armijo condition is f(z + t*d) <= f(z) + c1 * t * dir_deriv(z)
    c2
        The Wolfe parameter. (2nd) Wolfe condition is dir_deriv(z + t*d) >= c2 * dir_deriv(z)
    tol_change
        value changes below this tolerance are considered to be zero
    max_iter
        maximum number of line-search iterations

    Returns
    -------
    cost_new
        The cost function at the new point
    grad_new
        The gradient of the cost function at the new point
    t
        The step length required to reach the new point, that satisfies the wolfe conditions
    num_evals
        The number of times the cost function was evaluated
    """

    d_norm = _max_abs(direction)

    # evaluate at initially proposed step
    cost_new, grad_new, num_evals = evaluate_cost_line(t, 0)
    dir_deriv_new = np.real(_vec_scalar_prod(grad_new, direction))

    # ------------------------
    #      bracket phase
    # ------------------------
    # bracket an interval containing a point that satisfies the Wolfe criteria
    t_last = 0
    cost_last = cost_init
    grad_last = grad_init
    dir_deriv_last = dir_deriv_init
    bracket, bracket_cost, bracket_grad, bracket_dir_deriv = [], [], [], []
    done = False
    n_iter = 1
    while n_iter <= max_iter:
        # check conditions
        if cost_new > (cost_init + c1 * t * dir_deriv_init) or (n_iter > 1 and cost_new >= cost_last):
            bracket = [t_last, t]
            bracket_cost = [cost_last, cost_new]
            bracket_grad = [grad_last, grad_new]
            bracket_dir_deriv = [dir_deriv_last, dir_deriv_new]
            break

        if abs(dir_deriv_new) <= -c2 * dir_deriv_init:
            bracket = [t]
            bracket_cost = [cost_new]
            bracket_grad = [grad_new]
            done = True
            break

        if dir_deriv_new >= 0:
            bracket = [t_last, t]
            bracket_cost = [cost_last, cost_new]
            bracket_grad = [grad_last, grad_new]
            bracket_dir_deriv = [dir_deriv_last, dir_deriv_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_last)
        max_step = 10 * t
        tmp = t
        t = _cubic_interpolate(t_last, cost_last, dir_deriv_last, t, cost_new, dir_deriv_new,
                               bounds=(min_step, max_step))

        # for next iteration
        t_last = tmp
        cost_last = cost_new
        grad_last = grad_new
        dir_deriv_last = dir_deriv_new
        cost_new, grad_new, num_evals = evaluate_cost_line(t, num_evals)
        dir_deriv_new = np.real(_vec_scalar_prod(grad_new, direction))
        n_iter += 1

    # reached max number of iterations?
    if n_iter >= max_iter:
        bracket = [0, t]
        bracket_cost = [cost_init, cost_new]
        bracket_grad = [grad_init, grad_new]

    # ------------------------
    #       zoom phase
    # ------------------------
    # We either have a point satisfying the criteria or a bracket around it
    # Refine bracket until a point that satisfies criteria is found
    insufficient_progress = False
    low_pos, high_pos = (0, 1) if bracket_cost[0] <= bracket_cost[-1] else (1, 0)
    while not done and n_iter <= max_iter:
        # compute new trial value
        t = _cubic_interpolate(bracket[0], bracket_cost[0], bracket_dir_deriv[0],
                               bracket[1], bracket_cost[1], bracket_dir_deriv[1])

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   - we have made insufficient progress in the last step, or
        #   - `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insufficient_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 10% away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insufficient_progress = False
            else:
                insufficient_progress = True
        else:
            insufficient_progress = False

        # evaluate at new point
        cost_new, grad_new, num_evals = evaluate_cost_line(t, num_evals)
        dir_deriv_new = np.real(_vec_scalar_prod(grad_new, direction))
        n_iter += 1

        if cost_new > (cost_init + c1 * t * dir_deriv_init) or cost_new >= bracket_cost[low_pos]:
            # Armijo condition not satisfied or not lower than low_pos
            bracket[high_pos] = t
            bracket_cost[high_pos] = cost_new
            bracket_grad[high_pos] = grad_new
            bracket_dir_deriv[high_pos] = dir_deriv_new
            low_pos, high_pos = (0, 1) if bracket_cost[0] <= bracket_cost[1] else (1, 0)
        else:
            # Armijo condition satisfied and lower than low_pos
            if abs(dir_deriv_new) <= -c2 * dir_deriv_init:
                # Wolfe condition satisfied
                done = True
            elif dir_deriv_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                bracket[high_pos] = bracket[low_pos]
                bracket_cost[high_pos] = bracket_cost[low_pos]
                bracket_grad[high_pos] = bracket_grad[low_pos]
                bracket_dir_deriv[high_pos] = bracket_dir_deriv[low_pos]

            bracket[low_pos] = t
            bracket_cost[low_pos] = cost_new
            bracket_grad[low_pos] = grad_new
            bracket_dir_deriv[low_pos] = dir_deriv_new

        # line-search bracket is small
        if abs(bracket[1] - bracket[0]) * d_norm < tol_change:
            break

    # return stuff
    t = bracket[low_pos]
    cost_new = bracket_cost[low_pos]
    grad_new = bracket_grad[low_pos]
    return cost_new, grad_new, t, num_evals
