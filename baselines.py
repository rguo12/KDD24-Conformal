import jax
import jax.numpy as jnp
from models.model import *
from models.causal_models import *
from sklearn.model_selection import train_test_split


def marginal_conformal(rng_key, model, train_data, cal_data, X_name, Y_name, 
                       N, alpha, intervene_set, intervene_func):
    X_cal, y_cal = cal_data[X_name].values, cal_data[Y_name].values
    model.fit(train_data[X_name].values, train_data[Y_name].values)

    interval_width_list = []
    coverage_list = []
    for intervene_A in intervene_set:
        intervention_data = intervene_func(rng_key, N, intervene_A, 1.0)
        potential_X = jnp.stack((intervention_data[0], intervention_data[1])).T
        potential_Y = intervention_data[-1]

        cal_n = X_cal.shape[0]
        y_cal_pred = model(X_cal)
        y_test_pred = model(potential_X)
        cal_scores = jnp.abs(y_cal - y_cal_pred)
        qhat = jnp.quantile(cal_scores, jnp.ceil((cal_n + 1) * (1 - alpha)) / cal_n, 
                            interpolation='higher')
        test_interval = [y_test_pred.squeeze() - qhat, y_test_pred.squeeze() + qhat]
        interval_width = jnp.mean(test_interval[1] - test_interval[0])   
        coverage = jnp.mean((potential_Y >= test_interval[0]) & (potential_Y <= test_interval[1]))
        interval_width_list.append(interval_width)
        coverage_list.append(coverage)
    return interval_width_list, coverage_list


def conditional_conformal(rng_key, model, train_data, cal_data, X_name, Y_name, 
                          N, alpha, intervene_set, intervene_func):
    """
    This is conditional conformal inference, Conditional CP (joint) in the "malice against one" paper.
    Extension to other conditional conformal inference methods in the "malice against one" paper is straightforward.
    """
    model.fit(train_data[X_name].values, train_data[Y_name].values)
    interval_width_list = []
    coverage_list = []

    for intervene_A in intervene_set:
        cal_data_conditioned_on_A =  cal_data[cal_data['R'] == intervene_A]
        X_cal, y_cal = cal_data_conditioned_on_A[X_name].values, cal_data_conditioned_on_A[Y_name].values

        intervention_data = intervene_func(rng_key, N, intervene_A, 1.0)
        potential_X = jnp.stack((intervention_data[0], intervention_data[1])).T
        potential_Y = intervention_data[-1]

        cal_n = X_cal.shape[0]
        y_cal_pred = model(X_cal)
        y_test_pred = model(potential_X)
        cal_scores = jnp.abs(y_cal - y_cal_pred)
        qhat = jnp.quantile(cal_scores, jnp.ceil((cal_n + 1) * (1 - alpha)) / cal_n, 
                            interpolation='higher')
        test_interval = [y_test_pred.squeeze() - qhat, y_test_pred.squeeze() + qhat]
        interval_width = jnp.mean(test_interval[1] - test_interval[0])   
        coverage = jnp.mean((potential_Y >= test_interval[0]) & (potential_Y <= test_interval[1]))
        interval_width_list.append(interval_width)
        coverage_list.append(coverage)
    return interval_width_list, coverage_list
 
def interventional_conformal(rng_key, model, train_data, cal_data, X_name, Y_name, 
                             N, alpha, intervene_set, intervene_func):
    """
    """
    interval_width_list = []
    coverage_list = []

    for intervene_A in intervene_set:
        intervention_data = intervene_func(rng_key, N, intervene_A, 1.0)
        potential_X_train, potential_X_cal = train_test_split(jnp.stack((intervention_data[0], intervention_data[1])).T, random_state = 1234, test_size = 0.2)
        potential_Y_train, potential_Y_cal = train_test_split(intervention_data[-1][:, None], random_state = 1234, test_size = 0.2)

        rng_key, _ = jax.random.split(rng_key)
        intervention_data_test = intervene_func(rng_key, N, intervene_A, 1.0)
        potential_X_test = jnp.stack((intervention_data_test[0], intervention_data_test[1])).T
        potential_Y_test = intervention_data_test[-1]

        model.fit(potential_X_train, potential_Y_train)
        cal_n = potential_X_cal.shape[0]
        y_cal_pred = model(potential_X_cal)
        y_test_pred = model(potential_X_test)
        cal_scores = jnp.abs(potential_Y_cal - y_cal_pred)
        qhat = jnp.quantile(cal_scores, jnp.ceil((cal_n + 1) * (1 - alpha)) / cal_n, 
                            interpolation='higher')
        test_interval = [y_test_pred.squeeze() - qhat, y_test_pred.squeeze() + qhat]
        interval_width = jnp.mean(test_interval[1] - test_interval[0])   
        coverage = jnp.mean((potential_Y_test >= test_interval[0]) & (potential_Y_test <= test_interval[1]))
        interval_width_list.append(interval_width)
        coverage_list.append(coverage)
    return interval_width_list, coverage_list
