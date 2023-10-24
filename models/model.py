import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

class LinearRegression:
    def __init__(self):
        self.coeff = None
        self.intercept = None

    def fit(self, X, y):
        X_with_intercept = jnp.hstack((jnp.ones((X.shape[0], 1)), X))
        
        solution = jnp.linalg.lstsq(X_with_intercept, y)
        self.coeff = solution[0][1:]
        self.intercept = solution[0][0]

    def __call__(self, X):
        return jnp.dot(X, self.coeff) + self.intercept

 
class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01, weight_penalty=0.01):
        class Model(nn.Module):
            hidden_size: int
            output_size: int

            def setup(self):
                self.dense1 = nn.Dense(self.hidden_size)
                self.dense2 = nn.Dense(self.output_size)

            def __call__(self, x):
                x = self.dense1(x)
                x = jax.nn.relu(x)
                x = self.dense2(x)
                return x

        self.model = Model(hidden_size=hidden_size, output_size=output_size)
        self.params = self.model.init(jax.random.PRNGKey(0), jnp.ones((1, input_size)))['params']
        
        self.optimizer = optax.chain(
            optax.scale_by_adam(),
            optax.scale(-lr)
        )
        self.optimizer_state = self.optimizer.init(self.params)
        self.weight_penalty = weight_penalty

    def fit(self, X, y, epochs=100):
        def loss_fn(params):
            logits = self.model.apply({'params': params}, X)
            mse_loss = jnp.mean((logits - y) ** 2)
            l2_loss = 0.5 * self.weight_penalty * (jnp.sum(jnp.square(params['dense1']['kernel'])) + jnp.sum(jnp.square(params['dense2']['kernel'])))
            return mse_loss + l2_loss

        for epoch in range(epochs):
            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(self.params)
            print(loss)
            updates, self.optimizer_state = self.optimizer.update(grads, self.optimizer_state)
            self.params = optax.apply_updates(self.params, updates)

    def __call__(self, X):
        return self.model.apply({'params': self.params}, X)
    
