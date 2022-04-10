import sys, os
import numpy as np
import tensorflow as tf
import sonnet as snt

class TSNE(snt.AbstractModule):
    def __init__(self, num_samples, dims=2, name="t_sne"):
        super(TSNE, self).__init__(name=name)
        if not isinstance(dims, int):
            raise ValueError("Error: number of dimensions should be an integer.")
        self._num_samples = num_samples
        self._dim = dims

    def _build(self, x, x_prob, perplexity=30.0):
        max_iter = 1000
        initial_momentum = 0.5
        final_momentum = 0.8
        eta = 500
        min_gain = 0.01
        assert callable(x_prob), "The provided x_prob parameter must be callable"

        with tf.control_dependencies([tf.assert_equal(self._num_samples, tf.shape(x)[0])]):
            iY = tf.zeros((self._num_samples, self._dim))
            gains = tf.ones((self._num_samples, self._dim))

        # Compute P-values
        def _x2p(x, tol=1e-5, perplexity=30.0):
            def _calc_prob_and_perp(squared_dist, idx, beta = 1.0):
                unnorm_prob = tf.math.exp(tf.math.negative(squared_dist) * beta)
                unnorm_prob = unnorm_prob - tf.scatter_nd([[idx]], tf.gather(unnorm_prob, [idx]), tf.shape(unnorm_prob))
                sum_unnorm_prob = tf.math.reduce_sum(unnorm_prob)
                perp = tf.math.log(sum_unnorm_prob) + beta * tf.math.reduce_sum(squared_dist * unnorm_prob) / sum_unnorm_prob
                prob = unnorm_prob / sum_unnorm_prob

                return prob, perp
                
            D = x_prob(x)
            D = D - tf.linalg.diag(tf.linalg.diag_part(D))
            dist = tf.math.square(D)
            n = self._num_samples
            indices = tf.cast(tf.expand_dims(tf.linspace(0.0, n-1, n), axis=-1), tf.float32)
            beta = tf.ones((n, 1), dtype = tf.float32)
            threshold_perp = tf.math.log(perplexity)
            P = tf.concat([dist, beta, indices], axis = -1) 

            def _map_fn(xi):
                tries = 0
                num_iters = 50
                betamin = tf.constant(-np.inf)
                betamax = tf.constant(np.inf)

                di = tf.slice(xi, [0], [n])
                bi = tf.squeeze(tf.slice(xi, [n], [1]))
                idx = tf.cast(tf.squeeze(tf.slice(xi, [n+1], [1])), tf.int32)
                prob, perp = _calc_prob_and_perp(di, idx, bi)
                Hdiff = perp - threshold_perp
        
                def cond(di, bi, idx, tries, Hdiff, betamin, betamax, Pi):
                    return tf.math.reduce_all([tf.math.less(tries, num_iters), tf.math.greater(tf.math.abs(Hdiff), tol)])
                
                def body(di, bi, idx, tries, Hdiff, betamin, betamax, Pi):
                    # If not, increase or decrease precision
                    def true_fn():
                        betamin = bi
                        return tf.cond(tf.math.is_inf(betamax), lambda: bi * 2, lambda: (bi + betamax) / 2)
                    def false_fn():
                        betamax = bi
                        return tf.cond(tf.math.is_inf(betamin), lambda: bi / 2, lambda: (bi + betamin) / 2)
                    bi = tf.cond(tf.math.greater(Hdiff, 0), true_fn, false_fn)
                    
                    # Recompute the values
                    (prob, perp) = _calc_prob_and_perp(di, idx, bi)
                    Hdiff = perp - threshold_perp

                    return [di, bi, idx, tries+1, Hdiff, betamin, betamax, prob]
        
                _, _, _, _, _, _, _, p = tf.while_loop(cond, body, loop_vars = [di, bi, idx, tries, Hdiff, betamin, betamax, prob], back_prop=False) 

                return p
            
            P = tf.map_fn(_map_fn, P, parallel_iterations=16)
            P = P - tf.linalg.diag(tf.linalg.diag_part(P))

            return P
        
        P = _x2p(x, 1e-5, perplexity)
        P = P + tf.transpose(P)
        P = P / tf.math.reduce_sum(P)
        P = P * 4. # early exaggeration
        P = tf.math.maximum(P, 1e-12)

        it = tf.constant(0, tf.int32)

        initializer = tf.initializers.random_normal(stddev=1e-4)
        y = tf.get_variable('y', (self._num_samples, self._dim), tf.float32, initializer, trainable=False)
            
        # Run iterations
        def cond(it, P, gains, iY, y): 
            return it < max_iter
        def body(it, P, gains, iY, y):
        
            def _y2p(v, threshold=1e-12):
                sum_squared_v = tf.math.reduce_sum(tf.math.square(v), axis=-1, keepdims=True)
                cross_components = -2 * tf.linalg.matmul(v, v, transpose_b=True)
                squared_dist = sum_squared_v + cross_components + tf.transpose(sum_squared_v)
                unnorm_prob = tf.math.reciprocal(1 + squared_dist)
                unnorm_prob = unnorm_prob - tf.linalg.diag(tf.linalg.diag_part(unnorm_prob))
                prob = unnorm_prob / tf.math.reduce_sum(unnorm_prob)
                prob = tf.math.maximum(prob, threshold)

                return prob, unnorm_prob

            # Compute pairwise affinities
            Q, unnorm_Q = _y2p(y)

            # Compute gradient
            PQ = P - Q
            inter = tf.expand_dims(tf.transpose(PQ * unnorm_Q), axis=-1) * (tf.expand_dims(y, axis=1) - y)
            dY = tf.math.reduce_sum(inter, axis=1)

            # Perform the update
            momentum = tf.cond(tf.math.less(it, 20), lambda: initial_momentum, lambda: final_momentum)

            dY_sign = tf.cast(tf.math.sign(dY), tf.int32)
            iY_sign = tf.cast(tf.math.sign(iY), tf.int32)
            gains = (gains + 0.2) * tf.cast(tf.math.not_equal(dY_sign, iY_sign), dtype=tf.float32) + \
                    (gains * 0.8) * tf.cast(tf.math.equal(dY_sign, iY_sign), dtype=tf.float32)
            gains = tf.where(tf.math.less(gains, min_gain), min_gain*tf.ones_like(gains), gains)
            iY = momentum * iY - eta * (gains * dY)
            y = y + iY
            y = y - tf.math.reduce_mean(y, axis=0, keepdims=True)

            # Compute current value of cost function
            C = tf.math.reduce_sum(P * tf.math.log(P / Q))
            tf_print = tf.print("Iteration: ", it, " error is ", C)
            with tf.control_dependencies([tf_print]):
                P = tf.cond(tf.math.equal(it, 100), lambda: P / 4., lambda: P)

            return [it+1, P, gains, iY, y]

        # Return solution
        it, _, _, _, y = tf.while_loop(cond, body, loop_vars=[it, P, gains, iY, y], back_prop=False)
        
        return y


    
