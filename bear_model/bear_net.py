import tensorflow.compat.v2 as tf
from . import core
from tqdm import tqdm

epsilon = tf.keras.backend.epsilon()


def _bear_kmer_counts(kmer_seqs, kmer_total_counts,
                      condition_trans_counts=None, h=None, ar_func=None):
    """Get random variable of kmer transition counts conditioned on observed counts.

    Either condition_trans_counts is not None or both of h and ar_func are not None.

    Parameters
    ----------
    kmer_seqs : dtype
        A tensor of shape [A1, ..., An, lag, alphabet_size+1] of one-hot encoded kmers.
    kmer_total_counts : int
        A tensor of shape [A1, ..., An] of counts of each kmer.
    condition_trans_counts : int, default = None
        A tensor of shape [Am, ..., An, alphabet_size+1] of transition counts of each
        kmer to condition on. Set to 0 if None. Will broadcast if m>1.
    h : dtype, default = None
        A positive constant of the :math:`h` parameter from the BEAR model. Set to 1 if None.
    ar_func : function, default = None
        A function that takes a tensor of shape [A1, ..., An, lag, alphabet_size+1]
        of dtype and returns a tensor of shape [A1, ..., An, alphabet_size+1] of dtype
        of transition probabilities for each kmer. The autoregressive function.
        Set to 0 if None.

    Returns
    -------
    x : tensorflow probability distribution
        Distribution of kmer transition counts for a BEAR model.
    """
    dtype = kmer_seqs.dtype
    if condition_trans_counts is None:
        condition_trans_counts = tf.constant(0., dtype)
    if h is None or ar_func is None:
        h = tf.constant(1., dtype)

        def ar_func(x):
            return tf.constant(0., dtype)
    concentrations = ar_func(kmer_seqs) / h + condition_trans_counts + epsilon
    x = core.tfpDirichletMultinomialPerm(kmer_total_counts, concentrations, name='x')
    return x


def _ar_kmer_counts(kmer_seqs, kmer_total_counts, ar_func):
    """Get Random variable of kmer transition counts conditioned on observed counts.

    Parameters
    ----------
    kmer_seqs : dtype
        A tensor of shape [A1, ..., An, lag, alphabet_size+1] of one-hot encoded kmers.
    kmer_total_counts : int
        A tensor of shape [A1, ..., An] of counts of each kmer.
    ar_func : function, default = None
        A function that takes a tensor of shape [A1, ..., An, lag, alphabet_size+1]
        of dtype and returns a tensor of shape [A1, ..., An, alphabet_size+1] of dtype
        of transition probabilities for each kmer. The autoregressive function.
        Set to 0 if None.

    Returns
    -------
    x : tensorflow probability distribution
        Distribution of kmer transition counts for an AR model.
    """
    probs = ar_func(kmer_seqs) + epsilon
    x = core.tfpMultinomialPerm(kmer_total_counts, probs, name='x')
    return x


def _create_params(lag, alphabet_size, make_ar_func,
                   af_kwargs, dtype=tf.float64):
    """Define and get parameters of BEAR or AR distribution.

    Parameters
    ----------
    lag : int
    alphabet_size : int
    make_ar_func : function
        Takes lag, alphabet_size, af_kwargs, dtype and returns an ar_func.
    af_kwargs : dict
        Keyword arguments for particular ar_func. For example, filter_width.
    dtype : tensorflow dtype, default = tf.float64
        dtype for the ar_func and h_signed.

    Returns
    -------
    params: list
        List of parameters as tensorflow variables.
    h_signed : dtype
        :math:`\log(h)` where :math:`h` is the BEAR parameter.
    ar_func : function
        The autoregressive function.
    """
    ar_func, ar_func_params = make_ar_func(lag, alphabet_size, **af_kwargs, dtype=dtype)
    h_signed = tf.Variable(0, dtype=dtype, name='h_signed')
    params = ([h_signed] + ar_func_params)
    return params, h_signed, ar_func


def change_scope_params(lag, alphabet_size, make_ar_func,
                        af_kwargs, params, dtype=tf.float64):
    """Redefine and get parameters of BEAR or AR distribution in given scope.

    Used to to get unmirrored variables after training on multiple GPUs in parallel
    or to unpack a list of params into h and the autoregressive function.

    Parameters
    ----------
    lag : int
    alphabet_size : int
    make_ar_func : function
        Takes lag, alphabet_size, af_kwargs, dtype and returns an ar_func.
    af_kwargs : dict
        Keyword arguments for particular ar_func. For example, filter_width.
    params: list
        List of parameters as tensorflow variables.
    dtype : dtype, default = tf.float64
        dtype for the ar_func and h.

    Returns
    -------
    params : list
        List of parameters as tensorflow variables.
    h_signed : dtype
        :math:`\log(h)` where :math:`h` is the BEAR concentration parameter.
    ar_func : function
        A function that takes a tensor of shape [A1, ..., An, lag, alphabet_size+1]
        of dtype and returns a tensor of shape [A1, ..., An, alphabet_size+1] of dtype
        of transition probabilities for each kmer. The autoregressive function.
    """
    pos_in_params = 0
    h_nmir = tf.Variable(0, dtype=dtype, name='h_signed')
    h_nmir.assign(params[pos_in_params])
    pos_in_params += 1
    ar_func_nmir, ar_func_params_nmir = make_ar_func(lag, alphabet_size, **af_kwargs, dtype=dtype)
    for param in ar_func_params_nmir:
        param.assign(params[pos_in_params])
        pos_in_params += 1
    params_nmir = ([h_nmir] + ar_func_params_nmir)
    return params_nmir, h_nmir, ar_func_nmir


def _train_step(batch, num_kmers, h_signed, ar_func,
                params, acc_grads, train_ar):
    """Add gradient of unbiased estimate of loss to accumulated gradients.

    Parameters
    ----------
    batch : list of two tensors of the same dtype.
        The first element is a one hot encoding of kmers of shape
        [kmer_batch_size, lag, alphabet_size+1] and the second is the transition
        counts of size [kmer_batch_size, alphabet_size+1].
    num_kmers : int
        Total number of kmers seen in data. Used to normalize estimate of loss.
    h_signed : dtype
        :math:`\log(h)` where :math:`h` is the BEAR parameter.
    ar_func : function
        A function that takes a tensor of shape [A1, ..., An, lag, alphabet_size+1]
        of dtype and returns a tensor of shape [A1, ..., An, alphabet_size+1] of dtype
        of transition probabilities for each kmer. The autoregressive function.
    params : list
        List of parameters as tensorflow variables.
    acc_grads : list
        List of accumulated gradients for parameters.
    train_ar : bool
        Whether to evaluate the likelihood using an AR (True) or BEAR (False) model.

    Returns
    -------
    loss : dtype
        An unbiased estimate of the log likelihood of the data.
    """
    with tf.GradientTape() as grad_tape:
        kmer_batch_size = tf.shape(batch[0])[0]
        kmer_seqs = batch[0]
        transition_counts = batch[1]
        kmer_total_counts = tf.math.reduce_sum(transition_counts, axis=-1)

        if train_ar:
            post = _ar_kmer_counts(kmer_seqs, kmer_total_counts, ar_func)
        else:
            post = _bear_kmer_counts(kmer_seqs, kmer_total_counts,
                                     h=tf.math.exp(h_signed), ar_func=ar_func)
        log_likelihood = tf.reduce_sum(
            post.counts_log_prob(transition_counts))

        elbo = (num_kmers / kmer_batch_size) * log_likelihood
        loss = -elbo

    gradients = grad_tape.gradient(loss, params)
    for tv, grad in zip(acc_grads, gradients):
        if grad is not None:
            tv.assign_add(grad)
    return loss


def train(data, num_kmers, epochs, ds_loc, alphabet, lag, make_ar_func, af_kwargs,
          learning_rate, optimizer_name, train_ar, acc_steps=1,
          params_restart=None, writer=None, loss_save=None, dtype=tf.float64):
    """Train a BEAR or AR model using all available GPUs in parallel.

    Parameters
    ----------
    data : tensorflow data object
        Load sequence data using tools in dataloader.py. Minibatch before passing.
    num_kmers : int
        Total number of kmers seen in data. Used to normalize estimate of loss.
    epochs : int
    ds_loc : int
        Column in count data that corresponds with the training data.
    alphabet : str
        One of 'dna', 'rna', 'prot'.
    lag : int
    make_ar_func : function
        Takes lag, alphabet_size, af_kwargs, dtype and returns an ar_func. See ar_funcs submodule.
    af_kwargs : dict
        Keyword arguments for particular ar_func. For example, filter_width.
    learning_rate : float
    optimizer_name : str
        For example 'Adam'.
    train_ar : bool
        Whether to train an AR (True) or BEAR (False) model.
    writer : tensorboard writer object, default = None
    loss_save : list, default = None
        Pass a list to have losses at each step appended to it.
    acc_steps : int, default = 1
        Number of steps to accumulate gradients over.
    params_restart : list of tensorflow variables, default = None
        Pass the parameter list from a previous run to restart training.
    dtype : dtype, default = tf.float64

    Returns
    -------
    params: list
        List of parameters as tensorflow variables.
    h_signed : dtype
        :math:`\log(h)` where :math:`h` is the BEAR concentration parameter.
    ar_func : function
        The autoregressive function.
    """
    alphabet_size = len(core.alphabets_tf[alphabet]) - 1

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # Define parameters in parallel GPU usage scope.
        if params_restart is None:
            params, h_signed, ar_func = _create_params(
                lag, alphabet_size, make_ar_func,
                af_kwargs, dtype=dtype)
        else:
            params, h_signed, ar_func = change_scope_params(
                lag, alphabet_size, make_ar_func,
                af_kwargs, params_restart, dtype=dtype)

        # Set up accumulated gradient variables.p;';
        acc_grads = [tf.Variable(tf.zeros_like(tv), trainable=False) for tv in params]
        for tv in acc_grads:
            tv.assign(tf.zeros_like(tv))

        # Define optimizer in parallel GPU usage scope.
        optimizer = getattr(tf.keras.optimizers, optimizer_name)(
                                    learning_rate=learning_rate)

    # One hot encode kmers and get appropriate column from training data.
    def map_(kmers, counts):
        return core.tf_one_hot(kmers, alphabet), tf.gather(counts, ds_loc, axis=1)
    data = data.map(
        map_, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(10)
    # Get data iterable for use with GPU parallelization.
    data_iter = iter(strategy.experimental_distribute_dataset(data))

    # Define functions for updating parameters and accumulating gradients
    # with GPU parallelization.
    def add_grads(params, acc_grads, optimizer):
        optimizer.apply_gradients(zip(acc_grads, params))

    @tf.function
    def dist_add_grads(params, acc_grads, optimizer):
        strategy.run(add_grads, args=(params, acc_grads, optimizer))

    @tf.function
    def dist_train_step(batch, num_kmers, h_signed, ar_func,
                        params, acc_grads, train_ar):
        losses = strategy.run(_train_step, args=(
            batch, num_kmers, h_signed, ar_func,
            params, acc_grads, train_ar))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, losses,
                               axis=None)

    # Training loop
    loss = 0.
    step = 1
    for batch in tqdm(data_iter):
        # Accumulate gradients and update loss.
        loss += dist_train_step(batch, num_kmers, h_signed, ar_func,
                                params, acc_grads, train_ar)

        if step % acc_steps == 0:
            # Record loss
            if writer is not None:
                with writer.as_default():
                    tf.summary.scalar('elbo', - loss / acc_steps, step=step)
            if loss_save != None:
                loss_save.append(-loss/acc_steps)
            # Update gradients
            dist_add_grads(params, acc_grads, optimizer)
            # Reset accumulated gradients and cumulative loss to zero.
            for tv in acc_grads:
                tv.assign(tf.zeros_like(tv))
            loss = 0

        step += 1

    # Remove the distributed scope from the parameters.
    params, h_signed, ar_func = change_scope_params(
        lag, alphabet_size, make_ar_func,
        af_kwargs, params, dtype=dtype)
    return params, h_signed, ar_func

def _evaluation_step(batch, h, ar_func, van_reg, alphabet_size, use_train, dtype=tf.float64):
        kmer_seqs = batch[0]
        transition_counts_test = batch[1]
        if use_train:
            transition_counts_train = batch[2]
            van_condition = transition_counts_train[:, None, :] + van_reg[:, None]
        else:
            transition_counts_train = None
            van_condition = van_reg[:, None] * tf.ones([1, alphabet_size + 1], dtype=dtype)
        
        kmer_total_counts_test = tf.math.reduce_sum(transition_counts_test, axis=-1)
        # Get posteriors.
        post_ear = _bear_kmer_counts(kmer_seqs, kmer_total_counts_test,
                                     condition_trans_counts=transition_counts_train,
                                     h=h, ar_func=ar_func)
        post_arm = _ar_kmer_counts(kmer_seqs, kmer_total_counts_test, ar_func)
        post_van = _bear_kmer_counts(kmer_seqs, kmer_total_counts_test[:, None],
                                     condition_trans_counts=van_condition)
        # Get likelihoods.
        log_likelihood_ear = tf.reduce_sum(
            post_ear.counts_log_prob(transition_counts_test), axis=-1)
        log_likelihood_arm = tf.reduce_sum(
            post_arm.counts_log_prob(transition_counts_test))
        log_likelihood_van = tf.reduce_sum(
            post_van.counts_log_prob(transition_counts_test[:, None, :]), axis=0)
        # Get most likely transition and accuracy.
        ml_ear = post_ear.ml_output()
        ml_arm = post_arm.ml_output()
        ml_van = post_van.ml_output()
        oh_ml_ear = tf.cast(tf.math.equal(ml_ear[..., None],
                                          tf.range(alphabet_size+1, dtype=dtype)),
                            dtype=dtype)
        oh_ml_arm = tf.cast(tf.math.equal(ml_arm[..., None],
                                          tf.range(alphabet_size+1, dtype=dtype)),
                            dtype=dtype)
        oh_ml_van = tf.cast(tf.math.equal(ml_van[..., None],
                                          tf.range(alphabet_size+1, dtype=dtype)),
                            dtype=dtype)
        
        correct_ear = tf.math.reduce_sum(tf.math.reduce_sum(
            transition_counts_test*oh_ml_ear, axis=-1), axis=-1)
        correct_arm = tf.math.reduce_sum(transition_counts_test*oh_ml_arm)
        correct_van = tf.math.reduce_sum(tf.math.reduce_sum(
            transition_counts_test[:, None, :]*oh_ml_van, axis=0), axis=-1)
        # Sum total number of transitions.
        total_len = tf.math.reduce_sum(transition_counts_test)
        
        return (log_likelihood_ear, log_likelihood_arm, log_likelihood_van,
                correct_ear, correct_arm, correct_van, total_len)
    
@tf.function
def _distributed_evaluation_step(batch, h, ar_func, van_reg, alphabet_size, use_train,
                                 strategy):
    (log_likelihood_ear, log_likelihood_arm, log_likelihood_van,
        correct_ear, correct_ar, correct_van, total_len) = strategy.run(
        _evaluation_step, args=(batch, h, ar_func, van_reg, alphabet_size, use_train))
    return (strategy.reduce(tf.distribute.ReduceOp.SUM, log_likelihood_ear, axis=None),
            strategy.reduce(tf.distribute.ReduceOp.SUM, log_likelihood_arm, axis=None),
            strategy.reduce(tf.distribute.ReduceOp.SUM, log_likelihood_van, axis=None),
            strategy.reduce(tf.distribute.ReduceOp.SUM, correct_ear, axis=None),
            strategy.reduce(tf.distribute.ReduceOp.SUM, correct_ar, axis=None),
            strategy.reduce(tf.distribute.ReduceOp.SUM, correct_van, axis=None),
            strategy.reduce(tf.distribute.ReduceOp.SUM, total_len, axis=None))

def evaluation(data, ds_loc_train, ds_loc_test,
               alphabet, h, ar_func, van_reg, dtype=tf.float64):
    """Evaluate a trained BEAR, AR or BMM model. Can use multiple GPUs in parallel.

    Parameters
    ----------
    data : tensorflow data object
        Load sequence data using tools in dataloader.py. Minibatch before passing.
    ds_loc_train : int
        Column in count data that corresponds with the training data.
        Set to -1 for conditioning on training data.
    ds_loc_test : int
        Column in count data that corresponds with the testing data.
    alphabet : str
        One of 'dna', 'rna', 'prot'.
    h : dtype
        The :math:`h` parameter from the BEAR model.
    ar_func : function
        A function that takes a tensor of shape [A1, ..., An, lag, alphabet_size+1]
        of dtype and returns a tensor of shape [A1, ..., An, alphabet_size+1] of dtype
        of transition probabilities for each kmer. The autoregressive function.
    van_reg : 1D numpy or tensorflow array
        Prior on vanilla BEAR model (Dirichlet concentration parameter).
    dtype : dtype, default = tf.float64

    Returns
    -------
    log_likelihood_ear, log_likelihood_arm, log_likelihood_van : float
        Total log likelihood of the data with the model evaluated as a BEAR,
        AR or vanilla BEAR model.
    perplexity_ear, perplexity_arm, perplexity_van : float
        Perplexity of the data with the model evaluated as a BEAR,
        AR or vanilla BEAR model.
    accuracy_ear, accuracy_arm, accuracy_van : float
        Accuracy of the data with the model evaluated as a BEAR,
        AR or vanilla BEAR model. Ties in maximum model probability are resolved randomly.
    """
    alphabet_size = len(core.alphabets_tf[alphabet]) - 1

    strategy = tf.distribute.MirroredStrategy()
    use_train = ds_loc_train >= 0
    if use_train:
        def map_(kmers, counts):
            return (core.tf_one_hot(kmers, alphabet),
                    tf.gather(counts, ds_loc_test, axis=1),
                    tf.gather(counts, ds_loc_train, axis=1))
    else:
        def map_(kmers, counts):
            return (core.tf_one_hot(kmers, alphabet),
                    tf.gather(counts, ds_loc_test, axis=1))
    data_iter = iter(strategy.experimental_distribute_dataset(data.map(map_).prefetch(10)))

    log_likelihood_ear = tf.constant(0., dtype=dtype)
    log_likelihood_arm = tf.constant(0., dtype=dtype)
    log_likelihood_van = tf.zeros(len(van_reg), dtype=dtype)
    correct_ear = tf.constant(0., dtype=dtype)
    correct_arm = tf.constant(0., dtype=dtype)
    correct_van = tf.zeros(len(van_reg), dtype=dtype)
    total_len = tf.constant(0., dtype=dtype)

    for batch in data_iter:
        (ll_ear, ll_arm, ll_van, cor_ear, cor_arm, cor_van, tot_lens
        ) = _distributed_evaluation_step(
            batch, h, ar_func, van_reg, alphabet_size, use_train, strategy)
        log_likelihood_ear += ll_ear
        log_likelihood_arm += ll_arm
        log_likelihood_van += ll_van
        correct_ear += cor_ear
        correct_arm += cor_arm
        correct_van += cor_van
        total_len += tot_lens

    return (log_likelihood_ear, log_likelihood_arm, log_likelihood_van,
            tf.exp(-log_likelihood_ear/total_len),
            tf.exp(-log_likelihood_arm/total_len),
            tf.exp(-log_likelihood_van/total_len),
            correct_ear/total_len, correct_arm/total_len, correct_van/total_len)

def h_scan(data, ds_loc_train, ds_loc_test,
           alphabet, h, ar_func, dtype=tf.float64):
    """Evaluate a trained BEAR model at multiple h values. Can use multiple GPUs in parallel.

    Parameters
    ----------
    data : tensorflow data object
        Load sequence data using tools in dataloader.py. Minibatch before passing.
    ds_loc_train : int
        Column in count data that corresponds with the training data.
        Set to -1 for conditioning on training data.
    ds_loc_test : int
        Column in count data that corresponds with the testing data.
    alphabet : str
        One of 'dna', 'rna', 'prot'.
    h : tensor
        The :math:`h` parameter from the BEAR model.
    ar_func : function
        A function that takes a tensor of shape [A1, ..., An, lag, alphabet_size+1]
        of dtype and returns a tensor of shape [A1, ..., An, alphabet_size+1] of dtype
        of transition probabilities for each kmer. The autoregressive function.
    dtype : dtype, default = tf.float64

    Returns
    -------
    log_likelihood_ear, log_likelihood_arm, log_likelihood_van : float
        Total log likelihood of the data with the model evaluated as a BEAR,
        AR or vanilla BEAR model.
    perplexity_ear, perplexity_arm, perplexity_van : float
        Perplexity of the data with the model evaluated as a BEAR,
        AR or vanilla BEAR model.
    accuracy_ear, accuracy_arm, accuracy_van : float
        Accuracy of the data with the model evaluated as a BEAR,
        AR or vanilla BEAR model. Ties in maximum model probability are resolved randomly.
    """
    alphabet_size = len(core.alphabets_tf[alphabet]) - 1
    van_reg = tf.ones(1, dtype=dtype)

    strategy = tf.distribute.MirroredStrategy()
    use_train = ds_loc_train >= 0
    if use_train:
        def map_(kmers, counts):
            return (core.tf_one_hot(kmers, alphabet),
                    tf.gather(counts, ds_loc_test, axis=1),
                    tf.gather(counts, ds_loc_train, axis=1))
    else:
        def map_(kmers, counts):
            return (core.tf_one_hot(kmers, alphabet),
                    tf.gather(counts, ds_loc_test, axis=1))
    data_iter = iter(strategy.experimental_distribute_dataset(data.map(map_).prefetch(10)))

    log_likelihood_ear = tf.zeros(tf.shape(h), dtype=dtype)
    correct_ear = tf.zeros(tf.shape(h), dtype=dtype)
    total_len = tf.constant(0., dtype=dtype)
    
    for i, batch in enumerate(data_iter):
        if i == 0:
            len_ar_func_out = len(tf.shape(batch[1]))
            hs = tf.reshape(h, tf.concat([tf.shape(h), tf.ones(len_ar_func_out, dtype=tf.int32)], axis=0))
        (ll_ear, ll_arm, ll_van, cor_ear, cor_arm, cor_van, tot_lens
        ) = _distributed_evaluation_step(
            batch, hs, ar_func, van_reg, alphabet_size, use_train, strategy)
        log_likelihood_ear += ll_ear
        correct_ear += cor_ear
        total_len += tot_lens

    return (log_likelihood_ear, tf.exp(-log_likelihood_ear/total_len), correct_ear/total_len, )
