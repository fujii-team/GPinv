import tensorflow as tf
from GPflow import param
from functools import wraps
from .param import ParamList, ConcatDataHolder

class MultiFlow:
    """
    It is decorator class that modifies the method decorated by GPflow.AutoFlow.
    After modification of this decoration, the argument is converted to
    ConcatDataHolder.
    """
    def __call__(self, tf_method):
        @wraps(tf_method)
        def runnable(instance, *list_args):
            storage_name = '_' + tf_method.__name__ + '_AF_storage'
            if hasattr(instance, storage_name):
                # the method has been compiled already, get things out of storage
                storage = getattr(instance, storage_name)
            else:
                # the method needs to be compiled.
                storage = {}  # an empty dict to keep things in
                setattr(instance, storage_name, storage)
                storage['free_vars'] = tf.placeholder(tf.float64, [None])
                instance.make_tf_array(storage['free_vars'])
                # storage dict_data for the arguments
                storage['concat_list'] = \
                    [ConcatDataHolder(list_arg) for list_arg in list_args]
                # tf_array is passed to the tf_method
                with instance.tf_mode():
                    [c._begin_tf_mode() for c in storage['concat_list']]
                    storage['tf_result'] = tf_method(instance, *storage['concat_list'])
                    [c._end_tf_mode() for c in storage['concat_list']]
                storage['session'] = tf.Session()
                storage['session'].run(tf.initialize_all_variables(),\
                                    feed_dict=instance.get_feed_dict())
            feed_dict = {}
            feed_dict[storage['free_vars']] = instance.get_free_state()
            feed_dict.update(instance.get_feed_dict())
            # setting data into ConcatDataHolder
            for c, l in zip(storage['concat_list'], list_args):
                for i in range(len(l)):
                    c.data_holder_list[i].set_data(l[i])
                feed_dict.update(c.get_feed_dict())
            return storage['session'].run(storage['tf_result'], feed_dict=feed_dict)

        return runnable
