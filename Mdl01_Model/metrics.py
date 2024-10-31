
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.metrics import Metric

class CustomF1Score(tf.keras.metrics.Metric):
    def __init__(self, name='custom_f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1_score = tf.keras.metrics.F1Score()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.f1_score.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.f1_score.result()

    def reset_states(self):
        self.f1_score.reset_states()

# # eval_configの定義
# class R2ScoreWrapper(tf.keras.metrics.Metric):
#   def __init__(self, name="r2_score_wrapper", **kwargs):
#     super().__init__(name=name, **kwargs)
#     self.r2_score = tf.keras.metrics.R2Score()

#   def update_state(self, y_true, y_pred, sample_weight=None):
#     self.r2_score.update_state(y_true, y_pred, sample_weight)

#   def result(self):
#     return self.r2_score.result()

#   def reset_state(self):
#     self.r2_score.reset_state()

# ...
#   linear_model.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=0.1),
#     loss='mean_absolute_error',
#     metrics=[R2ScoreWrapper()]
#     # metrics=[tf.keras.metrics.R2Score()]
#     )
