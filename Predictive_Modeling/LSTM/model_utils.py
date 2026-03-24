
from libraries import *
# model_utils.py

# LSTM model architecture

def create_model(numTS, numVariables, numNodes, percDropout,learning_rate, weight_decay,alpha, gamma
):
    # model = Sequential()
    # model.add(LSTM(numNodes, input_shape=(numTS, numVariables)))
    # model.add(Dropout(percDropout))
    # model.add(Dense(1, activation='sigmoid'))
    
    
    # LSTM learns sequence features, then Dense layer learns a more 
    # flexible combination before the final prediction
    model = Sequential()
    model.add(LSTM(numNodes, input_shape=(numTS, numVariables)))
    model.add(Dropout(percDropout))
    model.add(Dense(numNodes // 2, activation='relu'))
    model.add(Dropout(percDropout))
    model.add(Dense(1, activation='sigmoid'))
    




    # opt = Adagrad(learning_rate=learning_rate)
    # opt = tf.keras.optimizers.Adam(learning_rate=learning_rate,clipnorm=1.0)
    opt = tf.keras.optimizers.AdamW(learning_rate=learning_rate,
                                    weight_decay=weight_decay,
                                    clipnorm=1.0)
    
    loss1=tf.keras.losses.BinaryFocalCrossentropy(
    apply_class_balancing=True,
    alpha=alpha, 
    gamma=gamma)
    model.compile(
        loss=loss1,
        # loss='binary_crossentropy',
        optimizer=opt,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='acc'),
            tf.keras.metrics.AUC(name='auroc'),
            tf.keras.metrics.AUC(curve='PR', name='auprc'),
        ]
    )
    
    return model
