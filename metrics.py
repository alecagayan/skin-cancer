import keras.backend as K
import tensorflow as tf

def balanced_accuracy(num_classes, categorical=True):
    """
    Calculates the mean of the per-class accuracies (balanced accuracy), or standard categorical accuracy.
    Same as sklearn.metrics.balanced_accuracy_score and sklearn.metrics.recall_score with macro average
    
    # References
        https://stackoverflow.com/questions/44477489
        https://stackoverflow.com/a/41717938/2437361
        https://stackoverflow.com/a/50266195/2437361
    """
    def fn(y_true, y_pred):
        #print('--------------------------')
        #print('y_true:', y_true)
        #print('y_pred:', y_pred)
        #class_id_true = K.argmax(y_true, axis=-1)
        #print('class id true:', class_id_true)
        #class_id_pred = K.argmax(y_pred, axis=-1)
        #print('class id pred:', class_id_pred)
        #equal_check = K.equal(class_id_true, class_id_pred)
        #print('equal_check:', equal_check)
        #equal_check_cast = K.cast(equal_check, K.floatx())
        #print('equal_check_cast:', equal_check_cast)

        result = 0.0
        if categorical:
            """ Use default sklearn definition for categorical_accuracy """
            result = K.cast(K.equal(K.argmax(y_true, axis=-1),
                                    K.argmax(y_pred, axis=-1)),
                            K.floatx()) 
        else:
            """ Use custom definition for balanced accuracy """
            class_acc_total = 0
            seen_classes = 0
            
            for c in range(num_classes):
                #print('c:', c)
                class_id_true = K.argmax(y_true, axis=-1)
                class_id_pred = K.argmax(y_pred, axis=-1)
                accuracy_mask = K.cast(K.equal(class_id_true, c), 'int32')
                #print('accuracy_mask:', accuracy_mask)
                class_acc_tensor = K.cast(K.equal(class_id_true, class_id_pred), 'int32') * accuracy_mask
                #print('class_acc_tensor:', class_acc_tensor)
                accuracy_mask_sum = K.sum(accuracy_mask)
                #print('accuracy_mask_sum:', accuracy_mask_sum)
                class_acc = K.cast(K.sum(class_acc_tensor) / K.maximum(accuracy_mask_sum, 1), K.floatx())
                #print('class_acc:', class_acc)
                class_acc_total += class_acc
                #print('class_acc_total:', class_acc_total)
                
                condition = K.equal(accuracy_mask_sum, 0)
                #print('condition:', condition)
                seen_classes = K.switch(condition, seen_classes, seen_classes+1)
                #print('seen_classes:', seen_classes)

            #print('result:', class_acc_total / K.cast(seen_classes, K.floatx()))
            result = class_acc_total / K.cast(seen_classes, K.floatx())
           
        return result
    fn.__name__ = 'balanced_accuracy'
    return fn
