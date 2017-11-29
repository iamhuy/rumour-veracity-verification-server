from models import feature_selector, reducer, classifier, scaler

def predict(feature_vector):
    X_test = [feature_vector]

    if feature_selector != None:
        X_test = feature_selector.transform(X_test)

    if scaler != None:
        X_test = scaler.transform(X_test)

    if reducer != None:
        X_test = reducer.transform(X_test)

    y_pred_prob = classifier.predict_proba(X_test)
    y_pred = classifier.predict(X_test).tolist()

    return {
        'veracity': y_pred[0],
        'probability': y_pred_prob[0][y_pred][0]
    }


