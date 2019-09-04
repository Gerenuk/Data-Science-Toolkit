def detect_frequent_vals(series, threshold=10):   
    top_vals = series.value_counts()
    
    faulty_values = []
    
    if len(top_vals)>=3:
        if top_vals.iloc[0] / top_vals.iloc[1] > threshold:
            faulty_values.append(top_vals.index[0])
            
        if top_vals.iloc[-1] / top_vals.iloc[-2] > threshold:
            faulty_values.append(top_vals.index[-1])
    
    return faulty_values