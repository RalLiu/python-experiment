from flask import Flask, render_template, jsonify, request, send_from_directory
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
import datetime

app = Flask(__name__)
app.config['DATA_FOLDER'] = os.path.abspath('data_files')


def save_file(file):
    filename = file.filename
    file.save(os.path.join(app.config['DATA_FOLDER'], filename))


def delete_file(file):
    upload_dir = app.config['DATA_FOLDER']
    target_path = upload_dir + '\\' + file
    if os.path.isfile(target_path):
        os.remove(target_path)


@app.route('/')
def mainframe():
    file_names = os.listdir(app.config['DATA_FOLDER'])
    file_names = [f for f in file_names if os.path.isfile(os.path.join(app.config['DATA_FOLDER'], f))]
    return render_template('mainframe.html', list=file_names)


@app.route('/api/save_file', methods=['POST'])
def api_save_file():
    file = request.files['file']
    result = save_file(file)
    return jsonify({
        'status': 'success',
        'result': result
    })


@app.route('/api/delete_file', methods=['POST'])
def api_delete_file():
    file = request.form.get('filename')
    result = delete_file(file)
    return jsonify({
        'status': 'success',
        'result': result
    })


@app.route('/preview/<filename>')
def preview(filename):
    upload_dir = app.config['DATA_FOLDER']
    target_path = upload_dir + '\\' + filename
    if filename.endswith('.csv'):
        df = pd.read_csv(target_path)
    else:
        df = pd.read_excel(target_path)
    table_html = df[::len(df) // 50 + 1].to_html(index=False)
    return render_template('preview.html', table=table_html)


@app.route('/data_files/<path:filename>')
def download_file(filename):
    return send_from_directory('data_files', filename, as_attachment=True)


@app.route('/visualization/<path:filename>')
def visualization(filename):
    dp = pd.read_csv('data_files//' + filename)
    columns = dp.columns.tolist()
    if 'date' in columns:
        columns.remove('date')
    if 'dividends' in columns:
        columns.remove('dividends')
    if 'splits' in columns:
        columns.remove('splits')
    if 'symbol' in columns:
        columns.remove('symbol')
    return render_template('visualization.html', filename=filename, columns=columns)

@app.route('/data_cleaning/<filename>')
def data_cleaning_page(filename):
    return render_template('data_cleaning.html', filename=filename)


@app.route('/visualization/get-data', methods=['POST'])
def get_data():
    data = request.get_json()
    filename = data.get('filename')
    path = app.config['DATA_FOLDER'] + '\\' + filename
    xAxis = data.get('x')
    yAxis = data.get('y')
    if filename.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    x = df[xAxis].to_list()[::len(df) // 50 + 1]
    y = df[yAxis].to_list()[::len(df) // 50 + 1]
    open_lis = df['open'].to_list()[::len(df) // 50 + 1]
    high_lis = df['high'].to_list()[::len(df) // 50 + 1]
    low_lis = df['low'].to_list()[::len(df) // 50 + 1]
    close_lis = df['close'].to_list()[::len(df) // 50 + 1]
    adjclose_lis = df['adjclose'].to_list()[::len(df) // 50 + 1]
    volume_lis=df['volume'].to_list()[::len(df) // 50 + 1]
    columns = df.columns.to_list()
    if 'date' in columns:
        columns.remove('date')
    if 'dividends' in columns:
        columns.remove('dividends')
    if 'splits' in columns:
        columns.remove('splits')
    if 'symbol' in columns:
        columns.remove('symbol')

    return jsonify(
        {
            'columns': columns,
            'x': x,
            'y': y,
            'open': open_lis,
            'high': high_lis,
            'low': low_lis,
            'close': close_lis,
            'adjclose': adjclose_lis,
            'volume':volume_lis
        }
    )

@app.route('/prediction',methods=['POST'])
def prediction():
    filename = request.form.get('filename')
    path = app.config['DATA_FOLDER'] + '\\' + filename
    if filename.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    df=df.sort_values('date')
    #前一日数据
    df['open_lag1'] = df['open'].shift(1)
    df['high_lag1'] = df['high'].shift(1)
    df['low_lag1'] = df['low'].shift(1)
    df['close_lag1'] = df['close'].shift(1)
    df['volume_lag1'] = df['volume'].shift(1)
    df['return_lag1'] = (df['close'].shift(1) - df['open'].shift(1)) / df['open'].shift(1)
    #最近三天平均收益率：涨幅及趋势
    df['return_mean_3'] = ((df['close'] / df['close'].shift(1)) - 1).rolling(3).mean()
    #最近三天收盘价标准差：是否剧烈波动
    df['volatility_3'] = df['close'].rolling(3).std()
    #成交量和最近三日成交量比值：交易活跃度影响
    df['volume_ratio_3'] = df['volume'] / df['volume'].rolling(3).mean()
    features = ['open_lag1', 'high_lag1', 'low_lag1', 'close_lag1', 'volume_lag1',
            'return_lag1', 'return_mean_3', 'volatility_3', 'volume_ratio_3']
    target = ['open', 'high', 'low', 'close', 'volume']
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)
    history = df.copy()  # 用于滚动更新 
    future_data = []
    for i in range(20):
        history['return'] = (history['close'] / history['close'].shift(1)) - 1
        return_mean_3 = history['return'].iloc[-3:].mean()
        volatility_3 = history['close'].iloc[-3:].std()
        volume_ratio_3 = history['volume'].iloc[-1] / history['volume'].iloc[-3:].mean()
        last_row = history.iloc[-1]
        input_features = {
            'open_lag1': last_row['open'],
            'high_lag1': last_row['high'],
            'low_lag1': last_row['low'],
            'close_lag1': last_row['close'],
            'volume_lag1': last_row['volume'],
            'return_lag1': (last_row['close'] - last_row['open']) / last_row['open'],
            'return_mean_3': return_mean_3,
            'volatility_3': volatility_3,
            'volume_ratio_3': volume_ratio_3
        }
        X_future = pd.DataFrame([input_features])
        pred = model.predict(X_future)[0]
        pred_open, pred_high, pred_low, pred_close, pred_volume = pred
        date_obj = datetime.datetime.strptime(last_row['date'], '%Y-%m-%d')
        next_date = date_obj + datetime.timedelta(days=1)
        next_date = next_date.strftime('%Y-%m-%d')
        predicted_row = {
            'date': next_date,
            'open': pred_open,
            'high': pred_high,
            'low': pred_low,
            'close': pred_close,
            'volume': pred_volume,
            'adjclose': pred_close
        }
        future_data.append(predicted_row)
        history = pd.concat([history, pd.DataFrame([predicted_row])], ignore_index=True)
        
    future_df = pd.DataFrame(future_data)
    if filename.endswith('.csv') or filename.endswith('.xls'):
        change_name=filename[:-4]+'_prediction_data.csv'
        future_df.to_csv(app.config['DATA_FOLDER']+'//'+filename[:-4]+'_prediction_data.csv', index=False)
    else :
        change_name=filename[:-5]+'_prediction_data.csv'
        future_df.to_csv(app.config['DATA_FOLDER']+'//'+filename[:-5]+'_prediction_data.csv', index=False)
    
    return jsonify({
        'status': 'success',
        'change_name':change_name
    })

@app.route('/data_cleaning',methods=['POST'])
def data_cleaning():
    filename = request.form.get('filename')
    path = app.config['DATA_FOLDER'] + '\\' + filename
    if filename.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    
    # 创建数据清洗表单
    form_data = request.form
    
    # 默认使用均值填充缺失值
    missing_method = form_data.get('missing_method', 'mean')
    # 默认使用Z-score方法检测异常值
    outlier_method = form_data.get('outlier_method', 'zscore')
    # 默认Z-score阈值为3
    zscore_threshold = float(form_data.get('zscore_threshold', 3.0))
    # 默认IQR系数为1.5
    iqr_factor = float(form_data.get('iqr_factor', 1.5))
    
    # 创建原始数据的副本
    change_df = df.copy()
    
    # 处理缺失值
    numeric_columns = change_df.select_dtypes(include=['number']).columns
    
    for column in numeric_columns:
        # 检查列中是否有缺失值
        if change_df[column].isnull().any():
            if missing_method == 'mean':
                # 使用均值填充缺失值
                mean_value = change_df[column].mean()
                change_df[column].fillna(mean_value, inplace=True)
            elif missing_method == 'median':
                # 使用中位数填充缺失值
                median_value = change_df[column].median()
                change_df[column].fillna(median_value, inplace=True)
            elif missing_method == 'mode':
                # 使用众数填充缺失值
                mode_value = change_df[column].mode()[0]
                change_df[column].fillna(mode_value, inplace=True)
    
    # 处理非数值列的缺失值（使用众数填充）
    non_numeric_columns = change_df.select_dtypes(exclude=['number']).columns
    for column in non_numeric_columns:
        if change_df[column].isnull().any():
            # 对于非数值列，使用众数填充
            mode_value = change_df[column].mode()[0]
            change_df[column].fillna(mode_value, inplace=True)
    
    # 检测并处理异常值
    for column in numeric_columns:
        if outlier_method == 'zscore':
            # 使用Z-score方法检测异常值
            z_scores = np.abs((change_df[column] - change_df[column].mean()) / change_df[column].std())
            outliers = z_scores > zscore_threshold
            
            # 将异常值替换为列的均值
            if outliers.any():
                mean_value = change_df.loc[~outliers, column].mean()
                change_df.loc[outliers, column] = mean_value
                
        elif outlier_method == 'iqr':
            # 使用IQR方法检测异常值
            Q1 = change_df[column].quantile(0.25)
            Q3 = change_df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - iqr_factor * IQR
            upper_bound = Q3 + iqr_factor * IQR
            
            # 将异常值替换为上下界
            change_df.loc[change_df[column] < lower_bound, column] = lower_bound
            change_df.loc[change_df[column] > upper_bound, column] = upper_bound
    
    if filename.endswith('.csv') or filename.endswith('.xls'):
        change_name=filename[:-4]+'_cleaning_data.csv'
        change_df.to_csv(app.config['DATA_FOLDER']+'//'+filename[:-4]+'_cleaning_data.csv', index=False)
    else :
        change_name=filename[:-5]+'_cleaning_data.csv'
        change_df.to_csv(app.config['DATA_FOLDER']+'//'+filename[:-5]+'_cleaning_data.csv', index=False)
    return jsonify({
        'status': 'success',
        'change_name':change_name
    })

if __name__ == '__main__':
    if not os.path.exists(app.config['DATA_FOLDER']):
        os.makedirs(app.config['DATA_FOLDER'])
    app.run(debug=True)
