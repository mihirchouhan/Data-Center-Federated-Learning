"""Validation-only selection for residual LSTM/GRU and native categorical XGBoost."""
import copy
import logging
import time
import numpy as np
from .experiment_data import weather_view, immutable_json
from .io_utils import atomic_write

LOG = logging.getLogger(__name__)


def lag_mapping():
    from .experiment_data import FEATURES
    return [{'column': step * 13 + j, 'feature': name, 'minutes_before_origin': 60 - step * 10,
             'type': 'q'} for step in range(6) for j, name in enumerate(FEATURES[:-1])] + [
             {'column': 78, 'feature': 'node_index', 'type': 'c', 'encoding': 'frozen node_names array index'}]


def lag_features(x, weather):
    if x.shape[1:] != (6, 14):
        raise ValueError('Unexpected lag shape')
    result = np.empty((len(x), 79), dtype=np.float32)
    result[:, :78] = x[:, :, :13].reshape(len(x), 78)
    result[:, 78] = x[:, -1, 13]
    if weather == 'masked':
        result[:, [step * 13 + j for step in range(6) for j in (7, 8)]] = 0
    elif weather != 'enabled':
        raise ValueError('Unknown weather option')
    return result


def training_indices(data, config):
    count = len(data.array('y_train'))
    limit = config.get('train_limit')
    if limit is None:
        return np.arange(count, dtype=np.int64)
    # Smoke only: fixed, evenly distributed indices preserve coverage across node blocks.
    if config['kind'] != 'pilot' or not 0 < limit <= count:
        raise ValueError('Training subsets are restricted to declared pilots')
    return np.linspace(0, count - 1, limit, dtype=np.int64)


def neural_model(data, config):
    from .model import SequenceRegressor
    return SequenceRegressor(14, 1, hidden_size=config['hidden_size'], num_layers=config['num_layers'],
                             model=config['model'], node_count=len(data.names),
                             node_embedding_size=config['node_embedding_size'],
                             baseline_coefficients=data.coefficients.tolist())


def neural_predict(model, x, config):
    import torch
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(x), config['eval_batch_size']):
            batch = weather_view(np.array(x[start:start + config['eval_batch_size']]), config['weather'])
            result.append(model(torch.from_numpy(batch).to(config['device'])).cpu().numpy().reshape(-1))
    return np.concatenate(result)


def select_if_better(score, best, selected, iteration):
    # Strict improvement keeps persistence (iteration 0) and the earliest tied iteration.
    if not np.isfinite(score):
        raise ValueError('Nonfinite validation score')
    return (score, iteration, True) if score < best else (best, selected, False)


def fit_neural(data, config, output):
    import torch
    torch.set_num_threads(config['num_threads'])
    torch.manual_seed(config['seed'])
    if config['device'].startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; CPU fallback is disabled')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if config['device'].startswith('cuda'):
        torch.cuda.manual_seed_all(config['seed'])
        torch.cuda.reset_peak_memory_stats(config['device'])
    model = neural_model(data, config).to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    x, y, _, _ = data.split('train')
    xv, yv, _, _ = data.split('val')
    indices = training_indices(data, config)
    rng = np.random.default_rng(config['seed'])
    # Initial residual zero-head is the same persistence candidate for every family.
    best = data.selection_rmse_w(neural_predict(model, xv, config), yv)
    baseline_score = best
    selected, bad_epochs, global_step = 0, 0, 0
    best_weights = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
    history = [{'epoch': 0, 'validation_rmse_w': best, 'candidate': 'persistence'}]
    for epoch in range(1, config['epochs'] + 1):
        started = time.perf_counter()
        model.train()
        order = rng.permutation(indices)
        total, observed = 0., 0
        for start in range(0, len(order), config['batch_size']):
            idx = order[start:start + config['batch_size']]
            batch = weather_view(np.array(x[idx]), config['weather'])
            target = torch.from_numpy(np.array(y[idx])).to(config['device'])
            optimizer.zero_grad(set_to_none=True)
            prediction = model(torch.from_numpy(batch).to(config['device']))
            loss = torch.nn.functional.mse_loss(prediction, target)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(idx)
            observed += len(idx)
            global_step += 1
        training_seconds = time.perf_counter() - started
        validation_started = time.perf_counter()
        score = data.selection_rmse_w(neural_predict(model, xv, config), yv)
        validation_seconds = time.perf_counter() - validation_started
        best, selected, improved = select_if_better(score, best, selected, epoch)
        if improved:
            best_weights = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1
        record = {'epoch': epoch, 'train_mse_scaled': total / observed, 'validation_rmse_w': score,
                  'epoch_seconds': time.perf_counter() - started, 'training_seconds': training_seconds,
                  'validation_seconds': validation_seconds}
        history.append(record)
        LOG.info('%s epoch=%d validation RMSE=%.4f W selected=%d duration=%.2fs', config['model'], epoch, score, selected, record['epoch_seconds'])
        atomic_write(output / 'last.pt', lambda f: torch.save({'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(), 'completed_epochs': epoch, 'global_step': global_step,
                     'rng_state': rng.bit_generator.state, 'torch_rng_state': torch.get_rng_state(),
                     'best_epoch': selected, 'best_validation_rmse_w': best}, f))
        if bad_epochs >= config['patience']:
            break
    model.load_state_dict(best_weights)
    atomic_write(output / 'selected.pt', lambda f: torch.save({'model_state_dict': best_weights}, f))
    immutable_json(output / 'history.json', history)
    result = {'selection_metric': 'validation_rmse_w', 'selected_iteration': selected,
              'selected_candidate': 'persistence' if selected == 0 else config['model'],
              'best_validation_rmse_w': best, 'persistence_validation_rmse_w': baseline_score,
              'completed_epochs': epoch, 'global_step': global_step, 'training_windows': len(indices),
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'model_file': 'selected.pt', 'torch_peak_allocated_gpu_bytes':
              torch.cuda.max_memory_allocated(config['device']) if config['device'].startswith('cuda') else None}
    return result, neural_predict(model, xv, config)


def xgb_matrix(x, weather, nthread, labels=None, ref=None, quantile=False, max_bin=256):
    import xgboost as xgb
    features = lag_features(x, weather)
    options = dict(feature_names=[f'lag_{i}' for i in range(78)] + ['node_index'],
                   feature_types=['q'] * 78 + ['c'], nthread=nthread, enable_categorical=True)
    if labels is not None:
        options['label'] = labels
    if quantile:
        return xgb.QuantileDMatrix(features, ref=ref, max_bin=max_bin, **options)
    return xgb.DMatrix(features, **options)


def fit_xgboost(data, config, output):
    import xgboost as xgb
    if config['device'].startswith('cuda'):
        import torch
        if not torch.cuda.is_available() or not xgb.build_info().get('USE_CUDA', False):
            raise RuntimeError('CUDA is unavailable in PyTorch or the XGBoost build')
    x, y, _, _ = data.split('train')
    xv, yv, _, _ = data.split('val')
    indices = training_indices(data, config)
    if len(indices) != len(x):
        x, y = x[indices], y[indices]
    train_residual = np.asarray(y[:, 0], dtype=np.float64) - data.baseline_scaled(x)
    val_baseline = data.baseline_scaled(xv)
    val_residual = np.asarray(yv[:, 0], dtype=np.float64) - val_baseline
    baseline_score = data.selection_rmse_w(val_baseline, yv)
    quant_started = time.perf_counter()
    dtrain = xgb_matrix(x, config['weather'], config['num_threads'], train_residual,
                        quantile=True, max_bin=config['max_bin'])
    dval = xgb_matrix(xv, config['weather'], config['num_threads'], val_residual,
                      ref=dtrain, quantile=True, max_bin=config['max_bin'])
    quant_seconds = time.perf_counter() - quant_started
    params = {'objective': 'reg:squarederror', 'tree_method': 'hist', 'device': config['device'],
              'seed': config['seed'], 'eta': config['learning_rate'], 'max_depth': config['max_depth'],
              'max_bin': config['max_bin'], 'subsample': config['subsample'],
              'colsample_bytree': config['colsample_bytree'], 'nthread': config['num_threads'],
              'base_score': 0.0, 'disable_default_eval_metric': 1,
              'max_cat_to_onehot': 4, 'validate_parameters': True}
    history = {}
    scale_w = data.meta['y_std'][0] * 1000
    def validation_metric(prediction, matrix):
        error = prediction.astype(np.float64) + val_baseline - np.asarray(yv[:, 0], dtype=np.float64)
        return 'rmse_w', float(np.sqrt(np.mean(error ** 2)) * scale_w)
    started = time.perf_counter()
    booster = xgb.train(params, dtrain, num_boost_round=config['trees'], evals=[(dval, 'validation')],
                        custom_metric=validation_metric, maximize=False, evals_result=history,
                        early_stopping_rounds=config['patience'], verbose_eval=10)
    fit_seconds = time.perf_counter() - started
    # Native Booster prediction does NOT automatically restrict to best_iteration.
    selected_trees = int(booster.best_iteration) + 1
    predictions = booster.predict(dval, iteration_range=(0, selected_trees)).astype(np.float64) + val_baseline
    score = data.selection_rmse_w(predictions, yv)
    best, selected, improved = select_if_better(score, baseline_score, 0, selected_trees)
    if not improved:
        predictions = val_baseline
    model_path = output / 'selected.ubj'
    # Save only selected trees; a baseline selection is explicitly recorded and ignored at inference.
    selected_booster = booster[:selected_trees]
    selected_booster.save_model(model_path)
    immutable_json(output / 'lag_mapping.json', lag_mapping())
    immutable_json(output / 'history.json', {'epoch_0_persistence_rmse_w': baseline_score, **history})
    config_saved = selected_booster.save_config()
    immutable_json(output / 'xgboost_config.json', __import__('json').loads(config_saved))
    if config['device'].startswith('cuda') and '"device":"cuda:' not in config_saved:
        raise RuntimeError('XGBoost silently fell back from requested CUDA device')
    return {'selection_metric': 'validation_rmse_w', 'selected_iteration': selected,
            'selected_candidate': 'xgboost' if improved else 'persistence', 'best_validation_rmse_w': best,
            'persistence_validation_rmse_w': baseline_score, 'completed_trees': booster.num_boosted_rounds(),
            'model_file': 'selected.ubj', 'quantization_seconds': quant_seconds, 'boosting_seconds': fit_seconds,
            'training_windows': len(indices), 'build_info': xgb.build_info()}, predictions


def predict_selected(data, config, output, selection, split):
    x, _, _, _ = data.split(split)
    if config['model'] in ('lstm', 'gru'):
        import torch
        torch.set_num_threads(config['num_threads'])
        model = neural_model(data, config).to(config['device'])
        model.load_state_dict(torch.load(output / selection['model_file'], map_location='cpu', weights_only=True)['model_state_dict'])
        return neural_predict(model, x, config)
    if selection['selected_iteration'] == 0:
        return data.baseline_scaled(x)
    import xgboost as xgb
    booster = xgb.Booster(model_file=output / selection['model_file'])
    booster.set_param({'device': config['device'], 'nthread': config['num_threads']})
    matrix = xgb_matrix(x, config['weather'], config['num_threads'])
    return booster.predict(matrix, iteration_range=(0, selection['selected_iteration'])).astype(np.float64) + data.baseline_scaled(x)
