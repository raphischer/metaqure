import argparse
import os
import time
from itertools import product

import pandas as pd
import numpy as np
import umap
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import sample_colorscale, make_colorscale
from plotly.subplots import make_subplots
from PIL import Image

from run_meta_feature_extraction import load_meta_features
from data_loading import ds_name_to_subsample
from run_meta_learning import ML_RES_DIR
from run_log_processing import DB_COMPLETE, DB_BL, DB_SUB

from strep.index_and_rate import rate_database, load_database
from strep.util import load_meta, prop_dict_to_val
from strep.unit_reformatting import CustomUnitReformater
from strep.elex.graphs import assemble_scatter_data, GRAD, ENV_SYMBOLS
from strep.elex.util import rgb_to_rgba


PLOT_WIDTH = 700
PLOT_HEIGHT = PLOT_WIDTH // 3
LAMARR_COLORS = [
    '#009ee3', # aqua
    '#983082', # fresh violet
    '#ffbc29', # sunshine
    '#35cdb4', # carribean
    '#e82e82', # fuchsia
    '#59bdf7', # sky blue
    '#ec6469', # indian red
    '#706f6f', # gray
    '#4a4ad8', # corn flower
    '#0c122b',
    '#ffffff'
]
LAM_COL_SCALE = make_colorscale([LAMARR_COLORS[0], LAMARR_COLORS[2], LAMARR_COLORS[4]])
COL_FIVE = sample_colorscale(LAM_COL_SCALE, np.linspace(0, 1, 5))
COL_TEN = sample_colorscale(LAM_COL_SCALE, np.linspace(0, 1, 10))


TEX_TABLE_GENERAL = r'''
    \begin{tabular}$ALIGN
        \toprule 
        $DATA
        \bottomrule
    \end{tabular}'''


def format_value_for_table(val):
    if val < 1000 and val > 0.01:
        return f'{val:6.3f}'.strip()
    return f'{val:.1E}'.replace('E+0', 'E+').replace('E-0', 'E-').strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default='complete')
    args = parser.parse_args()
    DB = DB_COMPLETE if args.database == 'complete' else DB_SUB

    ##########################     LOAD RESULTS     ##########################

    all_meta_features = load_meta_features()
    formatter = CustomUnitReformater()
    db = load_database(DB)
    baselines = load_database(DB_BL)
    db['environment'] = db['environment'].map(lambda v: v.split(' - ')[0])
    baselines['environment'] = baselines['environment'].map(lambda v: v.split(' - ')[0])
    env_cols = {env: COL_FIVE[idx] for idx, env in enumerate(['Intel i9-13900K', 'Intel i7-6700', 'Intel i7-10610U', 'ARMv8 rev 1 (v8l)'])}
    meta_info = load_meta()
    meta_results = { fname[:-4]: pd.read_pickle(os.path.join(ML_RES_DIR, fname)) for fname in os.listdir(ML_RES_DIR) }

    ########################## PREPARE FOR PLOTTING ##########################

    objectives = list(zip(['accuracy', 'train_power_draw', 'compound_index'], ['Most accurate', 'Lowest energy', 'Best balanced'], ['O1', 'O2', 'O3']))
    col_short = {col: p_meta['shortname'] for col, p_meta in meta_info['properties'].items()}
    star_cols = list(col_short.keys()) + [list(col_short.keys())[0]]
    star_cols_short = [col_short[col] for col in star_cols]
    model_colors = {mod:col for mod, col in zip(pd.unique(db['model']), COL_TEN)}
    os.chdir('exp_results/paper_results')

    ####### DUMMY OUTPUT - for setting up pdf export of plotly
    fig = px.scatter(x=[0, 1, 2], y=[0, 1, 4])
    fig.write_image("dummy.pdf")
    time.sleep(0.5)
    os.remove("dummy.pdf")

    # ERRORS ACROSS PROPERTIES
    traces, titles = [], []
    for idx, (prop, prop_meta) in enumerate(meta_info['properties'].items()):
        row, col = 2 if idx >= len(meta_info['properties']) / 2 else 1, int(idx % (len(meta_info['properties']) / 2)) + 1
        for e_idx, (scale, trace, color) in enumerate(zip(['recalc_value', 'value'], ['Index', 'Value'], [COL_FIVE[0], COL_FIVE[4]])):
            res = meta_results['combined'][(scale, f'{prop}_test_err')]
            if e_idx == 0: # use same target unit for both scales!
                _, to_unit = formatter.reformat_value(res.iloc[0], prop_meta['unit'])
            reformatted = res.abs().map(lambda val: formatter.reformat_value(val, prop_meta['unit'], unit_to=to_unit, as_str=False))
            traces.append( (row, col, go.Box(name=trace, y=reformatted, legendgroup=trace, showlegend=idx==0, marker_color=color)) )
            if e_idx == 0:
                titles.append(f"{prop_meta['shortname']} {to_unit}")
    fig = make_subplots(rows=2, cols=int(len(meta_info['properties']) / 2), y_title='Real-valued abs. est. error', subplot_titles=titles, vertical_spacing=0.12, horizontal_spacing=0.05)
    for row, col, trace in traces:
        fig.add_trace(trace, row=row, col=col)
        fig.update_xaxes(visible=False, showticklabels=False, row=row, col=col)
        if row==2:
            fig.update_yaxes(type="log", row=row, col=col) 
    fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT*1.3, margin={'l': 57, 'r': 0, 'b': 0, 't': 18},
                      legend=dict(title='Meta-learning from values on scale:', orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5))
    fig.show()
    fig.write_image('errors_across_properties.pdf')

    ####### BASELINE COMPARISONS
    pfn_ds = pd.unique(baselines[baselines['model'] == 'PFN']['dataset'])
    meta_results['combined'][['dataset', 'environment', 'model']] = db[['dataset', 'environment', 'model']]
    comparison_data = {
        'Small data sets (71)': (
            meta_results['combined'][meta_results['combined']['dataset'].isin(pfn_ds)],
            baselines[baselines['dataset'].isin(pfn_ds)],
            db[db['dataset'].isin(pfn_ds)]
        ),
        'Large data sets (129)': (
            meta_results['combined'][~meta_results['combined']['dataset'].isin(pfn_ds)],
            baselines[~baselines['dataset'].isin(pfn_ds)],
            db[~db['dataset'].isin(pfn_ds)]
        )
    }
    fig = make_subplots(rows=2, cols=2, shared_yaxes=True, horizontal_spacing=0.01, vertical_spacing=0.01, shared_xaxes=True, row_titles=list(comparison_data.keys()))
    for row_idx, (meta_res, bl_res, exhau_res) in enumerate(comparison_data.values()):
        baseline_results = {mod: {'ene': [], 'acc': [], 'env': []} for mod in ['OURS', 'EXH'] + list(pd.unique(bl_res['model']))}
        for env, mod in product(reversed(env_cols.keys()), baseline_results.keys()):
            if mod == 'OURS':
                # access results of our method
                sub_pred = meta_res[meta_res['environment'] == env]
                rec_models = sub_pred.sort_values(['dataset', ('index', 'accuracy_test_pred')], ascending=False).groupby('dataset').first()['model']
                data = pd.concat([db[(db['environment'] == env) & (db['dataset'] == ds) & (db['model'] == mod)] for ds, mod in rec_models.items()])
            elif mod == 'EXH':
                # access results of our method
                sub_db = exhau_res[(exhau_res['environment'] == env)]
                data = sub_db.sort_values(['dataset','accuracy'], ascending=False).groupby('dataset').first()
                data['power_draw'] = sub_db.groupby('dataset')['power_draw'].sum()
                data['train_power_draw'] = sub_db.groupby('dataset')['train_power_draw'].sum()
            else:
                data = bl_res.loc[(bl_res['model'] == mod) & (bl_res['environment'] == env),['train_power_draw', 'power_draw', 'accuracy']].dropna()
                if data.size < 1:
                    continue
            baseline_results[mod]['ene'] = baseline_results[mod]['ene'] + data[['train_power_draw', 'power_draw']].sum(axis=1).values.tolist()
            baseline_results[mod]['acc'] = baseline_results[mod]['acc'] + data['accuracy'].values.tolist()
            baseline_results[mod]['env'] = baseline_results[mod]['env'] + [env] * data.shape[0]
        for idx, (mod, results) in enumerate( baseline_results.items() ):
            fig.add_trace(go.Box(x=results['ene'], y=results['env'], offsetgroup=f'{mod}{mod}', name=mod, legendgroup=mod, marker_color=COL_FIVE[idx], showlegend=row_idx==0), row=1+row_idx, col=1)
            fig.add_trace(go.Box(x=results['acc'], y=results['env'], offsetgroup=f'{mod}{mod}', name=mod, legendgroup=mod, marker_color=COL_FIVE[idx], showlegend=False), row=1+row_idx, col=2)
    fig.update_layout(boxmode='group', width=PLOT_WIDTH, height=PLOT_HEIGHT*2.5, margin={'l': 0, 'r': 15, 'b': 46, 't': 0},
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5))
    fig.update_traces(orientation='h')
    fig.update_xaxes(type="log", title='', row=1, col=1)
    fig.update_xaxes(type="log", title='Energy Draw [Ws]', row=2, col=1)
    fig.update_xaxes(title='Accuracy [%]', row=2, col=2)
    fig.show()
    fig.write_image(f'baseline_comparisons.pdf')

    ####### DS EMBEDDING
    ft_names = {'statistical': 'Manual', 'pca': 'PCA', 'ds2vec': 'DS2VEC', 'combined': 'Joined'}
    fig = make_subplots(rows=2, cols=4, subplot_titles=[f'{name} |X|={all_meta_features[key].shape[1]}' for key, name in ft_names.items()], horizontal_spacing=0.01, vertical_spacing=0.1)
    for idx, (key, name) in enumerate( ft_names.items() ):
        # add scatter
        reducer = umap.UMAP(random_state=42)
        embedding = reducer.fit_transform(all_meta_features[key])
        min_x, max_x, min_y, max_y = embedding[:,0].min(), embedding[:,0].max(), embedding[:,1].min(), embedding[:,1].max()
        fig.update_xaxes(range=[min_x-0.5, max_x+0.5], showticklabels=False, row=1, col=idx+1)
        fig.update_yaxes(range=[min_y-0.5, max_y+0.5], showticklabels=False, row=1, col=idx+1)
        colors, sizes = zip(*[(all_meta_features['statistical'].loc[ds,'n_predictors'], all_meta_features['statistical'].loc[ds,'n_instances']) for ds in all_meta_features[key].index])
        fig.add_trace( go.Scatter(x=embedding[:,0], y=embedding[:,1], mode='markers', showlegend=False, marker={'color': np.log(colors), 'size': np.log(sizes), 'coloraxis': 'coloraxis', 'sizemin': 1}), row=1, col=idx+1)
        # add bars for objective errors
        pred_error_mean = [meta_results[key][('index', f'{col}_test_err')].abs().mean() for col, _, _ in objectives]
        pred_error_stds = [meta_results[key][('index', f'{col}_test_err')].abs().std() for col, _, _ in objectives]
        fig.add_trace(go.Bar(x=list(zip(*objectives))[2], y=pred_error_mean, text=[f'{v:4.3f}' for v in pred_error_mean], textposition='auto', marker_color=COL_FIVE[0], showlegend=False), row=2, col=idx+1)
        fig.update_yaxes(range=[0, 0.18], showticklabels=idx==0, row=2, col=idx+1)
    fig.update_yaxes(title='S(a, c) MAE', row=2, col=1)
    # add traces for the scatter size legend
    for idx, n in enumerate([int(min(list(sizes))), 500, 5000, int(max(list(sizes)))]):
        fig.add_trace( go.Scatter(x=[-100], y=[-100], mode='markers', marker={'color': [1], 'size': [np.log(n)], 'colorscale': LAM_COL_SCALE, 'sizemin':1}, name=n), row=1, col=1)
    bar_ticks = [int(min(list(colors))), 10, 100, 1000, int(max(list(colors)))]
    fig.update_layout(
        coloraxis={'colorscale': LAM_COL_SCALE, 'colorbar': {'title': '# Features', 'len': 0.55, 'xanchor': 'right', 'x': 0.01, 'y': 0.8, 'tickvals': np.log(bar_ticks), 'ticktext': bar_ticks}},
        legend=dict(title='# Instances', y=0.5, x=0.5, xanchor='center', yanchor='middle', orientation='h'),
        width=PLOT_WIDTH, height=PLOT_HEIGHT*1.5, margin={'l': 0, 'r': 0, 'b': 0, 't': 18}
    )
    fig.show()
    fig.write_image(f'ds_embeddings.pdf')

    # PLOTS THAT REQUIRE RATED DB
    rated_db, bounds, _, _ = rate_database(db, meta_info)
    index_db = prop_dict_to_val(rated_db, 'index')
    
    ########### OTPIMAL MODEL CHOICE
    fig = make_subplots(rows=len(objectives), cols=len(env_cols), shared_yaxes=True, shared_xaxes=True, horizontal_spacing=0.01, vertical_spacing=0.01, subplot_titles=list(env_cols.keys()))
    for row_idx, (sort_col, text, _) in enumerate(objectives):
        for col_idx, env in enumerate( env_cols.keys() ):
            groundtruth = index_db[index_db['environment'] == env][['dataset','model',sort_col]]
            pred_col = ('index', f'{sort_col}_test_pred')
            predicted = meta_results['combined'].loc[groundtruth.index,pred_col]
            gt_and_pred = pd.concat([groundtruth, predicted], axis=1)
            true_best = gt_and_pred.sort_values(['dataset', sort_col], ascending=False).groupby('dataset').first()['model'].values
            pred_best = gt_and_pred.sort_values(['dataset', pred_col], ascending=False).groupby('dataset').first()['model'].values
            for bar_idx, (models, name) in enumerate(zip([true_best, pred_best], ['True best (exhaustive search)', 'Estimated best (via compositional meta-learning)'])):
                mods, counts = np.unique(models, return_counts=True)
                all_mod_counts = {mod: 0 for mod in model_colors.keys()}
                for mod, cnt in zip(mods, counts):
                    all_mod_counts[mod] = cnt
                fig.add_trace(go.Bar(x=list(all_mod_counts.keys()), y=list(all_mod_counts.values()), marker_color=COL_FIVE[bar_idx*4], name=name, showlegend=row_idx+col_idx==0), row=row_idx+1, col=col_idx+1)
        fig.update_yaxes(title=text, row=row_idx+1, col=1)
    fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT*1.8, margin={'l': 0, 'r': 0, 'b': 0, 't': 18},
                      legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5))
    fig.show()
    fig.write_image(f'optimal_model_choice.pdf')


    # collect differences across environments
    avg_dist = {col: [] for col in meta_info['properties'].keys()}
    mod_ds_mean_std, mod_ds_acc_std = [], []
    for (ds, model), data in index_db.groupby(['dataset', 'model']):
        for col in avg_dist.keys():
            avg_dist[col].append(data[col].std())
        if ds_name_to_subsample(ds)[0] is None: # base ds, not a subsampled variant
            mod_ds_acc_std.append( (avg_dist['accuracy'][-1], (ds, model)) )
            mod_ds_mean_std.append( (np.mean([val[-1] for val in avg_dist.values() ]), (ds, model)) )
    for val, (ds, model) in sorted(mod_ds_acc_std)[-10:]:
        print(f'{ds:<80} {model:<10} {val:5.3f}')
    MOD_DISP_IMPORT = sorted(mod_ds_mean_std)
    print(MOD_DISP_IMPORT[:10])
    print(MOD_DISP_IMPORT[-10:])
    DS_SEL = 'parkinsons'
    MOD_SEL = [('parkinsons', 'GNB'), ('parkinsons', 'SGD'), ('dry_bean_dataset', 'MLP')]


    # ##### VIOLIN of standard devs across environments
    fig = go.Figure()
    mean_std = [np.mean(std) for std in avg_dist.values()]
    mean_std = (mean_std - min(mean_std)) / (max(mean_std) - min(mean_std))
    colors = sample_colorscale(colorscale=LAM_COL_SCALE, samplepoints=mean_std)
    for color, (col, std_devs) in zip(colors, avg_dist.items()):
        fig.add_trace( go.Box(y=std_devs, x=[meta_info['properties'][col]["shortname"]] * len(std_devs), marker_color=color, showlegend=False) )
    fig.update_layout(width=PLOT_WIDTH * 0.38, height=PLOT_HEIGHT * 0.6, margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
    fig.update_yaxes(title='Std deviation')
    fig.show()
    fig.write_image('property_std_distrib.pdf')


    ####### SCATTER landscapes for a single data set
    ds_sub_db = rated_db[rated_db['dataset'] == DS_SEL]
    scatter = make_subplots(rows=1, cols=2, horizontal_spacing=0.04, subplot_titles=['Real-valued scale', 'Index scale'])
    marker_width = 8
    for fig_idx, scale in enumerate( ['value', 'index']) :
        plot_data, _, rating_pos = assemble_scatter_data(['Intel i9-13900K', 'ARMv8 rev 1 (v8l)'], ds_sub_db, scale, 'train_power_draw', 'accuracy', meta_info, bounds)
        axis_names = ['Train energy draw [Ws]', 'Accuracy [%]'] if scale == 'value' else ['Relative train energy draw [%]', 'Relative accuracy [%]']
        i_min, i_max = min([min(vals['index']) for vals in plot_data.values()]), max([max(vals['index']) for vals in plot_data.values()])
        # link model scatter points across multiple environment
        models = set.union(*[set(data['names']) for data in plot_data.values()])
        x, y, text = [], [], []
        for model in models:
            avail = 0
            for d_idx, data in enumerate(plot_data.values()):
                try:
                    idx = data['names'].index(model)
                    avail += 1
                    x.append(data['x'][idx])
                    y.append(data['y'][idx])
                except ValueError:
                    pass
            model_text = ['' if i != (avail - 1) // 2 else model for i in range(avail + 1)]
            text = text + model_text # place text near most middle node
            x.append(None)
            y.append(None)
        scatter.add_trace(go.Scatter(x=x, y=y, text=text, mode='lines+text', line={'color': 'black', 'width': marker_width / 8}, showlegend=False), row=1, col=1+fig_idx)
        for env_i, (env_name, data) in enumerate(plot_data.items()):
            text = [''] * len(data['x']) if ('names' not in data) or (len(plot_data) > 1) else data['names']
            scatter.add_trace(go.Scatter(
                x=data['x'], y=data['y'], name=env_name, text=text, mode='markers+text', marker_symbol=ENV_SYMBOLS[env_i],
                legendgroup=env_name, marker=dict(color=np.array(data['index']), coloraxis='coloraxis', size=marker_width, line={'color': 'black', 'width': marker_width / 8}), showlegend=fig_idx==0), row=1, col=1+fig_idx
            )
        scatter.update_xaxes(title=axis_names[0], showgrid=False, row=1, col=1+fig_idx)
        scatter.update_yaxes(title=axis_names[1] if fig_idx==0 else '', showgrid=False, row=1, col=1+fig_idx)
        grad = GRAD if scale == 'index' else GRAD.transpose(getattr(Image, 'FLIP_LEFT_RIGHT'))
        scatter.add_layout_image(dict(source=grad, sizing="stretch", opacity=0.5, layer="below", xref="x domain", yref="y domain", x=0, y=1, sizex=1, sizey=1), row=1, col=fig_idx+1)
    scatter.update_layout(coloraxis={'colorscale': LAM_COL_SCALE, 'colorbar': {'title': 'S(a, c)'}},
                          width=PLOT_WIDTH, height=PLOT_HEIGHT*1.2, margin={'l': 0, 'r': 0, 'b': 0, 't': 18},
                          legend=dict(orientation='v', yanchor="bottom", y=0.2, xanchor="right", x=0.98))
    scatter.update_traces(textposition='top center')
    scatter.show()
    scatter.write_image(f"scatter.pdf")

    ####### STAR PLOTS for the biggest performance differences
    fig = make_subplots(rows=1, cols=len(MOD_SEL), specs=[[{'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}]], subplot_titles=[f'{mod} on {ds if len(ds) < 17 else ds[:15] + ".."}' for ds, mod in MOD_SEL])
    for idx, (ds, mod) in enumerate(MOD_SEL):
        for e_idx, env in enumerate(env_cols.keys()):
            subdb = index_db[(index_db['dataset'] == ds) & (index_db['model'] == mod) & (index_db['environment'] == env)].iloc[0]
            fig.add_trace(go.Scatterpolar(
                r=[subdb[col] for col in star_cols], line={'color': COL_FIVE[e_idx]}, fillcolor=rgb_to_rgba(COL_FIVE[e_idx], 0.1),
                theta=star_cols_short, fill='toself', name=env, showlegend=idx==0
            ), row=1, col=idx+1)
    fig.update_annotations(yshift=20)
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)), width=PLOT_WIDTH, height=PLOT_HEIGHT,
        legend=dict( yanchor="top", y=-0.1, xanchor="center", x=0.5, orientation='h'), margin={'l': 0, 'r': 0, 'b': 0, 't': 40}
    )
    fig.show()
    fig.write_image(f'star_differences.pdf')
    