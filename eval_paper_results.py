import argparse
import os
import time

import pandas as pd
import numpy as np
import umap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from data_loading import ds_name_to_subsample

from strep.index_and_rate import rate_database, find_relevant_metrics, load_database, index_to_value
from strep.util import load_meta, prop_dict_to_val, lookup_meta
from strep.unit_reformatting import CustomUnitReformater
from strep.elex.graphs import assemble_scatter_data, create_scatter_graph, add_rating_background
from strep.elex.util import RATING_COLORS, rgb_to_rgba, ENV_SYMBOLS, PATTERNS, RATING_COLOR_SCALE


PLOT_WIDTH = 700
PLOT_HEIGHT = PLOT_WIDTH // 3
LAMARR_COLORS = ['#009ee3', '#983082', '#ffbc29', '#35cdb4', '#e82e82', '#59bdf7', '#ec6469', '#706f6f', '#4a4ad8', '#0c122b', '#ffffff']


TEX_TABLE_GENERAL = r'''
    \begin{tabular}$ALIGN
        \toprule 
        $DATA
        \bottomrule
    \end{tabular}'''


def load_meta_features(dirname):
    meta_features = {}
    for meta_ft_file in os.listdir(dirname):
        if not '.csv' in meta_ft_file:
            continue
        meta_features[meta_ft_file.replace('.csv', '')] = pd.read_csv(os.path.join(dirname, meta_ft_file)).set_index('Unnamed: 0').fillna(0)
    meta_features['combined'] = pd.concat(meta_features.values(), axis=1)
    return meta_features


def format_value_for_table(val):
    if val < 1000 and val > 0.01:
        return f'{val:6.3f}'.strip()
    return f'{val:.1E}'.replace('E+0', 'E+').replace('E-0', 'E-').strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-features-dir", default='meta_features')
    parser.add_argument("--database", default='exp_results/databases/complete.pkl')
    args = parser.parse_args()

    ########################## PREPARE FOR PLOTTING ##########################

    all_meta_features = load_meta_features(args.meta_features_dir)
    formatter = CustomUnitReformater()
    del(all_meta_features['combined'])
    db = load_database(args.database)
    baselines = load_database('exp_results/databases/baselines.pkl')
    baselines = baselines.dropna(axis=1, how='all').dropna() # remove the data sets where baselines failed

    # for ds, data in baselines.groupby('dataset'):
    #     best_m = data.sort_values('accuracy').iloc[-1]
    #     best_m2 = db[(db['dataset'] == ds) & (db['environment'] == best_m['environment'])].sort_values('accuracy').iloc[-1]
    #     print(f'{ds:<50} {best_m["model"]:<5} ACC {best_m["accuracy"]:4.2f} PWR {best_m["power_draw"] + best_m["train_power_draw"]:8.2f}   {best_m2["model"]:<5} ACC {best_m2["accuracy"]:4.2f} PWR {best_m2["power_draw"] + best_m2["train_power_draw"]:8.2f}')

    db['environment'] = db['environment'].map(lambda v: v.split(' - ')[0])
    baselines['environment'] = baselines['environment'].map(lambda v: v.split(' - ')[0])
    meta = load_meta()
    rated_db, bounds, _, _ = rate_database(db, meta)
    index_db = prop_dict_to_val(rated_db, 'index')

    col_short = {col: p_meta['shortname'] for col, p_meta in meta['properties'].items()}
    star_cols = list(col_short.keys()) + [list(col_short.keys())[0]]
    star_cols_short = [col_short[col] for col in star_cols]
    model_colors = {mod:col for mod, col in zip(pd.unique(db['model']), ['rgb(84, 84, 107)', RATING_COLORS[1], RATING_COLORS[2], 'rgb(48, 155, 137)', RATING_COLORS[3], RATING_COLORS[0], 'rgb(85, 48, 155)', RATING_COLORS[4], 'rgb(155, 48, 78)', 'rgb(84, 107, 95)'])}



    # collect differences across environments
    avg_dist = {col: [] for col in meta['properties'].keys()}
    mod_ds_mean_std, mod_ds_acc_std = [], []
    for (ds, model), data in index_db.groupby(['dataset', 'model']):
        for col in avg_dist.keys():
            avg_dist[col].append(data[col].std())
        if ds_name_to_subsample(ds)[0] is None: # base ds, not a subsampled variant
            mod_ds_acc_std.append( (avg_dist['accuracy'][-1], (ds, model)) )
            mod_ds_mean_std.append( (np.mean([val[-1] for val in avg_dist.values() ]), (ds, model)) )
    for val, (ds, model) in sorted(mod_ds_acc_std)[-10:]:
        print(f'{ds:<80} {model:<10} {val:5.3f}')
    MOD_DISP_IMPORT = sorted(mod_ds_mean_std) + [sorted(mod_ds_acc_std)[-1]]
    print(MOD_DISP_IMPORT[-1])
    DS_SEL = MOD_DISP_IMPORT[-1][1][0]
    MOD_SEL = MOD_DISP_IMPORT[-3:]

    meta_res_path = os.path.join('exp_results', 'meta_learning')
    meta_results = { fname[:-4]: pd.read_pickle(os.path.join(meta_res_path, fname)) for fname in os.listdir(meta_res_path) }


    ##########################     PLOT FIGURES     ##########################
            
    os.chdir('paper_results')
    ####### DUMMY OUTPUT - for setting up pdf export of plotly
    fig = px.scatter(x=[0, 1, 2], y=[0, 1, 4])
    fig.write_image("dummy.pdf")
    time.sleep(0.5)
    os.remove("dummy.pdf")

    # TABLE COMPARISON WITH BASELINE
    env_results = {env: [] for env in pd.unique(rated_db['environment'])}
    for (ds, task, env), subdata in rated_db.groupby(['dataset', 'task', 'environment']):
        bl = baselines[(baselines['dataset'] == ds) & (baselines['environment'] == env) & (baselines['model'] == 'PFN16')]
        if bl.shape[0] > 0:
            pred = meta_results['combined'].loc[subdata.index]
            pred_best = pred.loc[:,(f'use_env__index')].sort_values('compound_index_test_pred', ascending=False).index[0]
            actual_best = subdata.sort_values('compound_index', ascending=False).index[0]
            env_results[env].append([
                db.loc[pred_best]['accuracy'], # ours acc
                db.loc[pred_best][['power_draw', 'train_power_draw']].sum(), # ours power
                bl.iloc[0]['accuracy'].sum(), # baseline acc
                bl.iloc[0][['power_draw', 'train_power_draw']].sum() # baseline power
            ])
    tex_rows = [
        r'Environment & \multicolumn{2}{c}{Our method} & \multicolumn{2}{c}{TabPFN} \\',
        r'& Accuracy [%] & Power Draw [Ws] & Accuracy [%] & Power Draw [Ws] \\',
        r'\midrule'
    ]
    for env, results in env_results.items():
        env_mean, env_std = np.array(results).mean(axis=0), np.array(results).std(axis=0)
        tex_rows.append( ' & '.join([env] + [f'{mean:5.2f} ($\pm${std:5.2f})' for mean, std in zip(env_mean, env_std)]) + r' \\')
    final_text = TEX_TABLE_GENERAL.replace('$DATA', '\n        '.join(tex_rows)).replace('$ALIGN', r'{c|cc|cc}')
    final_text = final_text.replace('%', r'\%').replace('#', r'\#').replace("µ", r"$\mu$")
    with open('baseline_comparison.tex', 'w') as outf:
        outf.write(final_text)
    

    # BEST METHOD PERFORMANCE ON EACH DATA SET
    # tex_rows = [
    #     # r'& \multicolumn{2}{c}{Default} & \multicolumn{2}{c}{Training on real measurements} \\',
    #     r'& Default & Without Env Info & Without Index Scaling \\',
    #     r'\midrule'
    # ]
    # fig = make_subplots(rows=1, cols=len(meta['properties']))
    # for idx, (col, col_meta) in enumerate(meta['properties'].items()):
    #     # lowest, lowest_idx = np.inf, -1
    #     for env_info, scale in [('use_env', 'rec_index'), ('use_env', 'value')]:
    #         res = meta_results['combined'].xs(f'{env_info}__{scale}', level=0, axis=1)[f'{col}_test_err']
    #         trace = 'Index' if scale == 'rec_index' else 'Value'
    #         fig.add_trace(go.Violin(x=[trace]*len(res), y=res, name=trace, legendgroup=trace, showlegend=idx==0), row=1, col=idx+1)
    # fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
    # fig.show()
        #     error_res = meta_results['combined'].xs(f'{env_info}__{scale}', level=0, axis=1)[f'{col}_test_err']
        #     results.append( (error_res.abs().mean(), error_res.abs().std()) )
    #     to_unit = formatter.reformat_value(results[0][0], col_meta['unit'])[1]
    #     format_res = [f'{formatter.reformat_value(mae, col_meta["unit"], to_unit)[0]} ($\pm${formatter.reformat_value(std, col_meta["unit"], to_unit)[0]})' for mae, std in results]
    #     lowest_idx = np.argmin([val[0] for val in results]) # check for lowest MAE
    #     format_res[lowest_idx] = r'\textbf{' + format_res[lowest_idx] + r'}'
    #     tex_rows.append(' & '.join([f'{col_meta["shortname"]} {to_unit}'] + format_res) + r' \\')
    # final_text = TEX_TABLE_GENERAL.replace('$DATA', '\n        '.join(tex_rows)).replace('$ALIGN', r'{c|ccc}')
    # final_text = final_text.replace('%', r'\%').replace('#', r'\#').replace("µ", r"$\mu$")
    # with open('meta_learn_errors.tex', 'w') as outf:
    #     outf.write(final_text)
        
    # ####### STAR PLOTS for the biggest performance differences
    fig = make_subplots(rows=1, cols=len(MOD_SEL), specs=[[{'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}]], subplot_titles=[f'{mod} on {ds if len(ds) < 20 else ds[:20] + ".."}' for _, (ds, mod) in MOD_SEL])
    for idx, (_, (ds, mod)) in enumerate(MOD_SEL):
        for e_idx, env in enumerate(pd.unique(index_db['environment'])):
            subdb = index_db[(index_db['dataset'] == ds) & (index_db['model'] == mod) & (index_db['environment'] == env)].iloc[0]
            fig.add_trace(go.Scatterpolar(
                r=[subdb[col] for col in star_cols], line={'color': RATING_COLORS[e_idx]}, fillcolor=rgb_to_rgba(RATING_COLORS[e_idx], 0.1),
                theta=star_cols_short, fill='toself', name=env, showlegend=idx==0
            ), row=1, col=idx+1)
    fig.update_annotations(yshift=20)
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)), width=PLOT_WIDTH, height=PLOT_HEIGHT,
        legend=dict( yanchor="top", y=-0.1, xanchor="center", x=0.5, orientation='h'), margin={'l': 0, 'r': 0, 'b': 50, 't': 40}
    )
    # fig.show()
    fig.write_image(f'star_differences.pdf')
    
    # ERRORS ACROSS PROPERTIES
    traces, titles = [], []
    for idx, (prop, prop_meta) in enumerate(meta['properties'].items()):
        row, col = 2 if idx >= len(meta['properties']) / 2 else 1, int(idx % (len(meta['properties']) / 2)) + 1
        for e_idx, (env_info, scale) in enumerate( [('use_env', 'rec_index'), ('use_env', 'value')] ):
            res = meta_results['combined'].xs(f'{env_info}__{scale}', level=0, axis=1)[f'{prop}_test_err']
            _, to_unit = formatter.reformat_value(res.iloc[0], prop_meta['unit'])
            trace, color = ('Index', RATING_COLORS[0]) if scale == 'rec_index' else ('Value', RATING_COLORS[4])
            reformatted = res.abs().map(lambda val: formatter.reformat_value(val, prop_meta['unit'], unit_to=to_unit, as_str=False))
            traces.append( (row, col, go.Box(name=trace, y=reformatted, legendgroup=trace, showlegend=idx==0, marker_color=color)) )
            if e_idx == 0:
                titles.append(f"{prop_meta['shortname']} {to_unit}")
    fig = make_subplots(rows=2, cols=int(len(meta['properties']) / 2), subplot_titles=titles, vertical_spacing=0.1)
    for row, col, trace in traces:
        fig.add_trace(trace, row=row, col=col)
        fig.update_xaxes(visible=False, showticklabels=False, row=row, col=col)
    fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 18},
                      legend=dict(title='Measurement scale:', orientation="h", yanchor="top", y=0, xanchor="center", x=0.5))
    # fig.show()
    fig.write_image('errors_across_properties.pdf')



    ####### DS EMBEDDING
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f'{name} (N={vals.shape[1]})' for name, vals in all_meta_features.items()])
    for idx, (name, values) in enumerate( all_meta_features.items() ):
        dropped = values.dropna()
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(dropped)
        if idx == 0:
            min_x, max_x, min_y, max_y = embedding[:,0].min(), embedding[:,0].max(), embedding[:,1].min(), embedding[:,1].max()
        colors, sizes = zip(*[(all_meta_features['statistical'].loc[ds,'n_predictors'], all_meta_features['statistical'].loc[ds,'n_instances']) for ds in dropped.index])
        fig.add_trace( go.Scatter(x=embedding[:,0], y=embedding[:,1], mode='markers', showlegend=False, marker={'color': np.log(colors), 'size': np.log(sizes), 'coloraxis': 'coloraxis', 'sizemin': 1}), row=1, col=idx+1)
    for idx, n in enumerate([int(min(list(sizes))), 1000, 10000, int(max(list(sizes)))]):
        name = f'{n} {"(min)" if idx==0 else ("(max)" if idx==3 else "")}' 
        fig.add_trace( go.Scatter(x=[-100], y=[-100], mode='markers', marker={'color': [1], 'size': [np.log(n)], 'colorscale': RATING_COLOR_SCALE, 'sizemin':1}, name=name), row=1, col=1)
    fig.update_xaxes(range=[min_x - 1, max_x + 1], row=1, col=1)
    fig.update_yaxes(range=[min_y - 1, max_y + 1], row=1, col=1)
    bar_ticks = [int(min(list(colors))), 100, 1000, 10000, int(max(list(colors)))]
    bar_text = [f'{n} {"(min)" if idx==0 else ("(max)" if idx==4 else "")}' for idx, n in enumerate(bar_ticks)]
    fig.update_layout(coloraxis={'colorscale': RATING_COLOR_SCALE, 'colorbar': {'title': '# Features', 'len': 0.75, 'y': 0.3, 'tickvals': np.log(bar_ticks), 'ticktext': bar_text}},
                      legend=dict(title='# Instances', yanchor="top", y=1.08),
                      width=PLOT_WIDTH, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 18})
    # fig.show()
    fig.write_image('ds_embeddings.pdf')

    ####### SCATTER landscapes for a single data set
    ds_sub_db = rated_db[rated_db['dataset'] == DS_SEL]
    for scale, scale_name in zip(['index', 'value'], ['Index Scale', 'Value Scale']):
        plot_data, axis_names, rating_pos = assemble_scatter_data(pd.unique(ds_sub_db['environment']).tolist(), ds_sub_db, scale, 'running_time', 'accuracy', meta, bounds)
        scatter = create_scatter_graph(plot_data, axis_names, dark_mode=False, marker_width=8)
        add_rating_background(scatter, None, None if scale == 'index' else 'FLIP_LEFT_RIGHT')
        scatter.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
                              legend=dict(yanchor="bottom", y=0.05, xanchor="center", x=0.5), showlegend=scale=='value')
        scatter.show()
        scatter.write_image(f"scatter_{scale}.pdf")

    # ##### VIOLIN of standard devs across environments
    fig = go.Figure()
    for col, std_devs in avg_dist.items():
        fig.add_trace( go.Violin(y=std_devs, x=[col] * len(std_devs), spanmode='hard', meanline_visible=True, showlegend=False) )
    fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
    # fig.show()
    fig.write_image('property_std_distrib.pdf')

    


###### TODO

    # IMPACT OF ENVIRONMENT FEATURES
    # IMPACT OF INDEX SCALING
    # => on PREDICTION QUALITY
